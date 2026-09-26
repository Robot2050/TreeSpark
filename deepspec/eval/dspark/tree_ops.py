from __future__ import annotations

from dataclasses import dataclass

import torch

from deepspec.eval.dspark.draft_ops import DSparkDraftProposal
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.markov_head import RNNHead
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel


DSparkModel = Qwen3DSparkModel | Gemma4DSparkModel


@dataclass
class DSparkTree:
    """Packed draft-tree proposal payload.

    Layout of ``verify_input_ids``: ``[anchor, node_0, ..., node_{N-1}]`` with
    the verified anchor token at packed index 0; non-root node ``i`` sits at
    packed index ``1 + i``.  ``parent_indices[i]`` is the packed index of the
    node's parent (0 for children of the anchor).  ``depths[i]`` is the node's
    zero-based draft depth, so the node's absolute sequence position is
    ``start + depths[i] + 1`` and same-depth siblings share a position id.
    """

    tokens: torch.Tensor  # [N] non-root node tokens
    parent_indices: torch.Tensor  # [N] packed parent index per non-root node
    depths: torch.Tensor  # [N] zero-based draft depth per node


def _frontier_step_logits(
    model: DSparkModel,
    *,
    depth_logits: torch.Tensor,
    depth_hidden: torch.Tensor,
    frontier_tokens: torch.Tensor,
    frontier_states: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Per-parent logits for one depth: ``z_d(p) = L_d + Markov(p)``.

    ``depth_logits`` is the shared backbone prediction ``L_d`` ([V]) and
    ``depth_hidden`` the shared backbone hidden state ([H]); both are
    broadcast over the ``b`` frontier parents.  Only the cheap Markov head is
    re-evaluated per parent.  Returns the corrected logits ([b, V]) and, for
    the RNN head, the per-parent post-step states ([b, r]).
    """
    head = model.markov_head
    batch = frontier_tokens.size(0)
    if head is None:
        return depth_logits.unsqueeze(0).expand(batch, -1), None
    step_hidden = depth_hidden.unsqueeze(0).expand(batch, -1)
    if isinstance(head, RNNHead):
        # The RNN state summarizes the node's drafted prefix, so every node
        # carries its own state; children of the same parent share the
        # parent's post-step state (it does not depend on the child token).
        prev_embeddings = head.get_prev_embeddings(frontier_tokens)
        new_states, bias = head._rnn_step(
            frontier_states,
            prev_embeddings,
            step_hidden.to(dtype=prev_embeddings.dtype),
        )
        return depth_logits.unsqueeze(0) + bias, new_states
    bias = head.compute_step_bias(frontier_tokens, step_hidden)
    return depth_logits.unsqueeze(0) + bias, None


def build_dspark_tree_proposal(
    model: DSparkModel,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    top_k: int,
    max_nodes: int,
) -> DSparkDraftProposal:
    """Build a per-parent top-k draft tree from one backbone forward.

    All depths share the single backbone output ``L_d``; depth ``d`` re-scores
    the shared logits with each frontier parent's Markov bias and expands
    every parent into its local top-k children.  Two-level pruning keeps the
    tree small: the active frontier is cut to the top-k candidates per depth
    (joint path log-prob), and a final global top-N over the whole candidate
    pool selects the nodes packed for verification.  Because a child's joint
    score never exceeds its parent's and ties resolve toward the shallower
    node, the global top-N is prefix-closed by construction.
    """
    assert draft_input_ids.size(0) == 1, "tree proposals require batch_size=1"
    assert top_k >= 2, "build_dspark_tree_proposal requires top_k >= 2"
    assert max_nodes >= 1, "build_dspark_tree_proposal requires max_nodes >= 1"
    device = block_hidden.device
    head = model.markov_head
    use_rnn = isinstance(head, RNNHead)

    base_logits = model.compute_logits(block_hidden[:, :block_size, :])[0]
    base_hidden = block_hidden[0, :block_size, :]
    vocab_size = base_logits.size(-1)
    k = max(1, min(int(top_k), int(vocab_size)))

    pool_tokens = torch.empty(0, dtype=torch.long, device=device)
    pool_scores = torch.empty(0, dtype=torch.float32, device=device)
    pool_parents = torch.empty(0, dtype=torch.long, device=device)
    pool_depths = torch.empty(0, dtype=torch.long, device=device)
    pool_states: torch.Tensor | None = None

    # The frontier starts at the verified anchor token (pool parent -1).
    frontier_tokens = draft_input_ids[:, 0]
    frontier_scores = torch.zeros(1, dtype=torch.float32, device=device)
    frontier_pool = torch.full((1,), -1, dtype=torch.long, device=device)
    frontier_states = None
    if use_rnn:
        frontier_states = torch.zeros(
            1,
            head.markov_rank,
            dtype=base_hidden.dtype,
            device=device,
        )

    for depth in range(block_size):
        batch = frontier_tokens.size(0)
        step_logits, new_states = _frontier_step_logits(
            model,
            depth_logits=base_logits[depth],
            depth_hidden=base_hidden[depth],
            frontier_tokens=frontier_tokens,
            frontier_states=frontier_states,
        )
        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
        top_vals, top_ids = log_probs.topk(k, dim=-1)
        candidate_scores = frontier_scores.unsqueeze(1) + top_vals
        num_candidates = batch * k

        pool_parents = torch.cat([pool_parents, frontier_pool.repeat_interleave(k)])
        pool_tokens = torch.cat([pool_tokens, top_ids.reshape(-1)])
        pool_scores = torch.cat([pool_scores, candidate_scores.reshape(-1)])
        pool_depths = torch.cat(
            [
                pool_depths,
                torch.full(
                    (num_candidates,), depth, dtype=torch.long, device=device
                ),
            ]
        )
        if use_rnn:
            candidate_states = new_states.repeat_interleave(k, dim=0)
            pool_states = (
                candidate_states
                if pool_states is None
                else torch.cat([pool_states, candidate_states], dim=0)
            )

        # Intra-layer pruning: only the top-k candidates of this depth stay
        # in the active frontier, so frontier width never exceeds k.
        keep = min(k, num_candidates)
        selected = candidate_scores.reshape(-1).topk(keep).indices
        frontier_tokens = top_ids.reshape(-1)[selected]
        frontier_scores = candidate_scores.reshape(-1)[selected]
        frontier_pool = (pool_tokens.size(0) - num_candidates) + selected
        if use_rnn:
            frontier_states = new_states[selected // k]

    # Global top-N: sort by score desc with ties broken toward shallower
    # nodes; any prefix of this order is prefix-closed.
    num_pool = int(pool_tokens.size(0))
    num_selected = min(int(max_nodes), num_pool)
    order = torch.argsort(pool_depths, stable=True)
    order = order[torch.argsort(pool_scores[order], descending=True, stable=True)]
    selected = order[:num_selected]
    # The pool is created depth by depth, so ascending pool indices give a
    # deterministic parent-before-child packed layout.
    packed = selected.sort().values
    num_nodes = int(packed.size(0))

    remap = torch.zeros(num_pool, dtype=torch.long, device=device)
    remap[packed] = torch.arange(1, num_nodes + 1, dtype=torch.long, device=device)
    pool_parent_packed = pool_parents[packed]
    non_root = pool_parent_packed >= 0
    # Prefix-closure holds by construction (parents always sort before their
    # children), so every remap lookup above hits a selected node; no runtime
    # check is kept on the hot path since it would force a host sync per round.
    parent_indices = torch.zeros(num_nodes, dtype=torch.long, device=device)
    parent_indices[non_root] = remap[pool_parent_packed[non_root]]

    tree = DSparkTree(
        tokens=pool_tokens[packed],
        parent_indices=parent_indices,
        depths=pool_depths[packed],
    )
    verify_input_ids = torch.cat(
        [draft_input_ids[:, :1], tree.tokens.unsqueeze(0)],
        dim=1,
    )
    return DSparkDraftProposal(
        draft_token_count=num_nodes,
        verify_input_ids=verify_input_ids,
        draft_probs=None,
        confidence_logits=None,
        tree=tree,
    )


def build_dspark_tree_proposals_batched(
    model: DSparkModel,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    top_k: int,
    max_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slot-batched version of ``build_dspark_tree_proposal``.

    Per-slot frontier widths are content-independent (1 at depth 0, ``k``
    afterwards: every depth expands <=k parents into k children and prunes
    back to k), so the whole frontier loop vectorizes over the slot batch.
    One call replaces the previous per-slot ``batch_size=1`` calls, which
    cost the engine B x block_size serialized vocab-wide kernel launches per
    round and starved the accelerator at high concurrency.

    Returns ``(tokens, parent_indices, depths)`` — each a ``[B, N]`` device
    tensor in the same packed layout as ``DSparkTree`` (node ``i`` sits at
    packed index ``1 + i``), row-wise identical to what the per-slot function
    builds for each slot.  ``N = min(max_nodes, pool_size)`` is uniform
    across slots because the candidate-pool shape is content-independent.
    The per-slot version's ``pool_states`` bookkeeping is dropped here: it
    was accumulated every depth but never read.
    """
    assert top_k >= 2, "build_dspark_tree_proposals_batched requires top_k >= 2"
    assert max_nodes >= 1, "build_dspark_tree_proposals_batched requires max_nodes >= 1"
    device = block_hidden.device
    head = model.markov_head
    use_rnn = isinstance(head, RNNHead)

    batch_size = int(draft_input_ids.size(0))
    base_logits = model.compute_logits(block_hidden[:, :block_size, :])  # [B, bs, V]
    base_hidden = block_hidden[:, :block_size, :]
    vocab_size = int(base_logits.size(-1))
    k = max(1, min(int(top_k), vocab_size))

    pool_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=device)
    pool_scores = torch.empty(batch_size, 0, dtype=torch.float32, device=device)
    pool_parents = torch.empty(batch_size, 0, dtype=torch.long, device=device)
    pool_depths = torch.empty(batch_size, 0, dtype=torch.long, device=device)

    # The frontier starts at each slot's verified anchor token (pool parent -1).
    frontier_tokens = draft_input_ids[:, :1]  # [B, 1]
    frontier_scores = torch.zeros(batch_size, 1, dtype=torch.float32, device=device)
    frontier_pool = torch.full((batch_size, 1), -1, dtype=torch.long, device=device)
    frontier_states = (
        torch.zeros(
            batch_size, 1, head.markov_rank, dtype=base_hidden.dtype, device=device
        )
        if use_rnn
        else None
    )

    for depth in range(block_size):
        width = int(frontier_tokens.size(1))  # 1 at depth 0, k afterwards
        flat = batch_size * width
        # Shared backbone prediction L_d broadcast over the frontier parents;
        # only the cheap Markov head is re-evaluated per parent.
        step_logits = (
            base_logits[:, depth, :]
            .unsqueeze(1)
            .expand(batch_size, width, vocab_size)
            .reshape(flat, vocab_size)
        )
        new_states = None
        if head is not None:
            step_hidden = (
                base_hidden[:, depth, :]
                .unsqueeze(1)
                .expand(batch_size, width, base_hidden.size(-1))
                .reshape(flat, -1)
            )
            flat_tokens = frontier_tokens.reshape(flat)
            if use_rnn:
                prev_embeddings = head.get_prev_embeddings(flat_tokens)
                new_states, bias = head._rnn_step(
                    frontier_states.reshape(flat, -1),
                    prev_embeddings,
                    step_hidden.to(dtype=prev_embeddings.dtype),
                )
            else:
                bias = head.compute_step_bias(flat_tokens, step_hidden)
            step_logits = step_logits + bias

        log_probs = torch.log_softmax(step_logits.float(), dim=-1)
        top_vals, top_ids = log_probs.topk(k, dim=-1)  # [flat, k]
        candidate_scores = frontier_scores.reshape(flat, 1) + top_vals
        num_candidates = width * k

        pool_parents = torch.cat(
            [pool_parents, frontier_pool.repeat_interleave(k, dim=1)], dim=1
        )
        pool_tokens = torch.cat(
            [pool_tokens, top_ids.reshape(batch_size, num_candidates)], dim=1
        )
        pool_scores = torch.cat(
            [pool_scores, candidate_scores.reshape(batch_size, num_candidates)], dim=1
        )
        pool_depths = torch.cat(
            [
                pool_depths,
                torch.full(
                    (batch_size, num_candidates),
                    depth,
                    dtype=torch.long,
                    device=device,
                ),
            ],
            dim=1,
        )

        # Intra-layer pruning: only the top-k candidates of this depth stay
        # in the active frontier, so frontier width never exceeds k.
        keep = min(k, num_candidates)
        selected = (
            candidate_scores.reshape(batch_size, num_candidates)
            .topk(keep, dim=1)
            .indices
        )  # [B, keep]
        frontier_tokens = top_ids.reshape(batch_size, num_candidates).gather(1, selected)
        frontier_scores = candidate_scores.reshape(batch_size, num_candidates).gather(
            1, selected
        )
        frontier_pool = (pool_tokens.size(1) - num_candidates) + selected
        if use_rnn:
            # Children of a parent share the parent's post-step state (it does
            # not depend on the child token).
            frontier_states = new_states.reshape(batch_size, width, -1).gather(
                1,
                (selected // k).unsqueeze(-1).expand(-1, -1, head.markov_rank),
            )

    # Global top-N per slot: sort by score desc with ties broken toward
    # shallower nodes; any prefix of this order is prefix-closed.
    num_pool = int(pool_tokens.size(1))
    num_selected = min(int(max_nodes), num_pool)
    order = pool_depths.argsort(dim=1, stable=True)
    order = order.gather(
        1,
        pool_scores.gather(1, order).argsort(dim=1, descending=True, stable=True),
    )
    selected = order[:, :num_selected]
    # The pool is created depth by depth, so ascending pool indices give a
    # deterministic parent-before-child packed layout.
    packed = selected.sort(dim=1).values  # [B, N]
    num_nodes = int(packed.size(1))

    remap = torch.zeros(batch_size, num_pool, dtype=torch.long, device=device)
    remap.scatter_(
        1,
        packed,
        torch.arange(1, num_nodes + 1, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1),
    )
    pool_parent_packed = pool_parents.gather(1, packed)
    non_root = pool_parent_packed >= 0
    # Prefix-closure holds by construction (parents always sort before their
    # children), so every clamped lookup for a non-root node hits a selected
    # parent; roots keep packed parent 0.
    remapped = remap.gather(1, pool_parent_packed.clamp(min=0))
    parent_indices = torch.where(non_root, remapped, torch.zeros_like(remapped))

    return (
        pool_tokens.gather(1, packed),
        parent_indices,
        pool_depths.gather(1, packed),
    )
