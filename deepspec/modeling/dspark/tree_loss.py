"""Inference-aligned tree objectives for DSpark training.

At inference time the drafter expands, per depth ``d``, the top-k children of
every frontier parent under the parent-conditioned distribution
``z_d(p) = L_d + MarkovBias(p)`` and prunes the frontier by joint path
log-probability (see ``build_dspark_tree_proposal`` in the serving branches).
Standard teacher-forced CE/TV training never sees this object: it supervises
a single ground-truth chain, so the Markov head is never conditioned on the
model's own drafted prefixes, and the loss spends capacity matching the full
target distribution instead of covering the target probability mass that
tree verification can actually accept.

This module replays the exact inference tree expansion inside the training
forward (topology is stop-gradient): per-parent top-k expansion, joint path
log-prob frontier pruning, and the global top-N (``--tree-max-nodes``)
packing that selects which nodes the target model actually verifies.  On
top of the realized packed tree it builds two complementary objectives:

* Target-mass coverage (ranking hinge): for every frontier parent, the
  target's top-J tokens (by ``p_T``) are pushed above the k-th draft logit
  with a margin: ``p_T(i) * relu(margin - z_i + z_kth)``.  This optimizes
  Mass@k recall directly, without forcing the draft to copy the target's
  ordering.
* Tree survival NLL: every candidate node is weighted by its target path
  mass ``m(node) = p_T(v_node) * m(parent)`` (detached) and supervised with
  its conditional log-prob, ``-sum m(node) * log q_theta(v_node | parent)``.
  This is the differentiable surrogate of maximizing how deep the target
  probability mass survives inside the draft tree; it also trains the
  Markov head on the model's own drafted prefixes, removing the
  teacher-forcing exposure bias at exactly the nodes the tree visits.

Both objectives are additive to the original CE/TV losses; TV is kept
because of its direct relation to the speculative acceptance rate.

Memory control: only ``num_anchors`` randomly subsampled anchor blocks take
part, and each depth's ``[n, k, vocab]`` logit slab is wrapped in
``torch.utils.checkpoint`` so it is recomputed during backward instead of
being retained.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from deepspec.modeling.dspark.markov_head import RNNHead
from deepspec.utils.metrics import add_metric


@dataclass
class DSparkTreeLossInputs:
    """Packed per-depth tensors of the simulated draft tree.

    All lists are indexed by draft depth ``d = 0..block_size-1``.  Candidate
    nodes at depth ``d`` are the flattened ``(frontier_parent, top-k child)``
    pairs, mirroring inference-time tree expansion; every topology-dependent
    quantity (which candidates exist, frontier pruning, path masses) is
    stop-gradient.
    """

    # Differentiable conditional log-prob log q_theta(v | parent) per node,
    # [bsz, num_sub, fanout_d] float32.
    node_logprobs: list = field(default_factory=list)
    # Detached target path mass per node, same shape, float32.
    node_masses: list = field(default_factory=list)
    # Node validity (position in-seq, loss-enabled, kept anchor), same shape.
    node_valids: list = field(default_factory=list)
    # Differentiable target-mass ranking hinge summed over target top-J,
    # [bsz, num_sub, frontier_d] float32.
    rank_hinges: list = field(default_factory=list)
    # Detached per-parent rank normalization weight (sum of p_T over top-J),
    # same shape, float32.
    rank_weights: list = field(default_factory=list)
    # Per-depth anchor validity, [bsz, num_sub] bool.
    depth_valids: list = field(default_factory=list)
    # Inference packing mask: True for candidates selected by the global
    # top-N (``max_nodes``) tree packing, same shape as node_masses; entries
    # are None when max_nodes <= 0 (packing disabled).
    node_packed_masks: list = field(default_factory=list)
    # Greedy-verification proxy: True when the target-argmax chain still
    # survives inside the packed tree at this depth, [bsz, num_sub] bool.
    argmax_survived: list = field(default_factory=list)


def _subsample_anchor_indices(
    *,
    block_keep_mask: torch.Tensor,
    num_sub: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Uniformly subsample up to ``num_sub`` kept anchors per sample."""
    bsz, num_anchors = block_keep_mask.shape
    device = block_keep_mask.device
    random_vals = torch.rand(bsz, num_anchors, device=device)
    random_vals = torch.where(
        block_keep_mask,
        random_vals,
        torch.full_like(random_vals, 2.0),
    )
    # The num_sub smallest random values are a uniform without-replacement
    # subsample; topk (float32) stays on AiCore, unlike integer argsort.
    selected = random_vals.topk(num_sub, dim=1, largest=False).indices
    keep_counts = block_keep_mask.sum(dim=1).clamp(max=num_sub)
    selected_valid = torch.arange(num_sub, device=device).unsqueeze(0) < (
        keep_counts.unsqueeze(1)
    )
    selected = torch.where(selected_valid, selected, torch.zeros_like(selected))
    return selected, selected_valid


def _tree_depth_segment(
    base_logits_d: torch.Tensor,
    hidden_d: torch.Tensor,
    frontier_tokens: torch.Tensor,
    frontier_states: Optional[torch.Tensor],
    topj_ids: torch.Tensor,
    topj_probs: torch.Tensor,
    *,
    markov_head,
    use_rnn: bool,
    top_k: int,
    margin: float,
    hinge_cap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One tree depth: parent-conditioned logits, top-k expansion, rank hinge.

    Runs inside ``torch.utils.checkpoint`` so the ``[n, F, vocab]`` slabs are
    recomputed in backward.  ``frontier_tokens``/``frontier_states`` and the
    target-side ``topj_*`` tensors are stop-gradient inputs.
    """
    bsz, num_sub, fanout = frontier_tokens.shape
    step_hidden = hidden_d.unsqueeze(2).expand(
        bsz, num_sub, fanout, hidden_d.size(-1)
    )
    if markov_head is None:
        z = base_logits_d.unsqueeze(2).expand(
            bsz, num_sub, fanout, base_logits_d.size(-1)
        )
        new_states = frontier_tokens.new_zeros((0,), dtype=torch.float32)
    elif use_rnn:
        prev_embeddings = markov_head.get_prev_embeddings(frontier_tokens)
        new_states, bias = markov_head._rnn_step(
            frontier_states,
            prev_embeddings,
            step_hidden.to(dtype=prev_embeddings.dtype),
        )
        z = base_logits_d.unsqueeze(2) + bias
    else:
        bias = markov_head.compute_step_bias(frontier_tokens, step_hidden)
        z = base_logits_d.unsqueeze(2) + bias
        new_states = frontier_tokens.new_zeros((0,), dtype=torch.float32)

    z_float = z.float()
    top_vals_z, top_ids = z_float.topk(top_k, dim=-1)
    log_q = F.log_softmax(z_float, dim=-1)
    # Differentiable log q_theta(child | parent) for each expanded child.
    node_logprobs = log_q.gather(-1, top_ids).reshape(bsz, num_sub, fanout * top_k)

    # Target-mass ranking hinge: push target-important tokens above the k-th
    # draft logit with a margin (no-op once they are comfortably inside top-k).
    # The hinge is capped so the loss stays bounded and gradient concentrates
    # on boundary-winnable tokens: with raw logit gaps growing over training
    # (weight_decay=0), an uncapped hinge would keep growing in value and
    # spend constant gradient on tokens hopelessly far from the top-k edge.
    z_kth = top_vals_z[..., -1:]
    num_topj = topj_ids.size(-1)
    gather_ids = topj_ids.unsqueeze(2).expand(bsz, num_sub, fanout, num_topj)
    z_topj = z_float.gather(-1, gather_ids)
    hinge = (margin - z_topj + z_kth).clamp_min(0.0)
    if hinge_cap > 0:
        hinge = hinge.clamp_max(float(hinge_cap))
    rank_hinge = (hinge * topj_probs.unsqueeze(2)).sum(dim=-1)

    return node_logprobs, top_ids.reshape(bsz, num_sub, fanout * top_k), (
        rank_hinge
    ), new_states


@torch.no_grad()
def _target_topj(
    target_logits_d: torch.Tensor,
    top_j: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Target distribution at one depth: full probs plus its top-J entries."""
    target_probs = F.softmax(target_logits_d.float(), dim=-1)
    topj_probs, topj_ids = target_probs.topk(top_j, dim=-1)
    return target_probs, topj_probs, topj_ids


def build_tree_loss_inputs(
    *,
    markov_head,
    base_logits: torch.Tensor,
    hidden_states: torch.Tensor,
    aligned_target_logits: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    eval_mask: torch.Tensor,
    block_keep_mask: torch.Tensor,
    top_k: int,
    top_j: int,
    margin: float,
    num_anchors_sub: int,
    max_nodes: int = 0,
    hinge_cap: float = 4.0,
    grad_checkpoint: bool = True,
) -> Optional[DSparkTreeLossInputs]:
    """Replay the inference-time draft tree on subsampled anchors.

    Args:
        markov_head: the model's Markov head (vanilla/gated/rnn) or None.
        base_logits: [bsz, num_anchors, block_size, vocab] pre-bias draft logits.
        hidden_states: [bsz, num_anchors, block_size, hidden] backbone output.
        aligned_target_logits: [bsz, num_anchors, block_size, vocab] detached
            teacher logits at the supervised positions.
        anchor_token_ids: [bsz, num_anchors] verified anchor tokens.
        eval_mask: [bsz, num_anchors, block_size] bool supervision mask.
        block_keep_mask: [bsz, num_anchors] bool real-anchor mask.
        top_k: inference tree branching factor (frontier width cap).
        top_j: number of target top tokens used by the ranking hinge.
        margin: ranking hinge margin in logit units.
        num_anchors_sub: anchors subsampled for the tree objectives.
        max_nodes: inference tree packing budget (global top-N over the
            candidate pool, ``--tree-max-nodes``); 0 disables packing and
            keeps the whole pool.
        hinge_cap: cap on the ranking hinge (logit units); 0 disables the
            cap.  Keeps the rank loss bounded and its gradient focused on
            boundary-winnable tokens.
        grad_checkpoint: recompute each depth's logits during backward.
    """
    bsz, num_anchors, block_size, vocab_size = base_logits.shape
    device = base_logits.device
    num_sub = min(int(num_anchors_sub), int(num_anchors))
    if num_sub <= 0 or block_size == 0:
        return None
    top_k = max(1, min(int(top_k), int(vocab_size)))
    top_j = max(1, min(int(top_j), int(vocab_size)))
    use_rnn = isinstance(markov_head, RNNHead)

    selected, selected_valid = _subsample_anchor_indices(
        block_keep_mask=block_keep_mask,
        num_sub=num_sub,
    )
    batch_idx = torch.arange(bsz, device=device).unsqueeze(1)
    base_sub = base_logits[batch_idx, selected]
    hidden_sub = hidden_states[batch_idx, selected]
    eval_mask_sub = eval_mask[batch_idx, selected]
    anchor_tokens_sub = anchor_token_ids[batch_idx, selected]

    # Frontier state (all stop-gradient): the verified anchor token is the
    # single root parent with unit target mass and zero joint log-prob.
    frontier_tokens = anchor_tokens_sub.unsqueeze(-1)
    frontier_scores = torch.zeros(bsz, num_sub, 1, device=device)
    frontier_mass = torch.ones(bsz, num_sub, 1, device=device)
    # Pool-parent pointer per frontier node (-1 = child of the root anchor),
    # mirroring ``frontier_pool`` in the inference tree builder.
    frontier_pool = torch.full(
        (bsz, num_sub, 1), -1, dtype=torch.long, device=device
    )
    frontier_states = None
    if use_rnn:
        frontier_states = torch.zeros(
            bsz,
            num_sub,
            1,
            markov_head.markov_rank,
            dtype=hidden_sub.dtype,
            device=device,
        )

    inputs = DSparkTreeLossInputs()
    pool_tokens_all = []
    pool_scores_all = []
    pool_depths_all = []
    pool_parents_all = []
    target_argmax_all = []
    pool_offset = 0
    for depth in range(block_size):
        fanout = frontier_tokens.size(-1)
        # Target logits are sliced per depth (transient [B, n, V]) instead of
        # materializing a retained [B, n, block_size, V] subsample copy.
        target_probs, topj_probs, topj_ids = _target_topj(
            aligned_target_logits[batch_idx, selected, depth],
            top_j,
        )
        segment_args = (
            base_sub[:, :, depth, :],
            hidden_sub[:, :, depth, :],
            frontier_tokens,
            frontier_states,
            topj_ids,
            topj_probs,
        )
        segment_kwargs = dict(
            markov_head=markov_head,
            use_rnn=use_rnn,
            top_k=top_k,
            margin=margin,
            hinge_cap=hinge_cap,
        )
        if grad_checkpoint and torch.is_grad_enabled():
            node_logprobs, flat_ids, rank_hinge, new_states = checkpoint(
                _tree_depth_segment,
                *segment_args,
                **segment_kwargs,
                use_reentrant=False,
            )
        else:
            node_logprobs, flat_ids, rank_hinge, new_states = _tree_depth_segment(
                *segment_args,
                **segment_kwargs,
            )

        depth_valid = eval_mask_sub[:, :, depth] & selected_valid
        inputs.node_logprobs.append(node_logprobs)
        inputs.rank_hinges.append(rank_hinge)
        inputs.rank_weights.append(
            topj_probs.sum(dim=-1).unsqueeze(-1).expand(-1, -1, fanout)
        )
        inputs.depth_valids.append(depth_valid)

        with torch.no_grad():
            # Target path mass of every candidate: m(child) = p_T(child) * m(parent).
            cand_mass = target_probs.gather(-1, flat_ids)
            parent_mass = frontier_mass.repeat_interleave(top_k, dim=-1)
            node_mass = cand_mass * parent_mass
            num_candidates = fanout * top_k
            inputs.node_masses.append(node_mass)
            inputs.node_valids.append(
                depth_valid.unsqueeze(-1).expand(-1, -1, num_candidates)
            )
            target_argmax_all.append(target_probs.argmax(dim=-1))

            # Inference frontier pruning: joint path log-prob top-k over all
            # (parent, child) candidates; frontier width never exceeds top_k.
            cand_scores = frontier_scores.unsqueeze(-1) + node_logprobs.detach().view(
                bsz, num_sub, fanout, top_k
            )
            cand_scores = cand_scores.reshape(bsz, num_sub, num_candidates)

            # Candidate-pool bookkeeping for the global top-N packing and the
            # greedy acceptance walk (same layout as the inference builder).
            pool_tokens_all.append(flat_ids)
            pool_scores_all.append(cand_scores)
            pool_depths_all.append(
                torch.full(
                    (bsz, num_sub, num_candidates),
                    depth,
                    dtype=torch.long,
                    device=device,
                )
            )
            pool_parents_all.append(frontier_pool.repeat_interleave(top_k, dim=-1))

            keep = min(top_k, num_candidates)
            sel_scores, sel_idx = cand_scores.topk(keep, dim=-1)
            frontier_tokens = flat_ids.gather(-1, sel_idx)
            frontier_scores = sel_scores
            frontier_mass = node_mass.gather(-1, sel_idx)
            frontier_pool = pool_offset + sel_idx
            if use_rnn:
                parent_idx = (sel_idx // top_k).unsqueeze(-1).expand(
                    -1, -1, -1, new_states.size(-1)
                )
                frontier_states = new_states.detach().gather(2, parent_idx)
            pool_offset += num_candidates
        del target_probs

    _pack_tree_and_walk_argmax(
        inputs=inputs,
        pool_tokens=pool_tokens_all,
        pool_scores=pool_scores_all,
        pool_depths=pool_depths_all,
        pool_parents=pool_parents_all,
        target_argmax=target_argmax_all,
        max_nodes=int(max_nodes),
    )
    return inputs


@torch.no_grad()
def _pack_tree_and_walk_argmax(
    *,
    inputs: DSparkTreeLossInputs,
    pool_tokens: list[torch.Tensor],
    pool_scores: list[torch.Tensor],
    pool_depths: list[torch.Tensor],
    pool_parents: list[torch.Tensor],
    target_argmax: list[torch.Tensor],
    max_nodes: int,
) -> None:
    """Apply inference tree packing and replay the greedy acceptance walk.

    Packing mirrors the inference builder exactly: sort the whole candidate
    pool by joint score (desc) with ties broken toward shallower nodes
    (two-level stable argsort), keep the top ``max_nodes`` — the only nodes
    the target model ever verifies at serving time.

    The acceptance walk mirrors greedy tree verification: start at the root
    anchor and repeatedly follow the packed child whose token matches the
    target argmax; the chain breaks at the first depth without such a child.
    """
    scores = torch.cat(pool_scores, dim=-1)  # [B, n, P]
    depths = torch.cat(pool_depths, dim=-1)
    tokens = torch.cat(pool_tokens, dim=-1)
    parents = torch.cat(pool_parents, dim=-1)
    bsz, num_sub, num_pool = scores.shape

    if max_nodes > 0:
        num_selected = min(int(max_nodes), num_pool)
        # Only the selected *set* matters for the mask, so a float32 topk
        # suffices — it stays on AiCore, whereas the inference builder's
        # two-level (score, depth) stable argsort on int64 depths falls back
        # to AiCPU on Ascend and can stall the stream under memory pressure.
        # Selection differs from inference only on exact score ties, which
        # have measure zero for continuous log-probs.
        packed_idx = scores.topk(num_selected, dim=-1).indices
        packed_mask = torch.zeros_like(depths, dtype=torch.bool)
        packed_mask.scatter_(-1, packed_idx, True)
    else:
        packed_mask = torch.ones_like(depths, dtype=torch.bool)

    # Greedy acceptance walk over the packed tree.
    argmax_prev = torch.full(
        (bsz, num_sub), -1, dtype=torch.long, device=scores.device
    )
    alive = torch.ones(bsz, num_sub, dtype=torch.bool, device=scores.device)
    pool_offset = 0
    for depth, depth_candidates in enumerate(pool_tokens):
        num_candidates = depth_candidates.size(-1)
        sl = slice(pool_offset, pool_offset + num_candidates)
        parents_d = parents[..., sl]
        tokens_d = tokens[..., sl]
        packed_d = packed_mask[..., sl]
        inputs.node_packed_masks.append(
            packed_d if max_nodes > 0 else None
        )
        match = (
            alive.unsqueeze(-1)
            & (parents_d == argmax_prev.unsqueeze(-1))
            & (tokens_d == target_argmax[depth].unsqueeze(-1))
            & packed_d
        )
        survived = match.any(dim=-1)
        # Children of one parent have distinct tokens (top-k of one
        # distribution), so at most one candidate matches; argmax is safe.
        # Float argmax stays on AiCore (integer sorts/argmax risk AiCPU
        # fallback on Ascend).
        match_idx = match.float().argmax(dim=-1)
        argmax_prev = torch.where(
            survived,
            torch.full_like(argmax_prev, pool_offset) + match_idx,
            argmax_prev,
        )
        alive = alive & survived
        inputs.argmax_survived.append(alive.clone())
        pool_offset += num_candidates


def compute_tree_loss_terms(
    *,
    tree_inputs: DSparkTreeLossInputs,
    loss_decay_gamma: Optional[float],
) -> dict[str, torch.Tensor]:
    """Reduce packed tree tensors to loss numerators/denominators.

    The tree NLL normalizes by the total surviving target mass, so its scale
    stays comparable to a ordinary token NLL.  The ranking hinge normalizes
    by the total target top-J mass.  Only candidates that survive the
    inference top-N packing are supervised/measured — nodes outside the
    packed tree are never verified at serving time.

    Per-depth metrics:
    * ``tree_mass_coverage@{d}``: S_d, target path probability still covered
      by the packed tree at depth d (``@0`` is Mass@k at the tree root).
    * ``tree_argmax_survival@{d}``: fraction of anchors whose target-argmax
      chain is still walkable inside the packed tree at depth d — the direct
      training-side proxy of greedy tree-verification acceptance length.
    """
    reference = tree_inputs.node_logprobs[0]
    zero = reference.new_zeros((), dtype=torch.float32)
    nll_num = zero.clone()
    nll_den = zero.clone()
    rank_num = zero.clone()
    rank_den = zero.clone()
    for depth, node_logprobs in enumerate(tree_inputs.node_logprobs):
        node_valid = tree_inputs.node_valids[depth]
        packed_mask = tree_inputs.node_packed_masks[depth]
        if packed_mask is not None:
            node_valid = node_valid & packed_mask
        node_valid = node_valid.to(torch.float32)
        node_mass = tree_inputs.node_masses[depth]
        nll_num = nll_num + (-(node_logprobs * node_mass * node_valid).sum())
        nll_den = nll_den + (node_mass * node_valid).sum()
        depth_valid_f = tree_inputs.depth_valids[depth].to(torch.float32)
        with torch.no_grad():
            coverage_num = (node_mass * node_valid).sum()
            coverage_den = depth_valid_f.sum()
            argmax_num = (
                tree_inputs.argmax_survived[depth].to(torch.float32) * depth_valid_f
            ).sum()
        add_metric(
            f"tree_mass_coverage@{depth}",
            coverage_num,
            den=coverage_den,
            tag="train",
        )
        add_metric(
            f"tree_argmax_survival@{depth}",
            argmax_num,
            den=coverage_den,
            tag="train",
        )

        depth_weight = depth_valid_f
        if loss_decay_gamma is not None and loss_decay_gamma > 0:
            depth_weight = depth_weight * float(
                math.exp(-float(depth) / float(loss_decay_gamma))
            )
        rank_valid = depth_weight.unsqueeze(-1)
        rank_num = rank_num + (tree_inputs.rank_hinges[depth] * rank_valid).sum()
        rank_den = rank_den + (tree_inputs.rank_weights[depth] * rank_valid).sum()

    return {
        "tree_nll_loss_num": nll_num,
        "tree_nll_loss_den": nll_den,
        "tree_rank_loss_num": rank_num,
        "tree_rank_loss_den": rank_den,
    }


__all__ = [
    "DSparkTreeLossInputs",
    "build_tree_loss_inputs",
    "compute_tree_loss_terms",
]
