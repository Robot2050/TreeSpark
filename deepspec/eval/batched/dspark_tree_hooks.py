"""Batched DSpark *tree* drafting hooks for the high-concurrency engine.

Tree mode (``--tree-top-k > 1``) is greedy-only: all slots build their packed
draft trees from the shared block-parallel backbone forward in ONE slot-batched
call (``build_dspark_tree_proposals_batched`` — per-slot frontier widths are
content-independent, so the per-depth top-k loop vectorizes over the slot
batch; the previous per-slot bsz=1 calls serialized B x block_size vocab-wide
kernel launches per round), and the engine verifies all trees in a single
target forward with per-slot ancestor masks.  Chain mode (``DSparkBatchedHooks``)
and the baseline are untouched: this subclass only overrides ``propose`` and
adds ``on_verify_tree``.

Verification data is packed in a flattened-tree format: ``_pack_tree_verification`` turns
the per-slot parent arrays into batched device tensors (ancestor chains via
pointer jumping, plus retrieve rows enumerating every padded root-to-leaf
path), so the engine resolves acceptance with a handful of batched kernels
and a single host transfer per round instead of per-slot pointer walks.

Invariants are identical to the chain hooks: the draft cache holds one valid
entry per committed position ``[0, start)`` (the block forward is shared and
lives in the parent class), and ``slot.ctx`` is the hidden-state chunk of the
newly committed tokens — here the accepted tree path gathered by the engine
(non-contiguous positions, unlike the chain prefix).
"""

from __future__ import annotations

import torch

from deepspec.eval.batched.engine import BatchedProposal, PackedTreeVerification
from deepspec.eval.batched.dspark_hooks import DSparkBatchedHooks
from deepspec.eval.dspark.tree_ops import DSparkTree, build_dspark_tree_proposals_batched
from deepspec.modeling.dspark.common import extract_context_feature


def _pack_tree_verification(
    trees: list,
    *,
    block_size: int,
    device: torch.device,
) -> PackedTreeVerification:
    """Pack per-slot draft trees into batched device verification tensors.

    All ops are batched across slots and vectorized over nodes — no per-node
    Python loops and no host synchronization (``n_list`` comes from tensor
    shapes, which are host-side metadata).

    ``L = block_size + 1`` bounds every root-to-node chain (depths are in
    ``[0, block_size - 1]`` by construction).  ``P = n_max`` is a static
    worst-case leaf budget, so no per-round leaf-count sync is needed.
    """
    batch_size = len(trees)
    n_list = [int(tree.tokens.numel()) for tree in trees]
    n_max = 1 + max(n_list)
    depth_budget = int(block_size)
    path_width = depth_budget + 1  # self + up to depth_budget ancestors
    num_paths = n_max

    # Packed parent array: parent_of[b, 0] = -1 (anchor has no parent),
    # parent_of[b, 1 + i] = packed parent of node i; -1 = padding.
    parent_of = torch.full((batch_size, n_max), -1, dtype=torch.long, device=device)
    depths_pad = torch.zeros(batch_size, n_max, dtype=torch.long, device=device)
    node_valid = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)
    packed_tokens = torch.full((batch_size, n_max), -1, dtype=torch.long, device=device)
    for row, (tree, n_i) in enumerate(zip(trees, n_list)):
        if n_i > 0:
            parent_of[row, 1 : 1 + n_i] = tree.parent_indices
            depths_pad[row, 1 : 1 + n_i] = tree.depths
            packed_tokens[row, 1 : 1 + n_i] = tree.tokens
        node_valid[row, : 1 + n_i] = True

    # Ancestor chains by pointer jumping: chain[b, i] = [i, parent(i), ...,
    # 0, -1, ...] (self -> root).  A node's parent chain reaches the anchor
    # within ``depth + 1 <= block_size`` hops, so ``path_width`` columns
    # always suffice.
    node_idx = torch.arange(n_max, device=device).unsqueeze(0).expand(batch_size, -1)
    chain_cols = [node_idx]
    cur = node_idx
    for _ in range(depth_budget):
        cur = parent_of.gather(1, cur.clamp(min=0))
        cur = torch.where(cur >= 0, cur, torch.full_like(cur, -1))
        chain_cols.append(cur)
    chain = torch.stack(chain_cols, dim=2)  # [B, n_max, L], self->root

    # Leaves: valid nodes that are nobody's parent (a childless anchor counts
    # as a leaf, covering the empty-tree case).  Padded parent targets land in
    # a throwaway column so scatter writes are all-True and deterministic.
    is_parent = torch.zeros(batch_size, n_max + 1, dtype=torch.bool, device=device)
    parent_targets = torch.where(
        parent_of[:, 1:] >= 0,
        parent_of[:, 1:],
        torch.full_like(parent_of[:, 1:], n_max),
    )
    is_parent.scatter_(1, parent_targets, True)
    is_leaf = node_valid & ~is_parent[:, :n_max]

    # Leaves first (stable -> ascending packed node order), padding last, so
    # argmax ties during verification always resolve to a real leaf path.
    leaf_order = torch.argsort((~is_leaf).to(torch.int64), dim=1, stable=True)
    leaf_slots = leaf_order[:, :num_paths]
    leaf_sel_valid = is_leaf.gather(1, leaf_slots)  # [B, P]
    leaf_chain = chain.gather(
        1, leaf_slots.unsqueeze(-1).expand(-1, -1, path_width)
    )  # [B, P, L], self->root

    path_lens = (leaf_chain >= 0).sum(dim=-1)  # [B, P]
    path_lens = torch.where(leaf_sel_valid, path_lens, torch.zeros_like(path_lens))

    # Reverse each chain to root->leaf order and left-align the valid part.
    rev = leaf_chain.flip(-1)
    shift = (path_width - path_lens).clamp(min=0)
    cols = torch.arange(path_width, device=device).view(1, 1, -1) + shift.unsqueeze(-1)
    cols = cols.clamp(max=path_width - 1)
    paths_idx = rev.gather(2, cols)
    col_valid = (
        torch.arange(path_width, device=device).view(1, 1, -1)
        < path_lens.unsqueeze(-1)
    )
    paths_idx = torch.where(col_valid, paths_idx, torch.full_like(paths_idx, -1))
    paths_tok = packed_tokens.gather(
        1, paths_idx.clamp(min=0).reshape(batch_size, -1)
    ).reshape(batch_size, num_paths, path_width)
    paths_tok = torch.where(col_valid, paths_tok, torch.full_like(paths_tok, -1))

    return PackedTreeVerification(
        depths_pad=depths_pad,
        node_valid=node_valid,
        chain=chain,
        paths_idx=paths_idx,
        paths_tok=paths_tok,
        paths_len=path_lens,
    )


class DSparkBatchedTreeHooks(DSparkBatchedHooks):
    def __init__(
        self,
        draft_model,
        *,
        temperature: float,
        device: torch.device,
        tree_top_k: int,
        tree_max_nodes: int,
        compact_factor: float = 2.0,
    ):
        assert float(temperature) < 1e-5, (
            "Tree proposals are only exact for greedy targets "
            f"(temperature=0), got temperature={float(temperature)}."
        )
        assert int(tree_top_k) >= 2, (
            f"DSparkBatchedTreeHooks requires tree_top_k >= 2, got {tree_top_k}."
        )
        super().__init__(
            draft_model,
            temperature=temperature,
            confidence_threshold=0.0,
            device=device,
            compact_factor=compact_factor,
        )
        self.tree_top_k = int(tree_top_k)
        self.tree_max_nodes = int(tree_max_nodes)

    def propose(self, slots, next_rng) -> BatchedProposal:
        del next_rng  # tree building is deterministic (top-k, no sampling)
        block_hidden, draft_input_ids, anchor = self._propose_block_hidden(slots)
        tokens, parent_indices, depths = build_dspark_tree_proposals_batched(
            self.model,
            draft_input_ids=draft_input_ids,
            block_hidden=block_hidden,
            block_size=int(self.model.block_size),
            top_k=self.tree_top_k,
            max_nodes=self.tree_max_nodes,
        )
        # Uniform node count across slots (the candidate-pool shape is
        # content-independent), so n_max is static and no padding is needed.
        num_nodes = int(tokens.shape[1])
        verify_input_ids = torch.cat([anchor.unsqueeze(1), tokens], dim=1)
        # Row views into the batched tensors — the packed layout (anchor at
        # packed index 0, node i at 1 + i) and the per-slot DSparkTree
        # contract consumed by the engine are unchanged.
        trees = [
            DSparkTree(
                tokens=tokens[row],
                parent_indices=parent_indices[row],
                depths=depths[row],
            )
            for row in range(len(slots))
        ]
        # Trees stay device-resident; the engine consumes the batched
        # verification pack (ancestor chains + retrieve rows) instead of
        # walking per-slot host copies.
        tree_pack = _pack_tree_verification(
            trees,
            block_size=int(self.model.block_size),
            device=self.device,
        )
        return BatchedProposal(
            verify_input_ids=verify_input_ids,
            proposal_lens=[num_nodes] * len(slots),
            draft_probs=None,
            trees=trees,
            tree_pack=tree_pack,
        )

    def on_verify_tree(
        self,
        slots,
        hidden_states,
        commit_positions_list: list[list[int]],
    ) -> None:
        """Per-slot ctx = hidden states of the accepted tree path.

        ``commit_positions_list`` holds each slot's packed indices (anchor 0
        plus accepted nodes) whose hidden states are committed — the tree
        analogue of the chain path's ``[:keep]`` slice.

        Vectorized: one layer-concat plus one batched gather for the whole
        slot batch, replacing the previous per-slot x per-layer
        ``index_select`` + ``clone`` (B x (L + 1) kernels per round).
        ``slot.ctx`` rows are views into the freshly gathered tensor (never
        mutated afterwards), matching the chain hooks' aliasing semantics.
        """
        features = extract_context_feature(
            hidden_states,
            self.model.target_layer_ids,
        )  # [B, n_max, F]
        batch_size = len(slots)
        max_keep = max(len(p) for p in commit_positions_list)
        gather_idx = torch.zeros(batch_size, max_keep, dtype=torch.long)
        for row, positions in enumerate(commit_positions_list):
            gather_idx[row, : len(positions)] = torch.tensor(
                positions, dtype=torch.long
            )
        gathered = features.gather(
            1,
            gather_idx.to(features.device).unsqueeze(-1).expand(
                -1, -1, features.size(-1)
            ),
        )
        for row, slot in enumerate(slots):
            slot.ctx = gathered[row, : len(commit_positions_list[row])]


__all__ = ["DSparkBatchedTreeHooks"]
