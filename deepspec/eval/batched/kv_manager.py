"""Ragged batched KV-cache manager for high-concurrency speculative decoding.

HF ``DynamicCache`` only supports uniform, append-only growth across the batch,
while speculative decoding rolls back a different number of rejected tokens per
sequence every round.  This wrapper keeps the batch-wide physical cache
append-only and tracks validity per slot instead:

- ``valid_mask[B, S]`` (CPU bool) marks which physical columns hold K/V of
  committed tokens, mirrored by ``valid_mask_dev`` on the accelerator.
  Rejected/padding columns are never deleted eagerly, only masked out.
- Attention masks are assembled on the accelerator from the device-side
  validity mirror: only the small per-round ``[B, A]`` new-column flags cross
  the host/device boundary, instead of the full ``[B, Q, S + A]`` mask being
  rebuilt on the CPU and uploaded every round.  The produced mask is
  bit-identical to the previous CPU-assembled one.
- When fragmentation exceeds ``compact_factor`` times the maximum valid
  length, ``rebuild`` gathers the valid columns per slot and re-pads,
  amortizing the copy cost.  ``rebuild`` is also the single path for slot
  membership changes (refill / drop / initial fill).

Invariants (mirrored from the sequential evaluator):

- The number of valid columns of slot ``i`` equals its current ``start`` —
  the logical position of its last committed token.  The cache therefore
  holds one entry per committed position ``[0, start)``; the anchor token at
  position ``start`` is re-forwarded in the next verification round.
- Logical positions are decoupled from physical columns: RoPE always uses
  explicit per-slot ``position_ids`` computed from ``start``.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache


def _layers_or_none(cache):
    """transformers 5.x stores per-layer tensors in ``cache.layers`` (a list,
    empty on a fresh cache); 4.x legacy caches expose ``key_cache`` instead.
    Note ``layers`` may be present but empty, so truthiness is not a valid
    layout check — use ``is not None`` for structure and ``len()`` for count.
    """
    return getattr(cache, "layers", None)


def num_cache_layers(cache) -> int:
    layers = _layers_or_none(cache)
    if layers is not None:
        return len(layers)
    return len(getattr(cache, "key_cache", []))


def get_layer_kv(cache, idx: int):
    layers = _layers_or_none(cache)
    if layers is not None:
        return layers[idx].keys, layers[idx].values
    return cache.key_cache[idx], cache.value_cache[idx]


def set_layer_kv(cache, idx: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    layers = _layers_or_none(cache)
    if layers is not None:
        layers[idx].keys = keys
        layers[idx].values = values
    else:
        cache.key_cache[idx] = keys
        cache.value_cache[idx] = values


def extract_valid_kv(cache, valid_mask_row: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (K, V) of the valid columns of one slot, shapes [H, L, D]."""
    keep_indices = valid_mask_row.nonzero(as_tuple=True)[0]
    result = []
    for layer_idx in range(num_cache_layers(cache)):
        keys, values = get_layer_kv(cache, layer_idx)
        result.append(
            (
                keys.index_select(2, keep_indices.to(keys.device)),
                values.index_select(2, keep_indices.to(values.device)),
            )
        )
    return result


class RebuildPlan:
    """One slot's content source for ``RaggedBatchKV.rebuild``.

    - ``("keep", old_slot)``: carry over the valid columns of an existing slot.
    - ``("external", layer_kv, valid_count)``: graft externally produced K/V
      (a freshly prefilled request).  ``layer_kv`` is a list of per-layer
      (K, V) with shape [1, H, L_i, D]; only the first ``valid_count`` columns
      are valid (the rest are left-padding from the prefill batch).
    - ``("empty",)``: a slot with no valid columns (fresh draft caches).
    """

    KEEP = "keep"
    EXTERNAL = "external"
    EMPTY = "empty"

    def __init__(self, kind: str, old_slot: int = -1, layer_kv=None, valid_count: int = 0):
        self.kind = kind
        self.old_slot = int(old_slot)
        self.layer_kv = layer_kv
        self.valid_count = int(valid_count)

    @classmethod
    def keep(cls, old_slot: int) -> "RebuildPlan":
        return cls(cls.KEEP, old_slot=old_slot)

    @classmethod
    def external(cls, layer_kv, valid_count: int) -> "RebuildPlan":
        return cls(cls.EXTERNAL, layer_kv=layer_kv, valid_count=valid_count)

    @classmethod
    def empty(cls) -> "RebuildPlan":
        return cls(cls.EMPTY)


class RaggedBatchKV:
    def __init__(self, device: torch.device, compact_factor: float = 2.0):
        self.device = device
        self.compact_factor = float(compact_factor)
        self.cache = DynamicCache()
        # Bookkeeping keeps a CPU copy (the engine reads it for compaction and
        # rebuild decisions) plus a device mirror used to assemble attention
        # masks on-device.  Only the small per-round [B, A] validity flags
        # cross the bus each round; the [B, 1, Q, S + A] mask itself is built
        # by accelerator kernels and never leaves the device.
        self.valid_mask: torch.Tensor | None = None  # [B, S] bool, CPU
        self.valid_mask_dev: torch.Tensor | None = None  # [B, S] bool, device
        self._causal_cache: dict[tuple[int, int], torch.Tensor] = {}

    @property
    def batch_size(self) -> int:
        return 0 if self.valid_mask is None else int(self.valid_mask.shape[0])

    @property
    def seq_length(self) -> int:
        return 0 if self.valid_mask is None else int(self.valid_mask.shape[1])

    @property
    def valid_len(self) -> torch.Tensor:
        if self.valid_mask is None:
            return torch.zeros(0, dtype=torch.long)
        return self.valid_mask.sum(dim=1)

    def initialize(self, valid_mask: torch.Tensor) -> None:
        """Attach bookkeeping after the (batched) prefill forward.

        ``valid_mask``: CPU bool [B, S] — for left-padded prefill this is the
        padding mask.  The model has already appended the prefill K/V to
        ``self.cache``.
        """
        assert valid_mask.dtype == torch.bool and valid_mask.ndim == 2
        self.valid_mask = valid_mask.cpu()
        self.valid_mask_dev = self.valid_mask.to(self.device)

    def note_append(self, new_valid_flags: torch.Tensor) -> None:
        """Record the validity of columns appended by the latest forward.

        ``new_valid_flags``: CPU bool [B, A] where A is the number of columns
        every slot appended in that forward (the model appends uniformly).
        """
        assert new_valid_flags.dtype == torch.bool and new_valid_flags.ndim == 2
        new_valid_flags = new_valid_flags.cpu()
        flags_dev = new_valid_flags.to(self.device)
        if self.valid_mask is None:
            self.valid_mask = new_valid_flags
            self.valid_mask_dev = flags_dev
        else:
            assert new_valid_flags.shape[0] == self.valid_mask.shape[0]
            self.valid_mask = torch.cat([self.valid_mask, new_valid_flags], dim=1)
            self.valid_mask_dev = torch.cat([self.valid_mask_dev, flags_dev], dim=1)

    def _causal_mask(self, q_len: int, appended: int) -> torch.Tensor:
        """Cached device-side lower-triangular [q_len, appended] bool mask."""
        key = (int(q_len), int(appended))
        cached = self._causal_cache.get(key)
        if cached is None:
            cached = torch.tril(
                torch.ones(q_len, appended, dtype=torch.bool, device=self.device)
            )
            self._causal_cache[key] = cached
        return cached

    def build_attention_mask(
        self,
        new_key_mask: torch.Tensor,
        *,
        causal_new: bool,
        query_len: int | None = None,
    ) -> torch.Tensor:
        """Bool ``[B, 1, Q, S + A]`` mask (True = attend) for the next forward.

        ``new_key_mask``: bool [B, A] marking which of the to-be-appended
        columns carry real tokens (per-slot proposals may be truncated).
        ``query_len`` defaults to A (target verification: queries are exactly
        the appended columns); pass it explicitly when the forward's query
        count differs from the appended column count (block-parallel draft:
        ``block`` queries attend ``ctx + block`` appended columns).
        With ``causal_new=True`` query ``j`` additionally only attends new
        columns ``m <= j``; otherwise every query attends every valid new
        column.

        The mask is assembled on the accelerator: ``new_key_mask`` is the only
        per-round host-to-device payload ([B, A]), the history segment comes
        from the device-side validity mirror, and the causal segment is cached
        per shape.  Bit-identical to CPU assembly + upload, without the
        per-round O(B * Q * (S + A)) host work and transfer.
        """
        assert new_key_mask.ndim == 2
        batch_size, appended = new_key_mask.shape
        q_len = int(query_len) if query_len is not None else appended
        device = self.device
        new_keys = new_key_mask.to(device)
        seq_len = self.seq_length
        if seq_len > 0:
            history = self.valid_mask_dev.unsqueeze(1).expand(
                batch_size, q_len, seq_len
            )
        else:
            history = torch.zeros(batch_size, q_len, 0, dtype=torch.bool, device=device)
        if causal_new:
            causal = self._causal_mask(q_len, appended)
            new_segment = new_keys.unsqueeze(1) & causal.unsqueeze(0)
        else:
            new_segment = new_keys.unsqueeze(1).expand(batch_size, q_len, appended)
        mask = torch.cat([history, new_segment], dim=-1)
        return mask.unsqueeze(1)

    def build_mask_with_new_segment(self, new_segment: torch.Tensor) -> torch.Tensor:
        """Bool ``[B, 1, Q, S + A]`` mask with a fully custom new segment.

        Used by tree verification, where per-slot ancestor relations cannot be
        expressed as (key mask x causal).  ``new_segment``: bool ``[B, Q, A]``
        (True = attend) for the to-be-appended columns — the ancestor walk is
        cheap scalar work assembled on the CPU; history comes from the
        device-side ``valid_mask`` mirror and the final concatenation runs on
        the accelerator, so only the small new segment crosses the bus.
        """
        assert new_segment.ndim == 3
        batch_size, q_len, appended = new_segment.shape
        device = self.device
        seq_len = self.seq_length
        if seq_len > 0:
            history = self.valid_mask_dev.unsqueeze(1).expand(
                batch_size, q_len, seq_len
            )
        else:
            history = torch.zeros(batch_size, q_len, 0, dtype=torch.bool, device=device)
        mask = torch.cat([history, new_segment.to(device)], dim=-1)
        return mask.unsqueeze(1)

    def needs_compaction(self) -> bool:
        if self.valid_mask is None or self.batch_size == 0:
            return False
        seq_len = self.seq_length
        max_valid = int(self.valid_len.max().item())
        return seq_len > max(1, max_valid) * self.compact_factor

    def rebuild(self, plans: list[RebuildPlan]) -> None:
        """Re-materialize the physical cache from per-slot plans.

        Gathers each slot's valid columns, right-pads them to a common length
        with (masked) zeros, and writes the result back into the same
        ``DynamicCache`` object.  Handles membership changes (the new slot
        list may be shorter/longer/differently ordered) and compaction.
        """
        if not plans:
            self.cache = DynamicCache()
            self.valid_mask = None
            self.valid_mask_dev = None
            return

        n_layers = num_cache_layers(self.cache)
        if n_layers == 0 and all(p.kind != RebuildPlan.EXTERNAL for p in plans):
            # Nothing materialized yet and nothing to graft: keep an empty
            # cache with zero-width bookkeeping (a fresh DynamicCache has no
            # layers; the next forward appends normally).
            self.valid_mask = torch.zeros(len(plans), 0, dtype=torch.bool)
            self.valid_mask_dev = self.valid_mask.to(self.device)
            return
        slot_columns: list[list[tuple[torch.Tensor, torch.Tensor] | None]] = []
        valid_counts: list[int] = []
        ref_k = ref_v = None
        for plan in plans:
            if plan.kind == RebuildPlan.KEEP:
                assert self.valid_mask is not None
                row_mask = self.valid_mask[plan.old_slot]
                keep_indices = row_mask.nonzero(as_tuple=True)[0]
                per_slot = []
                for layer_idx in range(n_layers):
                    keys, values = get_layer_kv(self.cache, layer_idx)
                    keep_dev = keep_indices.to(keys.device)
                    per_slot.append(
                        (
                            keys[plan.old_slot].index_select(1, keep_dev),
                            values[plan.old_slot].index_select(1, keep_dev),
                        )
                    )
                slot_columns.append(per_slot)
                valid_counts.append(int(keep_indices.numel()))
            elif plan.kind == RebuildPlan.EXTERNAL:
                assert plan.valid_count > 0, "external plans need >=1 valid column"
                per_slot = []
                for keys, values in plan.layer_kv:
                    keep = plan.valid_count
                    # External K/V come from a left-padded prefill: the valid
                    # columns are the LAST `valid_count` ones.
                    per_slot.append(
                        (
                            keys[0, :, -keep:, :],
                            values[0, :, -keep:, :],
                        )
                    )
                slot_columns.append(per_slot)
                valid_counts.append(plan.valid_count)
            elif plan.kind == RebuildPlan.EMPTY:
                slot_columns.append(None)  # type: ignore[arg-type]
                valid_counts.append(0)
            else:
                raise ValueError(f"Unknown rebuild plan kind: {plan.kind!r}")

        out_n_layers = n_layers
        if out_n_layers == 0:
            for plan, per_slot in zip(plans, slot_columns):
                if plan.kind == RebuildPlan.EXTERNAL:
                    out_n_layers = len(plan.layer_kv)
                    break
        assert out_n_layers > 0, "rebuild needs at least one non-empty source"
        for per_slot in slot_columns:
            if per_slot:  # non-None and non-empty (has per-layer tensors)
                assert len(per_slot) == out_n_layers, (
                    "all rebuild sources must have the same layer count"
                )
                ref_k, ref_v = per_slot[0]
                break
        if ref_k is None:
            # Every plan is EMPTY (e.g. all slots finished in the same round
            # and were refilled): no valid columns anywhere, so simply reset
            # to a fresh cache with zero-width bookkeeping — the next forward
            # appends normally.
            self.cache = DynamicCache()
            self.valid_mask = torch.zeros(len(plans), 0, dtype=torch.bool)
            self.valid_mask_dev = self.valid_mask.to(self.device)
            return

        new_seq_len = max(max(valid_counts), 1)
        new_batch = len(plans)
        cache_has_layers = n_layers > 0
        for layer_idx in range(out_n_layers):
            padded_keys = []
            padded_values = []
            for per_slot, count in zip(slot_columns, valid_counts):
                if per_slot is None:
                    keys = torch.zeros(
                        ref_k.shape[0], 0, ref_k.shape[-1],
                        dtype=ref_k.dtype, device=ref_k.device,
                    )
                    values = torch.zeros(
                        ref_v.shape[0], 0, ref_v.shape[-1],
                        dtype=ref_v.dtype, device=ref_v.device,
                    )
                else:
                    keys, values = per_slot[layer_idx]
                pad = new_seq_len - count
                if pad > 0:
                    keys = torch.nn.functional.pad(keys, (0, 0, 0, pad))
                    values = torch.nn.functional.pad(values, (0, 0, 0, pad))
                padded_keys.append(keys)
                padded_values.append(values)
            stacked_keys = torch.stack(padded_keys, dim=0)
            stacked_values = torch.stack(padded_values, dim=0)
            if cache_has_layers:
                set_layer_kv(self.cache, layer_idx, stacked_keys, stacked_values)
            else:
                # Fresh cache with no layers yet: update() materializes them.
                self.cache.update(stacked_keys, stacked_values, layer_idx)
        if hasattr(self.cache, "_seen_tokens"):
            self.cache._seen_tokens = new_seq_len

        counts = torch.tensor(valid_counts, dtype=torch.long)
        self.valid_mask = (
            torch.arange(new_seq_len).unsqueeze(0) < counts.unsqueeze(1)
        )
        assert self.valid_mask.shape == (new_batch, new_seq_len)
        self.valid_mask_dev = self.valid_mask.to(self.device)


__all__ = [
    "RaggedBatchKV",
    "RebuildPlan",
    "extract_valid_kv",
    "get_layer_kv",
    "num_cache_layers",
    "set_layer_kv",
]
