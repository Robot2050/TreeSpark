"""Batched DSpark drafting hooks for the high-concurrency engine.

Replicates the sequential DSpark proposal semantics per slot:

- The draft cache holds one valid entry per committed position ``[0, start)``
  (the block-parallel noise columns are invalidated right after each
  proposal, matching the sequential ``crop(start)``).
- ``slot.ctx`` is the chunk of verified target hidden states covering
  positions ``[start - ctx_len, start)`` — the full prompt hidden states for
  the first proposal, then the ``keep`` columns of the last verification.
- The block forward uses per-slot position rows and a 4D bool mask (padded
  context columns are never attended); Markov/confidence heads run batched.
- Per-slot confidence truncation yields per-slot proposal lengths ``p_i``;
  the engine pads shorter proposals and caps their acceptance.
"""

from __future__ import annotations

import torch

from deepspec.eval.batched.engine import BatchedProposal
from deepspec.eval.batched.kv_manager import RaggedBatchKV
from deepspec.modeling.dspark.common import extract_context_feature
from deepspec.utils.sampling import logits_to_probs


class DSparkBatchedHooks:
    def __init__(
        self,
        draft_model,
        *,
        temperature: float,
        confidence_threshold: float,
        device: torch.device,
        compact_factor: float = 2.0,
    ):
        self.model = draft_model
        self.temperature = float(temperature)
        self.confidence_threshold = float(confidence_threshold)
        self.device = device
        self.draft_kv = RaggedBatchKV(device, compact_factor=compact_factor)

    def kv_caches(self) -> list[RaggedBatchKV]:
        return [self.draft_kv]

    def on_prefill(
        self,
        new_slots,
        *,
        prefill_output,
        prompt_lens: list[int],
        max_prompt: int,
        prefill_rows: list[int],
    ) -> None:
        features = extract_context_feature(
            prefill_output.hidden_states,
            self.model.target_layer_ids,
        )  # [N, max_prompt, F] — left-padded rows
        for slot, length, row in zip(new_slots, prompt_lens, prefill_rows):
            # Clone: the slice is a view into the full prefill features, which
            # would otherwise stay alive until this slot's first proposal.
            slot.ctx = features[row, max_prompt - length :].clone()

    def _propose_block_hidden(self, slots):
        """Run the shared DSpark block-parallel draft forward.

        Returns ``(block_hidden, draft_input_ids, anchor)``.  Owns the draft
        KV bookkeeping for this round (only the first ``ctx_len`` context
        columns per slot stay valid, matching the sequential ``crop(start)``).
        Chain and tree proposals both start from this backbone output.
        """
        model = self.model
        batch_size = len(slots)
        block_size = int(model.block_size)
        device = self.device

        anchor = torch.tensor(
            [s.anchor_token for s in slots], dtype=torch.long, device=device
        )
        draft_input_ids = torch.full(
            (batch_size, block_size),
            int(model.mask_token_id),
            dtype=torch.long,
            device=device,
        )
        draft_input_ids[:, 0] = anchor

        ctx_lens_list = [int(s.ctx.shape[0]) for s in slots]
        ctx_max = max(ctx_lens_list)
        ctx = torch.zeros(
            batch_size,
            ctx_max,
            slots[0].ctx.shape[-1],
            dtype=slots[0].ctx.dtype,
            device=device,
        )
        for row, slot in enumerate(slots):
            ctx[row, : ctx_lens_list[row]] = slot.ctx

        starts = torch.tensor(
            [s.start for s in slots], dtype=torch.long, device=device
        )
        ctx_lens = torch.tensor(ctx_lens_list, dtype=torch.long, device=device)
        ctx_positions = (starts - ctx_lens).unsqueeze(1) + torch.arange(
            ctx_max, device=device
        ).unsqueeze(0)
        noise_positions = starts.unsqueeze(1) + torch.arange(
            block_size, device=device
        ).unsqueeze(0)
        position_ids = torch.cat([ctx_positions, noise_positions], dim=1)

        # ctx_lens_list is a host list: build the context validity flags on
        # the CPU directly (previously the list was uploaded and the result
        # downloaded again — a per-round host/device round-trip).
        ctx_flags = torch.arange(ctx_max).unsqueeze(0) < torch.tensor(
            ctx_lens_list, dtype=torch.long
        ).unsqueeze(1)
        new_key_mask = torch.cat(
            [ctx_flags, torch.ones(batch_size, block_size, dtype=torch.bool)],
            dim=1,
        )
        attention_mask = self.draft_kv.build_attention_mask(
            new_key_mask, causal_new=False, query_len=block_size
        )

        block_hidden = model._forward_backbone(
            target_hidden_states=ctx,
            noise_embedding=model.embed_tokens(draft_input_ids),
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=self.draft_kv.cache,
            use_cache=True,
            is_causal=False,
        )
        # The appended block holds [ctx columns | noise columns]; only the
        # first ctx_len context columns per slot stay valid (sequential crop).
        self.draft_kv.note_append(
            torch.cat(
                [
                    ctx_flags,
                    torch.zeros(batch_size, block_size, dtype=torch.bool),
                ],
                dim=1,
            )
        )
        return block_hidden, draft_input_ids, anchor

    def propose(self, slots, next_rng) -> BatchedProposal:
        model = self.model
        batch_size = len(slots)
        block_size = int(model.block_size)
        block_hidden, draft_input_ids, anchor = self._propose_block_hidden(slots)

        proposal_hidden = block_hidden[:, :block_size, :]
        base_logits = model.compute_logits(proposal_hidden)
        next_rng()
        sampled_tokens, draft_logits = model.sample_draft_tokens(
            base_logits,
            first_prev_token_ids=anchor,
            temperature=self.temperature,
            hidden_states=proposal_hidden,
        )

        proposal_lens = [block_size] * batch_size
        if model.confidence_head is not None and self.confidence_threshold > 0.0:
            prev_token_ids = torch.cat(
                [anchor.unsqueeze(1), sampled_tokens[:, :-1]], dim=1
            )
            confidence_logits = model.predict_confidence_step(
                proposal_hidden,
                prev_token_ids=prev_token_ids,
            )
            confidence_logits = confidence_logits.float().reshape(
                batch_size, block_size, -1
            )[:, :, 0]
            below = confidence_logits.sigmoid() < self.confidence_threshold
            # Vectorized first-sub-threshold index per row with ONE bulk D2H
            # transfer, replacing the previous per-row ``nonzero`` + ``.item()``
            # (B accelerator syncs per round).  Rows without a sub-threshold
            # position keep the full block.
            any_below = below.any(dim=1)
            first_below = below.to(torch.int64).argmax(dim=1)
            lens = torch.where(
                any_below, first_below, torch.full_like(first_below, block_size)
            )
            proposal_lens = [int(v) for v in lens.tolist()]

        verify_input_ids = torch.cat([anchor.unsqueeze(1), sampled_tokens], dim=1)
        draft_probs = logits_to_probs(draft_logits, self.temperature)
        return BatchedProposal(
            verify_input_ids=verify_input_ids,
            proposal_lens=proposal_lens,
            draft_probs=draft_probs,
        )

    def on_verify(self, slots, hidden_states, keeps: list[int]) -> None:
        features = extract_context_feature(
            hidden_states,
            self.model.target_layer_ids,
        )  # [B, q_max, F]
        for row, slot in enumerate(slots):
            slot.ctx = features[row, : keeps[row]]


__all__ = ["DSparkBatchedHooks"]
