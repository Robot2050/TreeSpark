"""Closed-loop high-concurrency speculative-decoding engine.

Runs up to ``concurrency`` requests concurrently through one target model
with continuous batching: a finished slot is refilled with a new request at
the next round boundary.  Speculative proposals come from pluggable hooks
(DSpark); with ``hooks=None`` the engine degenerates to plain batched
autoregressive decoding (the no-draft baseline), reusing the exact same
verification path with empty proposals.

Per-slot bookkeeping mirrors the sequential ``generate_decoding_sample``
semantics: ``slot.start`` is the logical position of the last committed
token, the target/draft caches hold one valid entry per position in
``[0, start)``, and each round re-forwards the anchor token followed by the
slot's draft tokens.

Timing: one device synchronization per round boundary keeps wall-clock
measurements honest (accelerator kernels are asynchronous) without
serializing the work inside a round.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from transformers import DynamicCache

from deepspec.eval.batched.kv_manager import (
    RaggedBatchKV,
    RebuildPlan,
    get_layer_kv,
    num_cache_layers,
)
from deepspec.utils import seed_all
from deepspec.utils.device import synchronize_device
from deepspec.utils.sampling import (
    gather_token_probs,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
)


@dataclass
class SlotState:
    request_idx: int
    prompt_len: int
    start: int  # logical position of the last committed token (next anchor)
    anchor_token: int
    generated: list[int]
    ctx: Any = None  # hooks-managed per-slot state
    finished: bool = False
    acquire_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0
    acceptance_lengths: list[int] = field(default_factory=list)
    proposal_lengths: list[int] = field(default_factory=list)
    accepted_draft_lengths: list[int] = field(default_factory=list)


@dataclass
class PackedTreeVerification:
    """Device-packed per-slot tree verification data for one round.

    Built once per round by the tree hooks with fully batched device ops —
    no per-node Python loops, no host round-trips:

    - ``depths_pad``: [B, n_max] node depths at packed columns ``1 + i``
      (column 0 is the anchor and stays 0).
    - ``node_valid``: [B, n_max] bool — which packed columns are real nodes.
    - ``chain``: [B, n_max, L] packed ancestor chain self→root, -1 padded
      (pointer-jumped from the parent array; used for the ancestor-only
      attention mask).
    - ``paths_idx`` / ``paths_tok`` / ``paths_len``: flattened-tree-style retrieve
      rows — every (padded) root-to-leaf path as packed positions / node
      tokens / path lengths.  Verification compares each row against the
      preceding target logit and commits the longest accepted row, which is
      exactly the path the sequential pointer walk would take (children of a
      node carry distinct tokens, so the accepted prefix is unique).
    """

    depths_pad: torch.Tensor  # [B, n_max] long
    node_valid: torch.Tensor  # [B, n_max] bool
    chain: torch.Tensor  # [B, n_max, L] long, -1 padded
    paths_idx: torch.Tensor  # [B, P, L] long, -1 padded
    paths_tok: torch.Tensor  # [B, P, L] long, -1 padded (anchor column unused)
    paths_len: torch.Tensor  # [B, P] long


@dataclass
class BatchedProposal:
    """Draft proposals for the whole slot batch.

    ``verify_input_ids``: [B, 1 + max_proposal] — column 0 is the anchor,
    columns 1..p_i are slot i's draft tokens (rows may be wider than p_i;
    the engine masks the tail).  ``draft_probs``: [B, max_proposal, V] or
    None for the baseline.  ``proposal_lens``: per-slot draft token counts.
    ``trees``: per-slot packed draft trees (``DSparkTree``) for tree
    verification; None on the chain/baseline path.  ``tree_pack`` carries
    the batched device-side verification tensors for tree rounds.
    """

    verify_input_ids: torch.Tensor
    proposal_lens: list[int]
    draft_probs: torch.Tensor | None
    trees: list | None = None
    tree_pack: PackedTreeVerification | None = None


@dataclass
class EngineResult:
    records: list[dict[str, Any]]  # per request, ordered by request_idx
    wall_time: float
    rounds: int
    active_slots_sum: int
    total_output_tokens: int


class BatchedSpecEngine:
    def __init__(
        self,
        *,
        target_model,
        device: torch.device,
        max_new_tokens: int,
        temperature: float = 0.0,
        stop_token_ids: list[int] | None = None,
        hooks=None,
        seed: int = 0,
        pad_token_id: int = 0,
        compact_factor: float = 2.0,
        progress: Callable[[int], None] | None = None,
    ):
        self.target_model = target_model
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.stop_token_set = {int(t) for t in (stop_token_ids or [])}
        self.hooks = hooks
        self.seed = int(seed)
        self.pad_token_id = int(pad_token_id)
        self.progress = progress
        self.rng_counter = 0
        self.target_kv = RaggedBatchKV(device, compact_factor=compact_factor)
        self.concurrency = 1

    def next_rng(self) -> None:
        """Deterministic per-stage RNG reseed (reproducible per run config)."""
        seed_all(self.seed + self.rng_counter)
        self.rng_counter += 1

    # ------------------------------------------------------------------
    # prefill / membership
    # ------------------------------------------------------------------
    def _prefill(
        self,
        req_indices: list[int],
        requests: list[torch.Tensor],
        records: dict[int, dict[str, Any]],
    ) -> tuple[list[SlotState], list[RebuildPlan]]:
        prompts = [requests[i] for i in req_indices]
        lens = [int(p.numel()) for p in prompts]
        batch_size = len(prompts)
        max_prompt = max(lens)
        input_ids = torch.full(
            (batch_size, max_prompt), self.pad_token_id, dtype=torch.long
        )
        pad_mask = torch.zeros(batch_size, max_prompt, dtype=torch.bool)
        for row, (prompt, length) in enumerate(zip(prompts, lens)):
            input_ids[row, max_prompt - length :] = prompt
            pad_mask[row, max_prompt - length :] = True
        position_ids = (pad_mask.long().cumsum(-1) - 1).clamp(min=0)

        temp_cache = DynamicCache()
        synchronize_device(self.device)
        prefill_start = time.perf_counter()
        output = self.target_model(
            input_ids=input_ids.to(self.device),
            position_ids=position_ids.to(self.device),
            attention_mask=pad_mask.long().to(self.device),
            past_key_values=temp_cache,
            use_cache=True,
            output_hidden_states=self.hooks is not None,
            logits_to_keep=1,
        )
        self.next_rng()
        first_tokens = sample_from_probs(
            logits_to_probs(output.logits[:, -1:, :], self.temperature)
        ).squeeze(1)
        synchronize_device(self.device)
        first_token_time = time.perf_counter()

        new_slots: list[SlotState] = []
        ext_plans: list[RebuildPlan] = []
        live_rows: list[int] = []
        # One bulk D2H transfer for the whole prefill batch instead of one
        # ``.item()`` sync per slot.
        first_token_list = first_tokens.tolist()
        for row, (req_idx, length) in enumerate(zip(req_indices, lens)):
            token = int(first_token_list[row])
            slot = SlotState(
                request_idx=req_idx,
                prompt_len=length,
                start=length,
                anchor_token=token,
                generated=[token],
                acquire_time=prefill_start,
                first_token_time=first_token_time,
            )
            if token in self.stop_token_set or self.max_new_tokens <= 1:
                slot.finished = True
                slot.finish_time = first_token_time
                records[req_idx] = self._build_record(slot)
                records[req_idx]["service_time"] = first_token_time - prefill_start
                if self.progress is not None:
                    self.progress(1)
                continue
            new_slots.append(slot)
            live_rows.append(row)

        if new_slots and self.hooks is not None:
            self.hooks.on_prefill(
                new_slots,
                prefill_output=output,
                prompt_lens=[lens[row] for row in live_rows],
                max_prompt=max_prompt,
                prefill_rows=live_rows,
            )
        if live_rows:
            # Batch the KV extraction: one index_select + clone per layer for
            # all new slots (previously one clone per slot per layer — with L
            # layers and N refilled slots that was L x N small kernels per
            # refill).  The per-slot entries are row views into the cloned
            # batch; rebuild only reads them (slice + pad/stack copy out).
            rows_dev = torch.tensor(live_rows, dtype=torch.long, device=self.device)
            batched_layer_kv = []
            for layer_idx in range(num_cache_layers(temp_cache)):
                keys, values = get_layer_kv(temp_cache, layer_idx)
                batched_layer_kv.append(
                    (
                        keys.index_select(0, rows_dev).clone(),
                        values.index_select(0, rows_dev).clone(),
                    )
                )
            for row_pos, length in enumerate(lens[r] for r in live_rows):
                layer_kv = [
                    (keys[row_pos : row_pos + 1], values[row_pos : row_pos + 1])
                    for keys, values in batched_layer_kv
                ]
                ext_plans.append(RebuildPlan.external(layer_kv, valid_count=length))
        return new_slots, ext_plans

    def _update_membership(
        self,
        slots: list[SlotState],
        queue: deque,
        requests: list[torch.Tensor],
        records: dict[int, dict[str, Any]],
    ) -> None:
        survivor_idx = [i for i, s in enumerate(slots) if not s.finished]
        survivors = [slots[i] for i in survivor_idx]
        desired = min(self.concurrency, len(survivors) + len(queue))
        n_new = max(0, desired - len(survivors))
        membership_changed = len(survivors) != len(slots) or n_new > 0
        if not membership_changed and not self.target_kv.needs_compaction():
            return

        new_slots: list[SlotState] = []
        ext_plans: list[RebuildPlan] = []
        if n_new > 0:
            req_indices = [queue.popleft() for _ in range(n_new)]
            new_slots, ext_plans = self._prefill(req_indices, requests, records)

        keep_plans = [RebuildPlan.keep(i) for i in survivor_idx]
        target_plans = keep_plans + ext_plans
        if not target_plans:
            slots.clear()
            self.target_kv.rebuild([])
            return
        self.target_kv.rebuild(target_plans)
        if self.hooks is not None:
            hook_plans = keep_plans + [RebuildPlan.empty() for _ in new_slots]
            for kv in self.hooks.kv_caches():
                if kv.valid_mask is None:
                    # Not materialized yet (no forward so far); it learns the
                    # slot count from its first note_append.
                    continue
                kv.rebuild(hook_plans)
        slots[:] = survivors + new_slots

    # ------------------------------------------------------------------
    # decode rounds
    # ------------------------------------------------------------------
    def _verify_forward(
        self,
        slots: list[SlotState],
        proposal: BatchedProposal,
    ) -> tuple[Any, torch.Tensor, list[int], int]:
        batch_size = len(slots)
        proposal_lens = [int(p) for p in proposal.proposal_lens]
        q_max = 1 + (max(proposal_lens) if proposal_lens else 0)
        device = self.device
        # proposal_lens is a host list, so the padding mask is computed on the
        # CPU directly.  Previously the list was uploaded, compared on-device,
        # and the result downloaded again — a pointless host/device round-trip
        # every round.
        col_idx = torch.arange(q_max).unsqueeze(0)
        lens_tensor = torch.tensor(proposal_lens, dtype=torch.long).unsqueeze(1)
        real_cols = col_idx <= lens_tensor  # CPU bool [B, q_max]
        real_cols_dev = real_cols.to(device)
        input_ids = torch.where(
            real_cols_dev, proposal.verify_input_ids[:, :q_max].to(device), 0
        )
        starts = torch.tensor(
            [s.start for s in slots], dtype=torch.long, device=device
        )
        position_ids = starts.unsqueeze(1) + torch.arange(q_max, device=device).unsqueeze(0)
        attention_mask = self.target_kv.build_attention_mask(
            real_cols_dev, causal_new=True
        )
        output = self.target_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=self.target_kv.cache,
            use_cache=True,
            output_hidden_states=self.hooks is not None,
        )
        return output, input_ids, proposal_lens, q_max

    def _run_round(
        self,
        slots: list[SlotState],
        records: dict[int, dict[str, Any]],
    ) -> None:
        batch_size = len(slots)
        if self.hooks is None:
            anchor = torch.tensor(
                [s.anchor_token for s in slots], dtype=torch.long
            )
            proposal = BatchedProposal(
                verify_input_ids=anchor.unsqueeze(1),
                proposal_lens=[0] * batch_size,
                draft_probs=None,
            )
        else:
            proposal = self.hooks.propose(slots, self.next_rng)

        if proposal.trees is not None:
            # Tree verification is greedy-only and consumes no RNG.
            self._run_tree_round(slots, records, proposal)
            return

        self.next_rng()
        output, input_ids, proposal_lens, q_max = self._verify_forward(
            slots, proposal
        )
        target_probs = logits_to_probs(output.logits, self.temperature)

        device = self.device
        if proposal.draft_probs is not None and q_max > 1:
            proposed = input_ids[:, 1:]
            draft_probs = proposal.draft_probs[:, : q_max - 1, :].to(device)
            selected_target = gather_token_probs(target_probs[:, :-1, :], proposed)
            selected_draft = gather_token_probs(draft_probs, proposed).clamp_min(1e-8)
            accept_prob = (selected_target / selected_draft).clamp(max=1.0)
            lens_grid = torch.tensor(proposal_lens, device=device).unsqueeze(1)
            pad_cols = (
                torch.arange(q_max - 1, device=device).unsqueeze(0) >= lens_grid
            )
            accept_prob = accept_prob.masked_fill(pad_cols, 0.0)
            accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
            accepted = accept_mask.cumprod(dim=1).sum(dim=1)
            lens_tensor = lens_grid.squeeze(1)
            pos = torch.where(accepted < lens_tensor, accepted, lens_tensor)
            target_rows = target_probs.gather(
                1, pos.view(-1, 1, 1).expand(-1, 1, target_probs.size(-1))
            ).squeeze(1)
            draft_rows = draft_probs.gather(
                1, pos.view(-1, 1, 1).expand(-1, 1, draft_probs.size(-1))
            ).squeeze(1)
            next_from_residual = sample_residual(target_rows, draft_rows)
            next_from_target = sample_from_probs(target_rows.unsqueeze(1)).squeeze(1)
            next_token = torch.where(
                accepted < lens_tensor, next_from_residual, next_from_target
            )
            accepted_list = accepted.tolist()
        else:
            accepted_list = [0] * batch_size
            next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)

        # Bulk device-to-host transfers: one for the accepted-token columns
        # and one for the bonus/next tokens, replacing the previous per-slot
        # ``.tolist()``/``.item()`` calls (~2 x B accelerator round-trips per
        # round, which stalled the pipeline at high concurrency).
        next_token_list = next_token.tolist()
        verify_ids_list = input_ids[:, 1:].tolist()

        # Per-slot commit.
        keeps: list[int] = []
        for row, slot in enumerate(slots):
            acc = int(accepted_list[row])
            p_i = proposal_lens[row]
            committed = list(verify_ids_list[row][:acc])
            terminated = False
            if self.stop_token_set and acc > 0:
                for eos_pos, token in enumerate(committed):
                    if token in self.stop_token_set:
                        acc = eos_pos + 1
                        committed = committed[:acc]
                        terminated = True
                        break
            if terminated:
                # Mirror the sequential verifier: the bonus token is dropped
                # and the round counts as `acc` accepted tokens.
                slot.proposal_lengths.append(acc)
                slot.accepted_draft_lengths.append(acc)
                slot.acceptance_lengths.append(acc)
                slot.generated.extend(committed)
                slot.start += acc
                keeps.append(acc)
                self._finish_slot(slot, records)
                continue
            committed = committed + [int(next_token_list[row])]
            slot.proposal_lengths.append(p_i)
            slot.accepted_draft_lengths.append(acc)
            slot.acceptance_lengths.append(acc + 1)
            slot.generated.extend(committed)
            slot.start += acc + 1
            keeps.append(acc + 1)
            if self.stop_token_set.intersection(committed):
                self._finish_slot(slot, records)
            elif len(slot.generated) >= self.max_new_tokens:
                slot.generated = slot.generated[: self.max_new_tokens]
                self._finish_slot(slot, records)
            else:
                slot.anchor_token = committed[-1]

        synchronize_device(device)
        now = time.perf_counter()
        for slot in slots:
            if slot.finished and slot.finish_time == 0.0:
                slot.finish_time = now
                records[slot.request_idx]["finish_time"] = now
                records[slot.request_idx]["decode_span"] = now - slot.first_token_time
                records[slot.request_idx]["service_time"] = now - slot.acquire_time

        flags = torch.zeros(batch_size, q_max, dtype=torch.bool)
        for row, keep in enumerate(keeps):
            flags[row, :keep] = True
        self.target_kv.note_append(flags)

        if self.hooks is not None:
            self.hooks.on_verify(slots, output.hidden_states, keeps)

    def _run_tree_round(
        self,
        slots: list[SlotState],
        records: dict[int, dict[str, Any]],
        proposal: BatchedProposal,
    ) -> None:
        """Verify one packed draft tree per slot in a single target forward.

        Greedy-only (tree hooks require temperature=0).  Per slot the packed
        layout mirrors the sequential verifier: anchor at packed index 0,
        non-root node ``i`` at ``1 + i``; acceptance is resolved with
        flattened-tree-style retrieve rows (every padded root-to-leaf path compared
        against the preceding target logit; the longest accepted row is
        committed — provably the same nodes the sequential pointer walk would
        take); the committed KV/hidden entries are the accepted path's
        positions (NOT a prefix, unlike the chain path).
        """
        device = self.device
        batch_size = len(slots)
        pack = proposal.tree_pack
        n_list = [int(tree.tokens.numel()) for tree in proposal.trees]
        counts = [1 + n for n in n_list]
        n_max = int(pack.depths_pad.shape[1])

        # ``verify_input_ids`` is already the zero-padded [B, n_max] packed
        # layout (the tree hooks fill exactly ``counts[row]`` leading columns
        # per row), so it feeds the forward directly — no per-row copy loop.
        input_ids = proposal.verify_input_ids.to(device)

        # Vectorized position ids: anchor at ``start``, non-root node i at
        # ``start + depth_i + 1`` (siblings share a position), padded columns
        # pinned to ``start`` (they are masked, so the value is irrelevant —
        # kept identical to the previous host-built layout on purpose).
        # ``depths_pad`` is already [B, n_max] with the anchor column 0 at
        # column 0, so only the non-root columns get the +1 offset.
        starts = torch.tensor(
            [s.start for s in slots], dtype=torch.long, device=device
        )
        position_ids = starts.unsqueeze(1) + torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.long, device=device),
                pack.depths_pad[:, 1:] + 1,
            ],
            dim=1,
        )
        position_ids = torch.where(pack.node_valid, position_ids, starts.unsqueeze(1))

        # Ancestor-only new segment, scattered on-device from the
        # pointer-jumped ancestor chains: node attends itself + its
        # root-to-node ancestors; padded entries land in a throwaway column
        # that is sliced off, and padded query rows are allow-all (their
        # outputs are discarded).  Bit-identical to the previous CPU
        # ancestor-walk mask.
        anc_valid = pack.chain >= 0
        anc = torch.where(anc_valid, pack.chain, torch.full_like(pack.chain, n_max))
        seg = torch.zeros(batch_size, n_max, n_max + 1, dtype=torch.bool, device=device)
        seg.scatter_(2, anc, True)
        new_segment = seg[:, :, :n_max]
        pad_rows = torch.arange(n_max, device=device).unsqueeze(0) >= torch.tensor(
            counts, dtype=torch.long, device=device
        ).unsqueeze(1)
        new_segment = new_segment | pad_rows.unsqueeze(-1)
        attention_mask = self.target_kv.build_mask_with_new_segment(new_segment)

        output = self.target_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=self.target_kv.cache,
            use_cache=True,
            output_hidden_states=True,
        )
        # next_preds[i] = target argmax continuation after packed node i.
        next_preds = output.logits.argmax(dim=-1)  # [B, n_max], device

        # Flattened-tree-style retrieve-indices verification: compare every padded
        # root-to-leaf row against the preceding target logit, then commit the
        # longest accepted row.  Fully vectorized across slots and paths; the
        # only host transfer is one small bundle per round (previously the
        # walk cost B x O(N) Python/dict steps plus several device syncs).
        num_paths, path_width = pack.paths_idx.shape[1], pack.paths_idx.shape[2]
        gather_idx = pack.paths_idx.clamp(min=0)
        preds_at = next_preds.gather(
            1, gather_idx.reshape(batch_size, -1)
        ).reshape(batch_size, num_paths, path_width)
        match = preds_at[:, :, :-1] == pack.paths_tok[:, :, 1:]
        col = torch.arange(path_width - 1, device=device).view(1, 1, -1)
        match = match & (col < (pack.paths_len - 1).unsqueeze(-1))
        acc_paths = match.to(torch.int64).cumprod(dim=-1).sum(dim=-1)  # [B, P]
        # Neutralize padding rows (len 0) so argmax can never select one on a
        # tie at acc 0; real leaf paths always sort first and score >= 0.
        acc_paths = acc_paths.masked_fill(pack.paths_len == 0, -1)
        best = acc_paths.argmax(dim=1)  # leaves sort first; ties keep order
        acc = acc_paths.gather(1, best.unsqueeze(1)).squeeze(1)  # [B]
        expand_best = best.view(-1, 1, 1).expand(-1, 1, path_width)
        chosen_idx = pack.paths_idx.gather(1, expand_best).squeeze(1)
        chosen_tok = pack.paths_tok.gather(1, expand_best).squeeze(1)
        preds_best = preds_at.gather(1, expand_best).squeeze(1)
        bonus = preds_best.gather(1, acc.unsqueeze(1)).squeeze(1)

        # One bulk D2H bundle: [acc | bonus | chosen_idx(L) | chosen_tok(L)].
        bundle = (
            torch.cat(
                [acc.unsqueeze(1), bonus.unsqueeze(1), chosen_idx, chosen_tok],
                dim=1,
            )
            .cpu()
            .tolist()
        )

        flags = torch.zeros(batch_size, n_max, dtype=torch.bool)
        commit_positions_list: list[list[int]] = []
        for row, slot in enumerate(slots):
            entry = bundle[row]
            acc = int(entry[0])
            bonus_token = int(entry[1])
            positions = [int(p) for p in entry[2 : 2 + path_width] if int(p) >= 0]
            tokens_row = [int(t) for t in entry[2 + path_width :]]
            n_nodes = n_list[row]

            accepted_tokens = tokens_row[1 : acc + 1]
            commit_positions = positions[: acc + 1]
            terminated = False
            if self.stop_token_set and accepted_tokens:
                for eos_pos, token in enumerate(accepted_tokens):
                    if token in self.stop_token_set:
                        accepted_tokens = accepted_tokens[: eos_pos + 1]
                        commit_positions = commit_positions[: eos_pos + 2]
                        terminated = True
                        break

            acc = len(accepted_tokens)
            if terminated:
                slot.proposal_lengths.append(n_nodes)
                slot.accepted_draft_lengths.append(acc)
                slot.acceptance_lengths.append(acc)
                slot.generated.extend(accepted_tokens)
                slot.start += acc
                self._finish_slot(slot, records)
            else:
                committed = accepted_tokens + [bonus_token]
                slot.proposal_lengths.append(n_nodes)
                slot.accepted_draft_lengths.append(acc)
                slot.acceptance_lengths.append(acc + 1)
                slot.generated.extend(committed)
                slot.start += acc + 1
                if self.stop_token_set.intersection(committed):
                    self._finish_slot(slot, records)
                elif len(slot.generated) >= self.max_new_tokens:
                    slot.generated = slot.generated[: self.max_new_tokens]
                    self._finish_slot(slot, records)
                else:
                    slot.anchor_token = bonus_token
            for col_idx_ in commit_positions:
                flags[row, col_idx_] = True
            commit_positions_list.append(commit_positions)

        synchronize_device(device)
        now = time.perf_counter()
        for slot in slots:
            if slot.finished and slot.finish_time == 0.0:
                slot.finish_time = now
                records[slot.request_idx]["finish_time"] = now
                records[slot.request_idx]["decode_span"] = now - slot.first_token_time
                records[slot.request_idx]["service_time"] = now - slot.acquire_time

        self.target_kv.note_append(flags)
        self.hooks.on_verify_tree(slots, output.hidden_states, commit_positions_list)

    def _finish_slot(
        self,
        slot: SlotState,
        records: dict[int, dict[str, Any]],
    ) -> None:
        slot.finished = True
        records[slot.request_idx] = self._build_record(slot)
        if self.progress is not None:
            self.progress(1)

    @staticmethod
    def _build_record(slot: SlotState) -> dict[str, Any]:
        return {
            "request_idx": slot.request_idx,
            "prompt_len": slot.prompt_len,
            "output_tokens": len(slot.generated),
            "generated": list(slot.generated),
            "acceptance_lengths": list(slot.acceptance_lengths),
            "proposal_lengths": list(slot.proposal_lengths),
            "accepted_draft_lengths": list(slot.accepted_draft_lengths),
            "verify_count": len(slot.proposal_lengths),
            "acquire_time": slot.acquire_time,
            "first_token_time": slot.first_token_time,
            "finish_time": slot.finish_time,
            "ttft": max(slot.first_token_time - slot.acquire_time, 0.0),
            "decode_span": 0.0,
            "service_time": 0.0,
        }

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def run(
        self,
        requests: list[torch.Tensor],
        concurrency: int,
    ) -> EngineResult:
        self.concurrency = max(1, int(concurrency))
        queue = deque(range(len(requests)))
        slots: list[SlotState] = []
        records: dict[int, dict[str, Any]] = {}
        rounds = 0
        active_slots_sum = 0
        membership_dirty = True

        synchronize_device(self.device)
        wall_start = time.perf_counter()
        while slots or queue:
            if membership_dirty or self.target_kv.needs_compaction():
                self._update_membership(slots, queue, requests, records)
                membership_dirty = False
            if not slots:
                if queue:
                    # Everything prefilled so far finished on its first
                    # token; keep filling from the queue.
                    membership_dirty = True
                    continue
                break
            self._run_round(slots, records)
            active_slots_sum += len(slots)
            rounds += 1
            if any(slot.finished for slot in slots):
                membership_dirty = True
        synchronize_device(self.device)
        wall_time = time.perf_counter() - wall_start

        ordered = [records[i] for i in sorted(records)]
        assert len(ordered) == len(requests), "every request must have a record"
        return EngineResult(
            records=ordered,
            wall_time=wall_time,
            rounds=rounds,
            active_slots_sum=active_slots_sum,
            total_output_tokens=sum(r["output_tokens"] for r in ordered),
        )


__all__ = [
    "BatchedProposal",
    "BatchedSpecEngine",
    "EngineResult",
    "SlotState",
]
