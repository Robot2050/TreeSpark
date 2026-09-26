from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, DynamicCache

from deepspec.data.parser import encode_chat_messages
from deepspec.utils.sampling import (
    gather_token_probs,
    logits_to_probs,
    sample_from_probs,
    sample_residual,
)
from deepspec.utils import (
    init_dist,
    seed_all,
)
from deepspec.utils.device import synchronize_device
from deepspec.utils.distributed import print_on_global_main
from tqdm import tqdm


DEFAULT_DATASET_ROOT = "./eval_datasets"


def load_and_process_dataset(
    data_name: str,
    dataset_root: str = DEFAULT_DATASET_ROOT,
):
    dataset_path = Path(dataset_root) / f"{data_name}.jsonl"
    assert dataset_path.exists()

    rows = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = row.get("turns")
            assert (
                isinstance(turns, list)
                and len(turns) > 0
                and all(isinstance(turn, str) for turn in turns)
            ), (
                f"{dataset_path}:{line_number} must contain a non-empty string list field `turns`."
            )
            row["turns"] = turns[:1]
            rows.append(row)
    return rows


def trim_output_ids(
    output_ids: torch.Tensor,
    num_input_tokens: int,
    stop_token_ids: list[int] | None,
) -> torch.Tensor:
    if stop_token_ids is None:
        return output_ids
    stop_token_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
    stop_token_indices = torch.isin(
        output_ids[0][num_input_tokens:],
        stop_token_tensor,
    ).nonzero(as_tuple=True)[0]
    if stop_token_indices.numel() == 0:
        return output_ids
    return output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]


def has_stop_token(token_ids: torch.Tensor, stop_token_ids: list[int] | None) -> bool:
    if stop_token_ids is None:
        return False
    stop_token_tensor = torch.tensor(stop_token_ids, device=token_ids.device)
    return bool(torch.isin(token_ids, stop_token_tensor).any().item())


def resolve_stop_token_ids(target_model, tokenizer) -> list[int] | None:
    generation_config = getattr(target_model, "generation_config", None)
    eos_token_id = getattr(generation_config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return None
    if isinstance(eos_token_id, int):
        return [int(eos_token_id)]

    stop_token_ids = []
    for token_id in eos_token_id:
        token_id = int(token_id)
        if token_id not in stop_token_ids:
            stop_token_ids.append(token_id)
    return stop_token_ids


def assert_no_final_target_layer(target_model, target_layer_ids) -> None:
    target_config = target_model.config
    if hasattr(target_config, "text_config"):
        target_config = target_config.text_config
    last_layer_id = int(target_config.num_hidden_layers) - 1
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    assert last_layer_id not in target_layer_ids, (
        "target_layer_ids must not include the final target decoder layer "
        f"{last_layer_id}. Transformers output_hidden_states stores the final "
        "normalized hidden state at that position, while target cache generation "
        "stores raw decoder-layer outputs. Use an earlier layer and regenerate "
        "the target cache and draft checkpoint."
    )


def build_results_table(
    *,
    rows: list[dict[str, object]],
    model_name_or_path: str,
    draft_name_or_path: str,
    header: bool = True,
) -> str:
    from prettytable import PrettyTable

    table = PrettyTable()
    max_positions = max(
        (len(metrics["accept_rates_by_position"]) for metrics in rows),
        default=0,
    )
    field_names = [
        "dataset",
        "target_model",
        "draft_model",
        "#propose",
        "accept_len",
        "verify_rate",
        "tokens/s",
    ]
    field_names.extend(f"accept_rate@{pos_idx}" for pos_idx in range(max_positions))
    table.field_names = field_names
    table.header = header

    normalized_target_model_name = model_name_or_path.rstrip("/")
    target_model_name = (
        os.path.basename(normalized_target_model_name) or normalized_target_model_name
    )
    normalized_draft_model_name = draft_name_or_path.rstrip("/")
    draft_model_name = (
        os.path.basename(normalized_draft_model_name) or normalized_draft_model_name
    )
    for metrics in rows:
        row = [
            metrics["dataset"],
            target_model_name,
            draft_model_name,
            f"{metrics['draft_tokens_per_proposal']:.2f}+1",
            f"{metrics['acceptance_length']:.2f}",
            f"{metrics['verify_rate']:.4f}",
            f"{metrics['decode_tokens_per_second']:.2f}",
        ]
        row.extend(
            f"{accept_rate:.4f}" if accept_rate is not None else "-"
            for accept_rate in metrics["accept_rates_by_position"]
        )
        row.extend(["-"] * (max_positions - len(metrics["accept_rates_by_position"])))
        table.add_row(row)
    return table.get_string()


@dataclass
class DraftProposal:
    draft_token_count: int
    verify_input_ids: torch.Tensor
    draft_probs: torch.Tensor | None


@dataclass
class VerificationResult:
    target_output: Any
    target_probs: torch.Tensor | None
    accept_prefix_mask: torch.Tensor | None
    accepted_draft_tokens: int
    next_token: torch.Tensor
    effective_proposal_length: int
    terminated_by_stop_token: bool = False
    committed_tokens: torch.Tensor | None = None


def verify_draft_tokens(
    *,
    target_model,
    proposal: DraftProposal,
    position_ids: torch.Tensor,
    start: int,
    past_key_values_target: DynamicCache,
    temperature: float,
    max_proposal_tokens: int,
    current_token_ids: torch.Tensor | None = None,
    stop_token_ids: list[int] | None = None,
) -> VerificationResult:
    """Verify draft tokens with the target model and rejection sampling."""
    if proposal.draft_token_count > max_proposal_tokens:
        raise ValueError(
            "DraftProposal.draft_token_count must not exceed "
            f"max_proposal_tokens={max_proposal_tokens}, "
            f"got {proposal.draft_token_count}."
        )
    if current_token_ids is not None and not torch.equal(
        proposal.verify_input_ids[:, :1],
        current_token_ids,
    ):
        raise ValueError(
            "DraftProposal.verify_input_ids must start with the current "
            "accepted token."
        )

    draft_token_count = int(proposal.draft_token_count)
    verify_length = draft_token_count + 1
    verify_position_ids = position_ids[:, start : start + verify_length]
    target_output = target_model(
        input_ids=proposal.verify_input_ids,
        position_ids=verify_position_ids,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    if target_output.logits.ndim != 3:
        raise ValueError(
            "target model must return rank-3 logits [B, S, V], "
            f"got ndim={target_output.logits.ndim}."
        )
    target_probs = logits_to_probs(target_output.logits, float(temperature))
    if (
        draft_token_count > 0
        and proposal.draft_probs is not None
        and proposal.draft_probs.size(-1) != target_probs.size(-1)
    ):
        raise ValueError(
            "DraftProposal.draft_probs vocab size must match target logits, "
            f"got {proposal.draft_probs.size(-1)} and {target_probs.size(-1)}."
        )

    accept_prefix_mask = None
    if draft_token_count > 0:
        assert proposal.draft_probs is not None
        proposed_tokens = proposal.verify_input_ids[:, 1:]
        selected_target_probs = gather_token_probs(
            target_probs[:, :-1, :],
            proposed_tokens,
        )
        selected_draft_probs = gather_token_probs(
            proposal.draft_probs,
            proposed_tokens,
        ).clamp_min(1e-8)
        accept_prob = torch.clamp(
            selected_target_probs / selected_draft_probs,
            max=1.0,
        )
        accept_mask = (torch.rand_like(accept_prob) < accept_prob).to(torch.int64)
        accept_prefix_mask = accept_mask.cumprod(dim=1)
        accepted_draft_tokens = int(accept_prefix_mask.sum(dim=1)[0].item())
    else:
        accepted_draft_tokens = 0

    effective_proposal_length = draft_token_count
    terminated_by_stop_token = False
    if stop_token_ids and accepted_draft_tokens > 0:
        accepted_slice = proposal.verify_input_ids[0, 1 : accepted_draft_tokens + 1]
        stop_tensor = torch.tensor(
            stop_token_ids,
            device=accepted_slice.device,
            dtype=accepted_slice.dtype,
        )
        eos_hits = torch.isin(accepted_slice, stop_tensor).nonzero(as_tuple=True)[0]
        if eos_hits.numel() > 0:
            eos_pos = int(eos_hits[0].item())
            accepted_draft_tokens = eos_pos + 1
            effective_proposal_length = eos_pos + 1
            terminated_by_stop_token = True

    if 0 < draft_token_count and accepted_draft_tokens < draft_token_count:
        assert proposal.draft_probs is not None
        next_token = sample_residual(
            target_probs[:, accepted_draft_tokens, :],
            proposal.draft_probs[:, accepted_draft_tokens, :],
        )
    else:
        next_token = sample_from_probs(target_probs[:, -1:, :]).squeeze(1)

    committed_tokens = torch.cat(
        [
            proposal.verify_input_ids[:, 1 : accepted_draft_tokens + 1],
            next_token.unsqueeze(1),
        ],
        dim=1,
    )

    return VerificationResult(
        target_output=target_output,
        target_probs=target_probs,
        accept_prefix_mask=accept_prefix_mask,
        accepted_draft_tokens=accepted_draft_tokens,
        next_token=next_token,
        effective_proposal_length=effective_proposal_length,
        terminated_by_stop_token=terminated_by_stop_token,
        committed_tokens=committed_tokens,
    )


def build_tree_attention_mask(
    *,
    parent_indices: torch.Tensor,
    start: int,
    device: torch.device,
) -> torch.Tensor:
    """Bool ``[1, 1, q, start + q]`` tree mask for packed verification.

    Every node attends to the whole verified context (``kv < start``).  Inside
    the new segment a node attends only to itself and its ancestor path; the
    anchor (packed index 0) attends only to itself.  Sibling and cross-branch
    attention is blocked.  ``parent_indices[i]`` is the packed parent index of
    non-root node ``i`` (0 for children of the anchor).

    The mask is assembled on the CPU and copied over in one transfer: the
    ancestor walk is cheap scalar work, and a single host-to-device copy beats
    hundreds of tiny accelerator-side scalar writes.
    """
    num_nodes = 1 + int(parent_indices.numel())
    parents = [-1] + [int(p) for p in parent_indices.tolist()]
    new_segment = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    for node in range(num_nodes):
        ancestor = node
        while ancestor >= 0:
            new_segment[node, ancestor] = True
            ancestor = parents[ancestor]
    context = torch.ones(num_nodes, start, dtype=torch.bool)
    mask = torch.cat([context, new_segment], dim=-1).view(
        1, 1, num_nodes, start + num_nodes
    )
    return mask.to(device)


def gather_past_key_values(
    past_key_values: DynamicCache,
    keep_indices: torch.Tensor,
) -> None:
    """Keep only ``keep_indices`` positions of every cache layer (in place).

    Handles both the legacy ``key_cache``/``value_cache`` lists and the newer
    ``layers``-based ``DynamicCache`` layout.
    """
    layers = getattr(past_key_values, "layers", None)
    if layers:
        for layer in layers:
            layer.keys = layer.keys.index_select(2, keep_indices)
            layer.values = layer.values.index_select(2, keep_indices)
    else:
        key_cache = past_key_values.key_cache
        value_cache = past_key_values.value_cache
        for idx in range(len(key_cache)):
            if key_cache[idx].numel():
                key_cache[idx] = key_cache[idx].index_select(2, keep_indices)
                value_cache[idx] = value_cache[idx].index_select(2, keep_indices)
    if hasattr(past_key_values, "_seen_tokens"):
        past_key_values._seen_tokens = int(keep_indices.numel())


def verify_draft_tree(
    *,
    target_model,
    proposal: DraftProposal,
    position_ids: torch.Tensor,
    start: int,
    past_key_values_target: DynamicCache,
    temperature: float,
    max_proposal_tokens: int,
    current_token_ids: torch.Tensor | None = None,
    stop_token_ids: list[int] | None = None,
) -> VerificationResult:
    """Verify a packed draft tree with one target forward (greedy targets only).

    For a greedy target (``temperature < 1e-5``) this is exactly
    distribution-preserving: accepted tokens are the target's argmax
    continuation by construction.  The acceptance walk starts at the anchor
    and repeatedly follows the child whose token matches the target's argmax
    at the current node, yielding the longest accepted root-to-node path.
    Sampling targets need recursive rejection sampling to stay unbiased and
    are rejected here.

    Instead of re-running the accepted path, the accepted nodes' KV entries
    and hidden states are gathered straight out of the tree forward: under
    the tree mask those nodes attend to exactly their causal prefix, so their
    K/V and hidden states are identical to a causal replay's.  The whole
    round therefore costs a single target forward.
    """
    tree = getattr(proposal, "tree", None)
    assert tree is not None, "verify_draft_tree requires a tree proposal."
    assert float(temperature) < 1e-5, (
        "verify_draft_tree is only exact for greedy targets (temperature=0); "
        "use tree_top_k=1 for sampling temperatures."
    )
    if current_token_ids is not None and not torch.equal(
        proposal.verify_input_ids[:, :1],
        current_token_ids,
    ):
        raise ValueError(
            "DraftProposal.verify_input_ids must start with the current "
            "accepted token."
        )

    device = proposal.verify_input_ids.device
    num_tree_nodes = int(tree.tokens.numel())
    tree_position_ids = start + torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=device),
            tree.depths.to(device) + 1,
        ]
    ).unsqueeze(0)
    attention_mask = build_tree_attention_mask(
        parent_indices=tree.parent_indices.to(device),
        start=start,
        device=device,
    )
    target_output = target_model(
        input_ids=proposal.verify_input_ids,
        position_ids=tree_position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
    )
    # next_preds[i] = target argmax continuation after packed node i; one host
    # sync covers the whole acceptance walk.
    next_preds = target_output.logits[0].argmax(dim=-1).tolist()

    children: dict[int, list[tuple[int, int]]] = {}
    node_tokens = [int(t) for t in tree.tokens.tolist()]
    node_parents = [int(p) for p in tree.parent_indices.tolist()]
    for node_idx, (token_id, parent_idx) in enumerate(zip(node_tokens, node_parents)):
        children.setdefault(parent_idx, []).append((node_idx + 1, token_id))

    accepted_nodes: list[int] = []
    accepted_tokens: list[int] = []
    cursor = 0
    while True:
        prediction = next_preds[cursor]
        match = None
        for child_idx, child_token in children.get(cursor, ()):
            if child_token == prediction:
                match = (child_idx, child_token)
                break
        if match is None:
            break
        accepted_nodes.append(match[0] - 1)
        accepted_tokens.append(match[1])
        cursor = match[0]

    terminated_by_stop_token = False
    if stop_token_ids and accepted_tokens:
        for eos_pos, token_id in enumerate(accepted_tokens):
            if token_id in stop_token_ids:
                accepted_tokens = accepted_tokens[: eos_pos + 1]
                accepted_nodes = accepted_nodes[: eos_pos + 1]
                terminated_by_stop_token = True
                break

    accepted_draft_tokens = len(accepted_tokens)
    # Bonus token: target prediction at the last accepted node (at the anchor
    # node when nothing was accepted).
    next_token = torch.tensor(
        [next_preds[cursor]], dtype=torch.long, device=device
    )
    committed_tokens = torch.tensor(
        accepted_tokens + [next_preds[cursor]],
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    # Keep exactly the accepted path's KV entries: verified context, the
    # anchor, and the accepted nodes (cache layout appends the packed tree in
    # order, so non-root node i sits at cache index start + 1 + i).
    keep_indices = torch.tensor(
        list(range(start + 1)) + [start + 1 + node for node in accepted_nodes],
        dtype=torch.long,
        device=device,
    )
    gather_past_key_values(past_key_values_target, keep_indices)

    # Gather the accepted path's hidden states as the hidden-state source for
    # the evaluator's ``update`` hook (packed layout: anchor at 0, non-root
    # node i at 1 + i).
    gather_positions = torch.tensor(
        [0] + [1 + node for node in accepted_nodes],
        dtype=torch.long,
        device=device,
    )
    accepted_hidden_states = tuple(
        hidden.index_select(1, gather_positions)
        for hidden in target_output.hidden_states
    )

    return VerificationResult(
        target_output=SimpleNamespace(hidden_states=accepted_hidden_states),
        target_probs=None,
        accept_prefix_mask=None,
        accepted_draft_tokens=accepted_draft_tokens,
        next_token=next_token,
        effective_proposal_length=num_tree_nodes,
        terminated_by_stop_token=terminated_by_stop_token,
        committed_tokens=committed_tokens,
    )


@torch.inference_mode()
def generate_decoding_sample(
    *,
    target_model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    max_proposal_tokens: int,
    temperature: float,
    stop_token_ids: list[int] | None,
    init_context: Callable[..., Any],
    propose: Callable[..., DraftProposal],
    update: Callable[[Any, VerificationResult], None],
    post_verify: Callable[[DraftProposal, VerificationResult], None] | None = None,
    verify: Callable[..., VerificationResult] | None = None,
) -> SimpleNamespace:
    """Speculative-decoding loop.

    `init_context(initial_output, output_ids, position_ids, num_input_tokens)`
    builds the algorithm-specific state once after prefill. `propose(context,
    output_ids, position_ids, start, stop_token_ids)` returns the next
    DraftProposal. `update(context, verification)` advances the state when the
    loop continues. `post_verify` is an optional diagnostic hook called after
    every verification (used for confidence calibration). `verify` optionally
    overrides the verification step (defaults to `verify_draft_tokens`); the
    dspark evaluator passes `verify_draft_tree` for tree proposals.
    """
    assert max_proposal_tokens >= 1
    assert input_ids.size(0) == 1, "only bsz=1 is supported"
    if verify is None:
        verify = verify_draft_tokens

    device = input_ids.device
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + int(max_new_tokens)

    output_ids = torch.empty(
        (1, max_length + max_proposal_tokens + 1),
        dtype=torch.long,
        device=device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()

    # Synchronize around the timed regions: accelerator kernels are
    # asynchronous, so wall-clock timing without a sync would under-report.
    synchronize_device(device)
    prefill_start_time = time.perf_counter()
    output = target_model(
        input_ids=input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        output_hidden_states=True,
        logits_to_keep=1,
    )
    synchronize_device(device)
    prefill_seconds = time.perf_counter() - prefill_start_time
    decode_start_time = time.perf_counter()

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample_from_probs(
        logits_to_probs(output.logits, float(temperature))
    )

    start = input_ids.shape[1]
    acceptance_lengths: list[int] = []
    proposal_lengths: list[int] = []
    accepted_draft_lengths: list[int] = []

    initial_token = output_ids[:, num_input_tokens : num_input_tokens + 1]
    if has_stop_token(initial_token, stop_token_ids):
        output_ids = output_ids[:, : num_input_tokens + 1]
        output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
        return SimpleNamespace(
            output_ids=output_ids,
            num_input_tokens=num_input_tokens,
            num_output_tokens=output_ids.shape[1] - num_input_tokens,
            acceptance_lengths=acceptance_lengths,
            proposal_lengths=proposal_lengths,
            accepted_draft_lengths=accepted_draft_lengths,
            verify_count=0,
            prefill_seconds=prefill_seconds,
            decode_seconds=0.0,
        )

    context = init_context(
        initial_output=output,
        output_ids=output_ids,
        position_ids=position_ids,
        num_input_tokens=num_input_tokens,
    )

    while start < max_length:
        proposal = propose(
            context=context,
            output_ids=output_ids,
            position_ids=position_ids,
            start=start,
            stop_token_ids=stop_token_ids,
        )
        verification = verify(
            target_model=target_model,
            proposal=proposal,
            position_ids=position_ids,
            start=start,
            past_key_values_target=past_key_values_target,
            temperature=temperature,
            max_proposal_tokens=max_proposal_tokens,
            current_token_ids=output_ids[:, start : start + 1],
            stop_token_ids=stop_token_ids,
        )
        if post_verify is not None:
            post_verify(proposal, verification)

        proposal_lengths.append(int(verification.effective_proposal_length))
        accepted_draft_tokens = int(verification.accepted_draft_tokens)
        accepted_draft_lengths.append(accepted_draft_tokens)

        # committed_tokens[:-1] holds the accepted draft tokens; for tree
        # proposals this picks the accepted path instead of assuming the
        # layout order of proposal.verify_input_ids.  Identical to the old
        # write for single-chain proposals.
        output_ids[:, start : start + accepted_draft_tokens + 1] = torch.cat(
            [
                proposal.verify_input_ids[:, :1],
                verification.committed_tokens[:, :-1],
            ],
            dim=1,
        )

        if verification.terminated_by_stop_token:
            acceptance_lengths.append(accepted_draft_tokens)
            start += accepted_draft_tokens
            past_key_values_target.crop(start)
            break

        output_ids[:, start + accepted_draft_tokens + 1] = verification.next_token
        new_token_ids = output_ids[:, start + 1 : start + accepted_draft_tokens + 2]
        acceptance_lengths.append(accepted_draft_tokens + 1)
        start += accepted_draft_tokens + 1
        past_key_values_target.crop(start)
        update(context, verification)

        if has_stop_token(new_token_ids, stop_token_ids):
            break

    synchronize_device(device)
    decode_seconds = time.perf_counter() - decode_start_time

    output_ids = output_ids[:, : min(start + 1, max_length)]
    output_ids = trim_output_ids(output_ids, num_input_tokens, stop_token_ids)
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=output_ids.shape[1] - num_input_tokens,
        acceptance_lengths=acceptance_lengths,
        proposal_lengths=proposal_lengths,
        accepted_draft_lengths=accepted_draft_lengths,
        verify_count=len(proposal_lengths),
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
    )


class BaseEvaluator:

    def __init__(self, local_rank: int, args):
        self.args = args
        self.device, self.global_rank, self.world_size = init_dist(local_rank)
        self.tasks = args.tasks

        self.target_model, self.draft_model, self.tokenizer = self.build_models()
        self.metrics_rows: list[dict[str, object]] = []

    @property
    def max_proposal_tokens(self) -> int:
        raise NotImplementedError

    def build_models(self):
        raise NotImplementedError

    def generate_one_sample(
        self,
        *,
        input_ids: torch.Tensor,
        stop_token_ids: list[int] | None,
    ) -> SimpleNamespace:
        raise NotImplementedError

    def build_metrics_row(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | float | list[int]],
    ) -> dict[str, object]:
        proposal_count = int(metric_summary["proposal_count"])
        if proposal_count == 0:
            acceptance_length = 0.0
            draft_tokens_per_proposal = 0.0
            verify_rate = 0.0
            accept_rates_by_position = [None] * self.max_proposal_tokens
        else:
            acceptance_length = (
                int(metric_summary["acceptance_length_sum"]) / proposal_count
            )
            draft_tokens_per_proposal = (
                int(metric_summary["proposal_length_sum"]) / proposal_count
            )
            verify_rate = int(metric_summary["acceptance_length_sum"]) / (
                int(metric_summary["proposal_length_sum"]) + proposal_count
            )
            proposals_at_pos = metric_summary["proposals_at_pos"]
            accepted_at_pos = metric_summary["accepted_at_pos"]
            assert isinstance(proposals_at_pos, list)
            assert isinstance(accepted_at_pos, list)
            accept_rates_by_position = []
            for pos_idx in range(self.max_proposal_tokens):
                position_proposal_count = proposals_at_pos[pos_idx]
                if position_proposal_count == 0:
                    accept_rates_by_position.append(None)
                    continue
                accept_rates_by_position.append(
                    accepted_at_pos[pos_idx] / position_proposal_count
                )
        decode_seconds_sum = float(metric_summary.get("decode_seconds_sum", 0.0))
        output_token_sum = int(metric_summary.get("output_token_sum", 0))
        prefill_seconds_sum = float(metric_summary.get("prefill_seconds_sum", 0.0))
        sample_count = max(int(metric_summary["sample_count"]), 1)
        return {
            "dataset": dataset_name,
            "num_samples": int(metric_summary["sample_count"]),
            "draft_tokens_per_proposal": draft_tokens_per_proposal,
            "acceptance_length": acceptance_length,
            "verify_rate": verify_rate,
            "accept_rates_by_position": accept_rates_by_position,
            "decode_tokens_per_second": output_token_sum / max(decode_seconds_sum, 1e-9),
            "avg_prefill_seconds": prefill_seconds_sum / sample_count,
        }

    def run_dataset(
        self,
        *,
        dataset_name: str,
        max_samples: int | None,
    ) -> list[SimpleNamespace]:
        seed_all(int(self.args.seed))
        dataset = load_and_process_dataset(dataset_name)

        if max_samples is not None and len(dataset) > max_samples:
            rng = random.Random(int(self.args.seed))
            dataset = list(dataset)
            rng.shuffle(dataset)
            dataset = dataset[:max_samples]

        dataset_size = len(dataset)
        stop_token_ids = resolve_stop_token_ids(self.target_model, self.tokenizer)
        responses = []

        is_main = self.global_rank == 0
        pbar = None
        if is_main:
            pbar = tqdm(
                total=dataset_size,
                desc=f"Eval {dataset_name}",
                unit="sample",
                dynamic_ncols=True,
                miniters=1,
            )

        # Running counters for the progress bar postfix and intermediate logging.
        local_sample_count = 0
        local_proposal_count = 0
        local_acceptance_length_sum = 0
        local_output_tokens_sum = 0
        local_verify_count_sum = 0
        local_prefill_seconds_sum = 0.0
        local_decode_seconds_sum = 0.0
        log_interval = max(1, dataset_size // 20)

        for idx in range(self.global_rank, dataset_size, self.world_size):
            seed_all(int(self.args.seed) + idx)
            instance = dataset[idx]
            messages = [{"role": "user", "content": instance["turns"][0]}]
            input_ids = encode_chat_messages(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                # enable_thinking=True,
            ).to(self.device)
            response = self.generate_one_sample(
                input_ids=input_ids,
                stop_token_ids=stop_token_ids,
            )
            responses.append(response)
            local_sample_count += 1

            acceptance_lengths = getattr(response, "acceptance_lengths", None)
            if acceptance_lengths:
                local_proposal_count += len(acceptance_lengths)
                local_acceptance_length_sum += sum(acceptance_lengths)
            local_output_tokens_sum += int(response.num_output_tokens)
            local_verify_count_sum += int(response.verify_count)
            local_prefill_seconds_sum += float(
                getattr(response, "prefill_seconds", 0.0)
            )
            local_decode_seconds_sum += float(
                getattr(response, "decode_seconds", 0.0)
            )

            if is_main:
                # Rank 0 advances the bar by the strided share of every rank,
                # capped so the bar never exceeds the dataset size.
                remaining = dataset_size - pbar.n
                pbar.update(min(self.world_size, remaining))

                postfix: dict[str, str] = {}
                if local_proposal_count > 0:
                    postfix["acc_len"] = (
                        f"{local_acceptance_length_sum / local_proposal_count:.2f}"
                    )
                postfix["out_tok"] = (
                    f"{local_output_tokens_sum / local_sample_count:.1f}"
                )
                postfix["verify"] = (
                    f"{local_verify_count_sum / local_sample_count:.1f}"
                )
                postfix["tok/s"] = (
                    f"{local_output_tokens_sum / max(local_decode_seconds_sum, 1e-9):.1f}"
                )
                pbar.set_postfix(postfix)

                # Intermediate summary logging (also keeps nohup logs readable,
                # where the carriage-return progress bar collapses).
                if pbar.n % log_interval == 0 or pbar.n == dataset_size:
                    avg_acc = (
                        local_acceptance_length_sum / local_proposal_count
                        if local_proposal_count > 0
                        else 0.0
                    )
                    print_on_global_main(
                        f"[{dataset_name}] {pbar.n}/{dataset_size} samples done | "
                        f"avg_out_tok={local_output_tokens_sum / local_sample_count:.1f} | "
                        f"avg_verify={local_verify_count_sum / local_sample_count:.1f} | "
                        f"avg_acc_len={avg_acc:.2f} | "
                        f"avg_tok_s={local_output_tokens_sum / max(local_decode_seconds_sum, 1e-9):.1f}"
                    )

        if is_main and pbar is not None:
            pbar.close()

        return responses

    def allreduce_response_metrics(
        self,
        responses: list[SimpleNamespace],
    ) -> dict[str, int | float | list[int]]:
        metric_summary: dict[str, int | float | list[int]] = {
            "sample_count": len(responses),
            "proposal_count": 0,
            "acceptance_length_sum": 0,
            "proposal_length_sum": 0,
            "proposals_at_pos": [0] * self.max_proposal_tokens,
            "accepted_at_pos": [0] * self.max_proposal_tokens,
        }
        proposals_at_pos = metric_summary["proposals_at_pos"]
        accepted_at_pos = metric_summary["accepted_at_pos"]
        assert isinstance(proposals_at_pos, list)
        assert isinstance(accepted_at_pos, list)

        output_token_sum = 0
        prefill_seconds_sum = 0.0
        decode_seconds_sum = 0.0
        for response in responses:
            output_token_sum += int(getattr(response, "num_output_tokens", 0))
            prefill_seconds_sum += float(getattr(response, "prefill_seconds", 0.0))
            decode_seconds_sum += float(getattr(response, "decode_seconds", 0.0))
            acceptance_lengths = getattr(response, "acceptance_lengths", None)
            proposal_lengths = getattr(response, "proposal_lengths", None)
            accepted_draft_lengths = getattr(response, "accepted_draft_lengths", None)
            if (
                acceptance_lengths is None
                or proposal_lengths is None
                or accepted_draft_lengths is None
            ):
                continue
            assert (
                len(acceptance_lengths)
                == len(proposal_lengths)
                == len(accepted_draft_lengths)
            ), (
                "acceptance_lengths, proposal_lengths and accepted_draft_lengths "
                "must have the same length."
            )
            for acceptance_length, proposal_length, accepted_draft_length in zip(
                acceptance_lengths,
                proposal_lengths,
                accepted_draft_lengths,
            ):
                metric_summary["proposal_count"] += 1
                metric_summary["acceptance_length_sum"] += int(acceptance_length)
                metric_summary["proposal_length_sum"] += int(proposal_length)
                accepted_draft_length = int(accepted_draft_length)
                for pos_idx in range(self.max_proposal_tokens):
                    if proposal_length > pos_idx:
                        proposals_at_pos[pos_idx] += 1
                    if accepted_draft_length > pos_idx:
                        accepted_at_pos[pos_idx] += 1

        scalar_tensor = torch.tensor(
            [
                int(metric_summary["sample_count"]),
                int(metric_summary["proposal_count"]),
                int(metric_summary["acceptance_length_sum"]),
                int(metric_summary["proposal_length_sum"]),
            ],
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM)

        position_tensor = torch.tensor(
            proposals_at_pos + accepted_at_pos,
            device=self.device,
            dtype=torch.int64,
        )
        if position_tensor.numel() > 0:
            dist.all_reduce(position_tensor, op=dist.ReduceOp.SUM)

        # Throughput accounting travels in a separate float tensor: ranks run
        # their strided subsets concurrently, so SUM(tokens)/SUM(decode time)
        # is the per-stream decode throughput (the overall throughput for the
        # single-card case).  float32 because HCCL has no float64 collectives
        # (same convention as confidence_metric_dtype / utils.metrics).
        timing_tensor = torch.tensor(
            [output_token_sum, prefill_seconds_sum, decode_seconds_sum],
            device=self.device,
            dtype=torch.float32,
        )
        dist.all_reduce(timing_tensor, op=dist.ReduceOp.SUM)
        return {
            "sample_count": int(scalar_tensor[0].item()),
            "proposal_count": int(scalar_tensor[1].item()),
            "acceptance_length_sum": int(scalar_tensor[2].item()),
            "proposal_length_sum": int(scalar_tensor[3].item()),
            "proposals_at_pos": position_tensor[
                : self.max_proposal_tokens
            ].tolist(),
            "accepted_at_pos": position_tensor[
                self.max_proposal_tokens :
            ].tolist(),
            "output_token_sum": int(timing_tensor[0].item()),
            "prefill_seconds_sum": float(timing_tensor[1].item()),
            "decode_seconds_sum": float(timing_tensor[2].item()),
        }

    def log_tensorboard(self) -> None:
        from torch.utils.tensorboard import SummaryWriter

        assert self.args.tensorboard_dir is not None
        assert self.args.step is not None
        Path(self.args.tensorboard_dir).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=self.args.tensorboard_dir)
        for metrics in self.metrics_rows:
            dataset_name = metrics["dataset"]
            writer.add_scalar(
                f"eval/{dataset_name}/draft_tokens_per_proposal",
                float(metrics["draft_tokens_per_proposal"]),
                global_step=self.args.step,
            )
            writer.add_scalar(
                f"eval/{dataset_name}/accept_length",
                float(metrics["acceptance_length"]),
                global_step=self.args.step,
            )
            writer.add_scalar(
                f"eval/{dataset_name}/verify_rate",
                float(metrics["verify_rate"]),
                global_step=self.args.step,
            )
            writer.add_scalar(
                f"eval/{dataset_name}/decode_tokens_per_second",
                float(metrics["decode_tokens_per_second"]),
                global_step=self.args.step,
            )
            for pos_idx, accept_rate in enumerate(metrics["accept_rates_by_position"]):
                if accept_rate is None:
                    continue
                writer.add_scalar(
                    f"eval/{dataset_name}/accept_rate@{pos_idx}",
                    float(accept_rate),
                    global_step=self.args.step,
                )
        writer.close()

    def print_results(self) -> None:
        if dist.get_rank() != 0:
            return
        print(
            build_results_table(
                rows=self.metrics_rows,
                model_name_or_path=self.args.target_name_or_path,
                draft_name_or_path=self.args.draft_name_or_path,
            ),
            flush=True,
        )

    def print_dataset_result(self, metrics_row: dict[str, object]) -> None:
        if dist.get_rank() != 0:
            return
        print(
            build_results_table(
                rows=[metrics_row],
                model_name_or_path=self.args.target_name_or_path,
                draft_name_or_path=self.args.draft_name_or_path,
                header=False,
            ),
            flush=True,
        )

    def record_dataset_metrics(
        self,
        *,
        dataset_name: str,
        metric_summary: dict[str, int | float | list[int]],
    ) -> dict[str, object] | None:
        if dist.get_rank() != 0 or int(metric_summary["sample_count"]) <= 0:
            return None
        metrics_row = self.build_metrics_row(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
        )
        self.metrics_rows.append(metrics_row)
        self.print_dataset_result(metrics_row)
        return metrics_row

    def report_results(self) -> None:
        if dist.get_rank() == 0 and self.metrics_rows:
            if self.args.tensorboard_dir is not None:
                self.log_tensorboard()
        self.print_results()
        output_json = getattr(self.args, "output_json", None)
        if dist.get_rank() == 0 and output_json:
            from deepspec.utils import CustomJSONEncoder, jsonable

            payload = {
                "args": jsonable(vars(self.args)),
                "rows": self.metrics_rows,
            }
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(payload, indent=2, cls=CustomJSONEncoder),
                encoding="utf-8",
            )
            print(f"Wrote eval metrics to {path}", flush=True)

    def evaluate(self) -> None:
        for dataset_name, max_samples in self.tasks:
            responses = self.run_dataset(
                dataset_name=dataset_name,
                max_samples=max_samples,
            )
            metric_summary = self.allreduce_response_metrics(responses)
            self.record_dataset_metrics(
                dataset_name=dataset_name,
                metric_summary=metric_summary,
            )

        self.report_results()

    def clean_up(self) -> None:
        dist.destroy_process_group()
