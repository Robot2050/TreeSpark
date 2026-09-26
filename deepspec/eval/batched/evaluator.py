"""High-concurrency evaluation driver.

Runs the batched speculative-decoding engine over the eval datasets at one or
more concurrency levels (closed loop: up to C requests in flight per worker,
finished slots are refilled immediately).  Dataset sharding across workers is
identical to the sequential evaluator (strided by global rank), so total
in-flight requests = workers x C.

Reported per (dataset, concurrency):
- the sequential evaluator's quality metrics (acceptance length, verify
  rate, per-position accept rates) — speculative quality must not degrade
  under batching,
- serving metrics: system throughput (output tokens/s of the worker group),
  per-request decode tokens/s, TTFT and end-to-end service latency
  (mean/p50/p95), average active batch size.

Run again with ``--baseline`` for the no-draft batched autoregressive
reference; the speedup column is computed offline from the two JSON dumps.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from deepspec.eval.base_evaluator import (
    BaseEvaluator,
    assert_no_final_target_layer,
    load_and_process_dataset,
    resolve_stop_token_ids,
)
from deepspec.eval.batched.dspark_hooks import DSparkBatchedHooks
from deepspec.eval.batched.dspark_tree_hooks import DSparkBatchedTreeHooks
from deepspec.eval.batched.engine import BatchedSpecEngine, EngineResult
from deepspec.data.parser import encode_chat_messages
from deepspec.modeling.dspark.gemma4 import Gemma4DSparkModel
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.utils import CustomJSONEncoder, jsonable, seed_all
from deepspec.utils.distributed import print_on_global_main


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[low])
    frac = rank - low
    return float(ordered[low] * (1.0 - frac) + ordered[high] * frac)


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / max(len(values), 1),
        "p50": _percentile(values, 50.0),
        "p95": _percentile(values, 95.0),
    }


def build_concurrency_table(
    *,
    rows: list[dict[str, Any]],
    model_name_or_path: str,
    draft_name_or_path: str | None,
    header: bool = True,
) -> str:
    from prettytable import PrettyTable

    table = PrettyTable()
    max_positions = max(
        (len(row["accept_rates_by_position"]) for row in rows),
        default=0,
    )
    field_names = [
        "dataset",
        "method",
        "conc",
        "#req",
        "#propose",
        "acc_len",
        "verify_rate",
        "sys_tok/s",
        "req_tok/s",
        "ttft_ms",
        "svc_p50_ms",
        "svc_p95_ms",
        "avg_batch",
    ]
    field_names.extend(f"acc@{i}" for i in range(max_positions))
    table.field_names = field_names
    table.header = header

    del model_name_or_path, draft_name_or_path  # rows are self-describing
    for row in rows:
        line = [
            row["dataset"],
            row["method"],
            row["concurrency"],
            row["num_samples"],
            f"{row['draft_tokens_per_proposal']:.2f}+1"
            if row["method"] != "baseline"
            else "-",
            f"{row['acceptance_length']:.2f}",
            f"{row['verify_rate']:.4f}",
            f"{row['system_tokens_per_second']:.2f}",
            f"{row['decode_tokens_per_second']:.2f}",
            f"{row['ttft']['mean'] * 1e3:.1f}",
            f"{row['service_time']['p50'] * 1e3:.1f}",
            f"{row['service_time']['p95'] * 1e3:.1f}",
            f"{row['avg_active_slots']:.2f}",
        ]
        line.extend(
            f"{rate:.4f}" if rate is not None else "-"
            for rate in row["accept_rates_by_position"]
        )
        line.extend(["-"] * (max_positions - len(row["accept_rates_by_position"])))
        table.add_row(line)
    return table.get_string()


class BatchedEngineEvaluator(BaseEvaluator):
    """Shared batched-eval driver; subclasses provide models and hooks."""

    method = "dspark"

    def build_hooks(self):
        raise NotImplementedError

    def validate_engine_args(self) -> None:
        """Engine-capability constraints, checked after model loading."""
        assert int(getattr(self.args, "tree_top_k", 1)) == 1, (
            f"{self.__class__.__name__} does not support tree proposals "
            f"(--tree-top-k {getattr(self.args, 'tree_top_k', 1)}); "
            "use --engine sequential or a DSpark batched evaluator."
        )

    def __init__(self, local_rank: int, args):
        super().__init__(local_rank, args)
        self.validate_engine_args()

    # ------------------------------------------------------------------
    def summarize_result(self, result: EngineResult) -> dict[str, Any]:
        proposals_at_pos = [0] * self.max_proposal_tokens
        accepted_at_pos = [0] * self.max_proposal_tokens
        proposal_count = 0
        acceptance_length_sum = 0
        proposal_length_sum = 0
        latencies: list[dict[str, float]] = []
        for record in result.records:
            acceptance_lengths = record["acceptance_lengths"]
            proposal_lengths = record["proposal_lengths"]
            accepted_draft_lengths = record["accepted_draft_lengths"]
            assert len(acceptance_lengths) == len(proposal_lengths) == len(
                accepted_draft_lengths
            )
            for acc_len, prop_len, acc_draft in zip(
                acceptance_lengths, proposal_lengths, accepted_draft_lengths
            ):
                proposal_count += 1
                acceptance_length_sum += int(acc_len)
                proposal_length_sum += int(prop_len)
                for pos in range(self.max_proposal_tokens):
                    if prop_len > pos:
                        proposals_at_pos[pos] += 1
                    if acc_draft > pos:
                        accepted_at_pos[pos] += 1
            decode_span = float(record["decode_span"])
            output_tokens = int(record["output_tokens"])
            latencies.append(
                {
                    "ttft": float(record["ttft"]),
                    "service_time": float(record["service_time"]),
                    "decode_tps": (
                        (output_tokens - 1) / decode_span
                        if decode_span > 0 and output_tokens > 1
                        else 0.0
                    ),
                }
            )
        return {
            "sample_count": len(result.records),
            "proposal_count": proposal_count,
            "acceptance_length_sum": acceptance_length_sum,
            "proposal_length_sum": proposal_length_sum,
            "proposals_at_pos": proposals_at_pos,
            "accepted_at_pos": accepted_at_pos,
            "output_token_sum": int(result.total_output_tokens),
            "prefill_seconds_sum": sum(l["ttft"] for l in latencies),
            "decode_seconds_sum": sum(
                float(r["decode_span"]) for r in result.records
            ),
            "service_seconds_sum": sum(
                float(r["service_time"]) for r in result.records
            ),
            "wall_time": float(result.wall_time),
            "active_slots_sum": int(result.active_slots_sum),
            "rounds": int(result.rounds),
            "latencies": latencies,
        }

    def allreduce_engine_metrics(
        self,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        int_tensor = torch.tensor(
            [
                summary["sample_count"],
                summary["proposal_count"],
                summary["acceptance_length_sum"],
                summary["proposal_length_sum"],
                summary["output_token_sum"],
                summary["active_slots_sum"],
                summary["rounds"],
            ],
            device=self.device,
            dtype=torch.int64,
        )
        dist.all_reduce(int_tensor, op=dist.ReduceOp.SUM)

        position_tensor = torch.tensor(
            summary["proposals_at_pos"] + summary["accepted_at_pos"],
            device=self.device,
            dtype=torch.int64,
        )
        if position_tensor.numel() > 0:
            dist.all_reduce(position_tensor, op=dist.ReduceOp.SUM)

        # HCCL (Ascend) does not support float64 collectives — keep the
        # cross-worker reduction in float32 (ample precision for timing sums).
        float_tensor = torch.tensor(
            [
                summary["prefill_seconds_sum"],
                summary["decode_seconds_sum"],
                summary["service_seconds_sum"],
                summary["wall_time"],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        dist.all_reduce(float_tensor[:3], op=dist.ReduceOp.SUM)
        # Workers run their shards concurrently: the group wall time is the max.
        dist.all_reduce(float_tensor[3:], op=dist.ReduceOp.MAX)

        latencies: list[dict[str, float]] = summary["latencies"]
        if self.world_size > 1:
            gathered: list[list[dict[str, float]]] = []
            try:
                if self.global_rank == 0:
                    gathered = [None] * self.world_size  # type: ignore[list-item]
                    dist.gather_object(latencies, gathered, dst=0)
                else:
                    dist.gather_object(latencies, None, dst=0)
                if self.global_rank == 0:
                    latencies = [item for sub in gathered for item in sub]
            except Exception:
                print_on_global_main(
                    "gather_object failed on this backend; latency percentiles "
                    "fall back to rank-0 samples only."
                )
        return {
            "sample_count": int(int_tensor[0].item()),
            "proposal_count": int(int_tensor[1].item()),
            "acceptance_length_sum": int(int_tensor[2].item()),
            "proposal_length_sum": int(int_tensor[3].item()),
            "output_token_sum": int(int_tensor[4].item()),
            "active_slots_sum": int(int_tensor[5].item()),
            "rounds": int(int_tensor[6].item()),
            "proposals_at_pos": position_tensor[: self.max_proposal_tokens].tolist(),
            "accepted_at_pos": position_tensor[self.max_proposal_tokens :].tolist(),
            "prefill_seconds_sum": float(float_tensor[0].item()),
            "decode_seconds_sum": float(float_tensor[1].item()),
            "service_seconds_sum": float(float_tensor[2].item()),
            "wall_time": float(float_tensor[3].item()),
            "latencies": latencies,
        }

    # ------------------------------------------------------------------
    def run_dataset_batched(
        self,
        *,
        dataset_name: str,
        max_samples: int | None,
        concurrency: int,
    ) -> dict[str, Any]:
        seed_all(int(self.args.seed))
        dataset = load_and_process_dataset(dataset_name)
        if max_samples is not None and len(dataset) > max_samples:
            rng = random.Random(int(self.args.seed))
            dataset = list(dataset)
            rng.shuffle(dataset)
            dataset = dataset[:max_samples]

        requests: list[torch.Tensor] = []
        for idx in range(self.global_rank, len(dataset), self.world_size):
            instance = dataset[idx]
            messages = [{"role": "user", "content": instance["turns"][0]}]
            input_ids = encode_chat_messages(
                self.tokenizer,
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            requests.append(input_ids[0].cpu())

        stop_token_ids = resolve_stop_token_ids(self.target_model, self.tokenizer)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0

        pbar = None
        if self.global_rank == 0:
            pbar = tqdm(
                total=len(requests) * self.world_size,
                desc=f"Eval {dataset_name} C={concurrency}",
                unit="req",
                dynamic_ncols=True,
                miniters=1,
            )

        def progress(n: int) -> None:
            if pbar is not None:
                pbar.update(min(n * self.world_size, max(pbar.total - pbar.n, 0)))

        engine = BatchedSpecEngine(
            target_model=self.target_model,
            device=self.device,
            max_new_tokens=int(self.args.max_new_tokens),
            temperature=float(self.args.temperature),
            stop_token_ids=stop_token_ids,
            hooks=self.build_hooks(),
            seed=int(self.args.seed) + self.global_rank * 1000003,
            pad_token_id=int(pad_token_id),
            progress=progress if self.global_rank == 0 else None,
        )
        result = engine.run(requests, concurrency)
        if pbar is not None:
            pbar.close()
        return self.summarize_result(result)

    def build_concurrency_row(
        self,
        *,
        dataset_name: str,
        concurrency: int,
        metric_summary: dict[str, Any],
    ) -> dict[str, Any] | None:
        if dist.get_rank() != 0 or int(metric_summary["sample_count"]) <= 0:
            return None
        row = self.build_metrics_row(
            dataset_name=dataset_name,
            metric_summary=metric_summary,
        )
        wall_time = max(float(metric_summary["wall_time"]), 1e-9)
        latencies = metric_summary["latencies"]
        row.update(
            {
                "method": self.method,
                "concurrency": int(concurrency),
                "wall_time": wall_time,
                "system_tokens_per_second": int(metric_summary["output_token_sum"])
                / wall_time,
                "avg_active_slots": int(metric_summary["active_slots_sum"])
                / max(int(metric_summary["rounds"]), 1),
                "ttft": _stats([l["ttft"] for l in latencies]),
                "service_time": _stats([l["service_time"] for l in latencies]),
                "decode_tps": _stats([l["decode_tps"] for l in latencies]),
            }
        )
        return row

    def evaluate(self) -> None:
        concurrency_list = [int(c) for c in self.args.concurrency]
        for dataset_name, max_samples in self.tasks:
            for concurrency in concurrency_list:
                summary = self.run_dataset_batched(
                    dataset_name=dataset_name,
                    max_samples=max_samples,
                    concurrency=concurrency,
                )
                reduced = self.allreduce_engine_metrics(summary)
                row = self.build_concurrency_row(
                    dataset_name=dataset_name,
                    concurrency=concurrency,
                    metric_summary=reduced,
                )
                if row is not None:
                    self.metrics_rows.append(row)
                    print(
                        build_concurrency_table(
                            rows=[row],
                            model_name_or_path=self.args.target_name_or_path,
                            draft_name_or_path=getattr(
                                self.args, "draft_name_or_path", None
                            ),
                            header=False,
                        ),
                        flush=True,
                    )
        self.report_results()

    def report_results(self) -> None:
        if dist.get_rank() != 0:
            return
        if self.metrics_rows:
            print(
                build_concurrency_table(
                    rows=self.metrics_rows,
                    model_name_or_path=self.args.target_name_or_path,
                    draft_name_or_path=getattr(self.args, "draft_name_or_path", None),
                ),
                flush=True,
            )
        output_json = getattr(self.args, "output_json", None)
        if output_json:
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
            print(f"Wrote high-concurrency metrics to {path}", flush=True)


class BatchedQwen3DSparkEvaluator(BatchedEngineEvaluator):
    EVAL_ATTN_IMPLEMENTATION = "sdpa"
    draft_model_cls = Qwen3DSparkModel

    @property
    def method(self) -> str:
        return (
            "dspark-tree"
            if int(getattr(self.args, "tree_top_k", 1)) > 1
            else "dspark"
        )

    @property
    def tree_enabled(self) -> bool:
        return int(getattr(self.args, "tree_top_k", 1)) > 1

    def validate_engine_args(self) -> None:
        if self.tree_enabled:
            assert float(self.args.temperature) < 1e-5, (
                "Tree proposals are only exact for greedy targets "
                f"(temperature=0), got temperature={float(self.args.temperature)}; "
                "use --tree-top-k 1 for sampling temperatures."
            )
            assert float(self.args.confidence_threshold) == 0.0, (
                "Tree proposals bypass confidence scheduling; set "
                f"--confidence-threshold 0.0, got {float(self.args.confidence_threshold)}."
            )

    @property
    def max_proposal_tokens(self) -> int:
        return int(self.draft_model.block_size)

    def build_models(self):
        target_model = AutoModelForCausalLM.from_pretrained(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()

        draft_model = self.draft_model_cls.from_pretrained(
            self.args.draft_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(self.device).eval()
        assert_no_final_target_layer(target_model, draft_model.target_layer_ids)
        assert 0.0 <= float(self.args.confidence_threshold) <= 1.0
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
        return target_model, draft_model, tokenizer

    def build_hooks(self):
        if self.tree_enabled:
            return DSparkBatchedTreeHooks(
                self.draft_model,
                temperature=float(self.args.temperature),
                device=self.device,
                tree_top_k=int(self.args.tree_top_k),
                tree_max_nodes=int(self.args.tree_max_nodes),
            )
        return DSparkBatchedHooks(
            self.draft_model,
            temperature=float(self.args.temperature),
            confidence_threshold=float(self.args.confidence_threshold),
            device=self.device,
        )


class BatchedGemma4DSparkEvaluator(BatchedQwen3DSparkEvaluator):
    draft_model_cls = Gemma4DSparkModel


class BatchedBaselineEvaluator(BatchedEngineEvaluator):
    """No-draft batched autoregressive baseline (target model only)."""

    EVAL_ATTN_IMPLEMENTATION = "sdpa"
    method = "baseline"

    @property
    def max_proposal_tokens(self) -> int:
        return 0

    def build_models(self):
        target_model = AutoModelForCausalLM.from_pretrained(
            self.args.target_name_or_path,
            dtype=torch.bfloat16,
            attn_implementation=self.EVAL_ATTN_IMPLEMENTATION,
        ).to(device=self.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.args.target_name_or_path)
        return target_model, None, tokenizer

    def build_hooks(self):
        return None


__all__ = [
    "BatchedBaselineEvaluator",
    "BatchedEngineEvaluator",
    "BatchedGemma4DSparkEvaluator",
    "BatchedQwen3DSparkEvaluator",
    "build_concurrency_table",
]
