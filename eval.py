from __future__ import annotations
import argparse
import json
import torch
from transformers import AutoConfig
from deepspec.eval.dspark import Gemma4DSparkEvaluator, Qwen3DSparkEvaluator
from deepspec.eval.eagle3 import Gemma4Eagle3Evaluator, Qwen3Eagle3Evaluator
from deepspec.eval.batched import (
    BatchedBaselineEvaluator,
    BatchedGemma4DSparkEvaluator,
    BatchedQwen3DSparkEvaluator,
)
from deepspec.utils import CustomJSONEncoder, device_count

EVALUATORS = {
    "Qwen3DSparkModel": Qwen3DSparkEvaluator,
    "Gemma4DSparkModel": Gemma4DSparkEvaluator,
    "Qwen3Eagle3Model": Qwen3Eagle3Evaluator,
    "Gemma4Eagle3Model": Gemma4Eagle3Evaluator,
    "Eagle3DraftModel": Qwen3Eagle3Evaluator,
}

BATCHED_EVALUATORS = {
    "Qwen3DSparkModel": BatchedQwen3DSparkEvaluator,
    "Gemma4DSparkModel": BatchedGemma4DSparkEvaluator,
}

TASKS = [
    # General suite (nine benchmarks, shipped under eval_datasets/).
    ("gsm8k", 500),
    ("math500", 500),
    ("aime25", 30),
    ("humaneval", 164),
    ("mbpp", 256),
    ("livecodebench", 500),
    ("mt-bench", 80),
    ("alpaca", 500),
    ("arena-hard-v2", 500),

    # Writing suite (six WritingBench domains, shipped under eval_datasets/;
    # uncomment to include):
    # ("Academic & Engineering", 167),
    # ("Advertising & Marketing", 128),
    # ("Education", 111),
    # ("Finance & Business", 210),
    # ("Literature & Arts", 183),
    # ("Politics & Law", 201),
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_name_or_path", type=str, required=True)
    parser.add_argument(
        "--draft_name_or_path",
        type=str,
        default=None,
        help="Draft checkpoint. Not required with --engine batched --baseline.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--tree-top-k",
        type=int,
        default=1,
        help=(
            "Tree frontier width: every depth re-scores the shared backbone "
            "logits with each active parent's Markov bias and expands local "
            "top-k children, keeping at most k parents per depth. Values >1 "
            "enable tree drafting and require "
            "--temperature 0 (greedy target); 1 keeps the single greedy chain."
        ),
    )
    parser.add_argument(
        "--tree-max-nodes",
        type=int,
        default=32,
        help=(
            "Global node budget for the packed draft tree verified in one "
            "target forward when --tree-top-k > 1."
        ),
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=("Confidence-head early-stop threshold. Confidence calibration metrics are collected only when this is 0.0."),
    )
    parser.add_argument("--tensorboard-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None,help=("step for tensorboard logging"),)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument(
        "--engine",
        type=str,
        choices=["sequential", "batched"],
        default="sequential",
        help=(
            "sequential: one request at a time per worker (original behavior). "
            "batched: closed-loop high-concurrency decoding with continuous "
            "batching; sweep levels via --concurrency."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=str,
        default="4",
        help=(
            "Comma-separated in-flight request levels per worker for "
            "--engine batched, e.g. \"1,2,4,8,16\". Total in-flight requests "
            "= workers x level."
        ),
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=(
            "Batched mode only: skip the draft model and measure the "
            "no-draft batched autoregressive reference. "
            "--draft_name_or_path is not required in this mode."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Batched mode: write per-(dataset, concurrency) metrics to this JSON file.",
    )
    args = parser.parse_args()
    if not (args.engine == "batched" and args.baseline):
        assert args.draft_name_or_path, (
            "--draft_name_or_path is required unless --engine batched --baseline"
        )
    args.tasks = list(TASKS)
    args.concurrency = [
        int(level) for level in str(args.concurrency).split(",") if level.strip()
    ]
    assert args.concurrency, "--concurrency must contain at least one level"
    return args


def main(local_rank: int, args):
    if local_rank == 0:
        print(json.dumps(args, indent=4, cls=CustomJSONEncoder), flush=True)
    if args.engine == "batched":
        # The batched engine's host side is pure orchestration: many tiny CPU
        # ops per round.  Single-threaded CPU execution avoids OpenMP
        # thread-pool thrash on those micro-ops (a major source of the high
        # CPU utilization seen at high concurrency); all heavy math runs on
        # the accelerator.
        torch.set_num_threads(1)
    if args.engine == "batched" and args.baseline:
        evaluator = BatchedBaselineEvaluator(local_rank, args)
    else:
        draft_config = AutoConfig.from_pretrained(args.draft_name_or_path)
        architecture = draft_config.architectures[0]
        if args.engine == "batched":
            assert architecture in BATCHED_EVALUATORS, (
                f"--engine batched does not support {architecture} yet "
                f"(supported: {sorted(BATCHED_EVALUATORS)}); "
                "use --engine sequential."
            )
            evaluator_cls = BATCHED_EVALUATORS[architecture]
        else:
            evaluator_cls = EVALUATORS[architecture]
        evaluator = evaluator_cls(local_rank, args)
    evaluator.evaluate()
    evaluator.clean_up()

if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(
        main,
        args=(args,),
        nprocs=device_count(),
    )
