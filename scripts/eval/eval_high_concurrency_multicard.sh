#!/usr/bin/env bash
# Multi-card high-concurrency speculative-decoding evaluation (batched engine).
#
# Launch model (same as the repo's other eval scripts): eval.py spawns one
# worker per *visible* device by itself; RANK/WORLD_SIZE mean
# node_rank/node_count, so WORLD_SIZE=1 is a single-node run and the worker
# count comes from ASCEND_RT_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES).
#
# Concurrency semantics: each worker runs a closed loop with C requests in
# flight on its own card, over its strided shard of the dataset.
#   total in-flight requests = #cards x C
# Metrics are aggregated across workers (tokens/acceptance summed, wall time
# maxed, latency samples gathered to rank 0). This is the setup used for the
# C=36 serving measurements in the paper (3 cards x 12 requests).
#
# Host-side single threading: the batched engine's host side is pure
# orchestration; eval.py sets torch.set_num_threads(1) per worker, and the
# OMP/MKL exports below keep every other native library single-threaded as
# well — with several workers per node, default OpenMP thread pools thrash
# the CPUs and starve the devices.
#
# Modes (MODE env var): tree | chain | baseline — see
# scripts/eval/eval_high_concurrency.sh.
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}

# Host-side single threading for the orchestration processes (see above).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

MODE=${MODE:-tree}                 # tree | chain | baseline
CONCURRENCY=${CONCURRENCY:-12}     # per-card concurrency C
TREE_TOP_K=${TREE_TOP_K:-4}
TREE_MAX_NODES=${TREE_MAX_NODES:-32}

# Visible cards for this run — worker count follows this list.
VISIBLE_CARDS=${VISIBLE_CARDS:-0,1,2,3}

TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-./checkpoints/deepspec-opt/qwen3_4b_mixed_50K_cache_dsparkopt_4K/step_latest}
OUTPUT_JSON=${OUTPUT_JSON:-./eval_results/eval_hc_${MODE}_multicard.json}

cmd=(
    python eval.py
    --engine batched
    --target_name_or_path "${TARGET_MODEL}"
    --temperature 0
    --concurrency "${CONCURRENCY}"
    --output-json "${OUTPUT_JSON}"
)

case "${MODE}" in
    tree)
        cmd+=(--draft_name_or_path "${DRAFT_MODEL}"
              --tree-top-k "${TREE_TOP_K}"
              --tree-max-nodes "${TREE_MAX_NODES}"
              --confidence-threshold 0.0)
        ;;
    chain)
        cmd+=(--draft_name_or_path "${DRAFT_MODEL}")
        ;;
    baseline)
        cmd+=(--baseline)
        ;;
    *)
        echo "Unknown MODE: ${MODE} (expected tree|chain|baseline)" >&2
        exit 1
        ;;
esac

ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_CARDS} MASTER_PORT=${MASTER_PORT:-29130} "${cmd[@]}"

# CUDA variant: drop the Ascend env sourcing and launch with
#   DEEPSPEC_DEVICE=cuda CUDA_VISIBLE_DEVICES=${VISIBLE_CARDS} ...
#
# Notes:
# - Throughput aggregation is total_tokens / max(worker wall time); keep
#   per-card sample counts balanced (the strided shard does this up to length
#   variance).
# - If latency-percentile gathering fails on HCCL, the run still succeeds and
#   percentiles fall back to rank-0 samples (a warning is printed).
# - Extend CONCURRENCY (e.g. "1,2,4,8,16,24,32") as needed.

# Background run:
#   nohup bash scripts/eval/eval_high_concurrency_multicard.sh > eval_results/eval_hc_${MODE}_multicard.log 2>&1 &
