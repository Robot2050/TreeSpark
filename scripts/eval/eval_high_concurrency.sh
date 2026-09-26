#!/usr/bin/env bash
# High-concurrency speculative-decoding evaluation (batched engine), Ascend NPU.
#
# Closed-loop continuous batching: each worker keeps C requests in flight and
# refills finished slots immediately. Total in-flight requests = #workers x C.
# The (dataset x concurrency) sweep writes a JSON metrics dump (system and
# per-request throughput, TTFT, end-to-end latency percentiles, acceptance
# length) for throughput/latency-vs-concurrency curves.
#
# Modes (MODE env var):
#   tree      full TreeSpark policy (--tree-top-k 4 --tree-max-nodes 32);
#             requires a greedy target and disables confidence scheduling.
#   chain     single-proposal DSpark drafting (--tree-top-k 1).
#   baseline  no-draft batched autoregressive reference; no draft checkpoint
#             needed. Run with the same CONCURRENCY sweep as the spec runs and
#             compare the JSON dumps.
#
# Overrides: TARGET_MODEL, DRAFT_MODEL, MODE, CONCURRENCY, TREE_TOP_K,
#            TREE_MAX_NODES, OUTPUT_JSON, MASTER_PORT, VISIBLE_CARDS.
set -euo pipefail

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}

MODE=${MODE:-tree}                 # tree | chain | baseline
CONCURRENCY=${CONCURRENCY:-12}     # e.g. "1,6,12" for a sweep
TREE_TOP_K=${TREE_TOP_K:-4}
TREE_MAX_NODES=${TREE_MAX_NODES:-32}
VISIBLE_CARDS=${VISIBLE_CARDS:-0}

TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-./checkpoints/deepspec-opt/qwen3_4b_mixed_50K_cache_dsparkopt_4K/step_latest}
OUTPUT_JSON=${OUTPUT_JSON:-./eval_results/eval_hc_${MODE}.json}

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

ASCEND_RT_VISIBLE_DEVICES=${VISIBLE_CARDS} MASTER_PORT=${MASTER_PORT:-29800} "${cmd[@]}"

# Background run:
#   nohup bash scripts/eval/eval_high_concurrency.sh > eval_results/eval_hc_${MODE}.log 2>&1 &
