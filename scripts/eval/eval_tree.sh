#!/usr/bin/env bash
# Sequential single-request TREE-mode speculative-decoding evaluation
# (the full TreeSpark candidate policy).
#
# Tree drafting is exact for greedy targets only, so --temperature must be 0
# and --confidence-threshold stays 0.0 (tree proposals bypass confidence
# scheduling). --tree-top-k > 1 enables tree mode; --tree-max-nodes is the
# global non-root node budget verified in one target forward. Use the values
# recorded in the draft checkpoint's training config (paper setting: k=4,
# N=32).
#
# Launch semantics and overrides are identical to scripts/eval/eval.sh.
set -euo pipefail

export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}
if [[ "${DEEPSPEC_DEVICE}" == "npu" ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-600}
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
DRAFT_MODEL=${DRAFT_MODEL:-./checkpoints/deepspec-opt/qwen3_4b_mixed_50K_cache_dsparkopt_4K/step_latest}

TREE_TOP_K=${TREE_TOP_K:-4}
TREE_MAX_NODES=${TREE_MAX_NODES:-32}

MASTER_PORT=${MASTER_PORT:-29669} python eval.py \
    --target_name_or_path "${TARGET_MODEL}" \
    --draft_name_or_path "${DRAFT_MODEL}" \
    --temperature 0 \
    --tree-top-k "${TREE_TOP_K}" \
    --tree-max-nodes "${TREE_MAX_NODES}" \
    --confidence-threshold 0.0

# Background run:
#   nohup bash scripts/eval/eval_tree.sh > eval_results/eval_tree.log 2>&1 &
