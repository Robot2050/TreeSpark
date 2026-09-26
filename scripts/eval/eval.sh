#!/usr/bin/env bash
# Sequential single-request speculative-decoding evaluation (chain mode).
#
# eval.py spawns one worker per visible accelerator by itself; RANK/WORLD_SIZE
# follow the repo's node-launcher semantics (node rank / node count), so
# WORLD_SIZE=1 is a single-node run and the worker count comes from
# ASCEND_RT_VISIBLE_DEVICES (NPU) or CUDA_VISIBLE_DEVICES (GPU).
#
# Overrides:
#   TARGET_MODEL=Qwen/Qwen3-8B DRAFT_MODEL=./checkpoints/.../step_latest \
#       bash scripts/eval/eval.sh
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

# Target model the draft checkpoint was trained against.
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
# Training writes checkpoints under ./checkpoints/<project_name>/<exp_name>/step_*;
# use step_latest or a specific step_<N>.
DRAFT_MODEL=${DRAFT_MODEL:-./checkpoints/deepspec-opt/qwen3_4b_mixed_50K_cache_dsparkopt_4K/step_latest}

MASTER_PORT=${MASTER_PORT:-29667} python eval.py \
    --target_name_or_path "${TARGET_MODEL}" \
    --draft_name_or_path "${DRAFT_MODEL}"

# Background run:
#   nohup bash scripts/eval/eval.sh > eval_results/eval_chain.log 2>&1 &
