#!/usr/bin/env bash
# Single-node training launcher.
#
# train.py spawns one worker per visible accelerator by itself; RANK/WORLD_SIZE
# here follow the repo's node-launcher semantics (node rank / node count), so
# WORLD_SIZE=1 is a single-node run and the worker count comes from
# ASCEND_RT_VISIBLE_DEVICES (NPU) or CUDA_VISIBLE_DEVICES (GPU).
#
# Common overrides (all optional):
#   config_path=config/dspark/dspark_qwen3_8b.py \
#   target_model_path=Qwen/Qwen3-8B \
#   target_cache_dir=/path/to/target_cache \
#   bash scripts/train/train.sh
set -euo pipefail

# Accelerator: "npu" (Ascend, default) or "cuda" (NVIDIA).
export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}
export TORCHDYNAMO_DISABLE=1

if [[ "${DEEPSPEC_DEVICE}" == "npu" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
fi
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

config_path=${config_path:-config/dspark/dspark_qwen3_4b.py}
target_model_path=${target_model_path:-Qwen/Qwen3-4B}
# Prepared target cache directory (see README, "Data preparation").
target_cache_dir=${target_cache_dir:-./data/target_cache}
target_layer_ids=${target_layer_ids:-}
global_batch_size=${global_batch_size:-64}
max_train_steps=${max_train_steps:-}
data_max_length=${data_max_length:-4096}
logging_steps=${logging_steps:-10}
checkpointing_steps=${checkpointing_steps:-1000}
exp_name=${exp_name:-}

cmd=(
    python train.py
    --config "${config_path}"
    --opts "model.target_model_name_or_path=${target_model_path}"
    --opts "data.target_cache_path=${target_cache_dir}"
    --opts "data.max_length=${data_max_length}"
    --opts "train.global_batch_size=${global_batch_size}"
    --opts "logging.logging_steps=${logging_steps}"
    --opts "logging.checkpointing_steps=${checkpointing_steps}"
)

if [[ -n "${target_layer_ids}" ]]; then
    cmd+=(--opts "model.target_layer_ids=${target_layer_ids}")
fi
if [[ -n "${max_train_steps}" ]]; then
    cmd+=(--opts "train.max_train_steps=${max_train_steps}")
fi
if [[ -n "${exp_name}" ]]; then
    cmd+=(--opts "exp_name=${exp_name}")
fi

MASTER_PORT=${MASTER_PORT:-33100} "${cmd[@]}"

# Background run:
#   nohup bash scripts/train/train.sh > checkpoints/train.log 2>&1 &
