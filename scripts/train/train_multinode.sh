#!/usr/bin/env bash
# =============================================================================
# Multi-node training launcher (two nodes, 8+8 accelerators = 16 workers).
#
# Rank semantics (mirrors deepspec/utils/distributed.py::init_dist):
#   train.py mp.spawns device_count() workers per node (8 cards -> 8 workers).
#   init_dist assigns global ranks as
#       global_rank = RANK * local_device_count + local_rank
#       world_size  = WORLD_SIZE * local_device_count
#   so RANK is the NODE index (0/1) and WORLD_SIZE is the NODE count (2):
#       node0 (RANK=0): global ranks 0-7   node1 (RANK=1): global ranks 8-15.
#
# Checked constraints for the default 8+8 topology (re-check after any change):
#   - Gradient accumulation: global_batch_size % (16 * local_batch_size) == 0
#     (default 64 % 16 == 0 -> gradient_accumulation_steps = 4).
#   - Sampling: samples_per_epoch is a multiple of global_batch_size.
#   - Sharding: default no_shard (DDP-equivalent), safe for 16 ranks.
#   - Loss normalization: denominators are globally all-reduced (SUM).
#
# Hard requirements:
#   1. Both nodes must see the SAME target_cache_dir and target_model_path
#      (shared storage, or identical local copies at identical paths).
#   2. Auto-resume requires checkpoint_dir to be visible to both nodes:
#      every rank loads weights from disk and reads its own
#      training_state.rank{N}.pt. If node1 cannot see the checkpoint dir it
#      silently trains from scratch while node0 resumes — state corruption.
#      Fresh training (no step_latest) has no such requirement. Resume also
#      requires an unchanged world_size (asserted in the checkpoint).
#   3. MASTER_ADDR:MASTER_PORT must be reachable between nodes; cross-node
#      NPU traffic uses HCCL (multi-NIC machines may need the optional HCCL
#      variables below).
#   4. Either node may start first, but both must meet within the init
#      timeout (60 minutes); starting node0 first is recommended.
#
# Usage (run once per node; only NODE_RANK differs):
#   node0:  NODE_RANK=0 MASTER_ADDR=<node0_ip> bash scripts/train/train_multinode.sh
#   node1:  NODE_RANK=1 MASTER_ADDR=<node0_ip> bash scripts/train/train_multinode.sh
# Background:
#   NODE_RANK=0 MASTER_ADDR=<node0_ip> nohup bash scripts/train/train_multinode.sh \
#       > checkpoints/multinode_rank0.log 2>&1 &
# =============================================================================
set -euo pipefail

export DEEPSPEC_DEVICE=${DEEPSPEC_DEVICE:-npu}
export TORCHDYNAMO_DISABLE=1

# ---- Cluster topology -------------------------------------------------------
NODE_RANK=${NODE_RANK:?"Set NODE_RANK (node0=0, node1=1)"}
NNODES=${NNODES:-2}
MASTER_ADDR=${MASTER_ADDR:?"Set MASTER_ADDR (node0's internal IP, same on both nodes)"}
MASTER_PORT=${MASTER_PORT:-33100}

if [[ "${NODE_RANK}" != "0" && "${NODE_RANK}" != "1" ]]; then
    echo "NODE_RANK must be 0 or 1 (got: ${NODE_RANK})" >&2
    exit 1
fi

# init_dist semantics: RANK = node index, WORLD_SIZE = node count (NOT cards).
export RANK=${NODE_RANK}
export WORLD_SIZE=${NNODES}

# ---- Local devices ------------------------------------------------------------
if [[ "${DEEPSPEC_DEVICE}" == "npu" ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
else
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
fi

# ---- Optional HCCL multi-NIC settings (uncomment and adapt) -------------------
# export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-eth0}
# export HCCL_IF_IP=${HCCL_IF_IP:-$(hostname -I | awk '{print $1}')}

# ---- Training parameters (identical on both nodes; env-overridable) -----------
config_path=${config_path:-config/dspark/dspark_qwen3_4b.py}
target_model_path=${target_model_path:-Qwen/Qwen3-4B}
target_cache_dir=${target_cache_dir:-./data/target_cache}
target_layer_ids=${target_layer_ids:-}
global_batch_size=${global_batch_size:-64}
max_train_steps=${max_train_steps:-}
data_max_length=${data_max_length:-4096}
logging_steps=${logging_steps:-10}
checkpointing_steps=${checkpointing_steps:-1000}
exp_name=${exp_name:-}

# ---- Pre-flight checks --------------------------------------------------------
[[ -f "${config_path}" ]] || { echo "Config not found: ${config_path}" >&2; exit 1; }
[[ -d "${target_cache_dir}" ]] || { echo "Target cache dir not found (needed on both nodes): ${target_cache_dir}" >&2; exit 1; }

# node1 may start before node0's TCP store is listening (node0 is still loading
# configs); wait up to 5 minutes softly. Failure only warns — init_process_group
# itself has a 60-minute timeout.
if [[ "${NODE_RANK}" == "1" ]]; then
    echo "Waiting for node0's TCP store (${MASTER_ADDR}:${MASTER_PORT}) ..."
    for i in $(seq 1 60); do
        if (echo > "/dev/tcp/${MASTER_ADDR}/${MASTER_PORT}") 2>/dev/null; then
            echo "TCP store reachable."
            break
        fi
        [[ ${i} == 60 ]] && echo "Warning: master unreachable after 5 min; continuing anyway (init timeout is 60 min)." >&2
        sleep 5
    done
fi

# ---- Launch -------------------------------------------------------------------
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

echo "node_rank=${NODE_RANK}/${NNODES} master=${MASTER_ADDR}:${MASTER_PORT} devices=${ASCEND_RT_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
MASTER_PORT=${MASTER_PORT} "${cmd[@]}"
