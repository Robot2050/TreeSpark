import os
from deepspec.trainer import Qwen3DSparkTrainer
BASE_TB_DIR = os.path.expanduser("./tensorboard")
BASE_CKPT_DIR = os.path.expanduser("./checkpoints")
project_name = "deepspec-opt"
exp_name = "qwen3_4b_mixed_50K_cache_dsparkopt_4K"
seed = 42

model = dict(
    target_model_name_or_path="Qwen/Qwen3-4B",
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 9, 17, 25, 33],
    mask_token_id=151669,
    num_anchors=512,

    ## draft attention head dim (None = inherit target's head_dim, 128 for Qwen3;
    ## 192 gave a clear training improvement over 64 in our sweep)
    draft_head_dim=192,

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,

    ## tree consistency loss (train/inference aligned, see tree_loss.py)
    tree_loss_top_k=4,
    tree_loss_num_anchors=128,
    tree_loss_max_nodes=32,
    tree_rank_margin=1.0,
    tree_rank_hinge_cap=4.0,
    tree_rank_top_j=4,
    tree_survival_alpha=0.1,
    tree_rank_alpha=0.1,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=64,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=1000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name=str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    
    return cfg
