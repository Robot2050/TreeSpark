import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids
from deepspec.utils import is_npu_available


TRAIN_ATTN_IMPLEMENTATION = "sdpa" if is_npu_available() else "flex_attention"


def build_draft_config(
    target_config,
    model_args,
):
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    # Optional draft attention head-dim override.  None (default) keeps the
    # target-inherited head_dim (hidden_size // num_attention_heads), i.e.
    # exactly the previous behavior.  Every draft-attention shape derives from
    # config.head_dim — q/k/v/o projections, q_norm/k_norm and the
    # attention scaling in Qwen3DSparkAttention — and the RoPE dimension is
    # also read from config.head_dim by transformers' rope_init_fn, so this
    # single field fully determines the draft attention geometry.  The value
    # must be even (RoPE rotate_half pairs adjacent dims); it does NOT need
    # to divide hidden_size because o_proj maps num_heads * head_dim back to
    # hidden_size.
    draft_head_dim = getattr(model_args, "draft_head_dim", None)
    if draft_head_dim is not None:
        draft_head_dim = int(draft_head_dim)
        assert draft_head_dim > 0 and draft_head_dim % 2 == 0, (
            f"draft_head_dim must be a positive even integer, got {draft_head_dim}"
        )
        draft_config.head_dim = draft_head_dim
    # Inference-aligned tree objectives; disabled when tree_loss_num_anchors == 0.
    draft_config.tree_loss_top_k = int(getattr(model_args, "tree_loss_top_k", 4))
    draft_config.tree_loss_num_anchors = int(
        getattr(model_args, "tree_loss_num_anchors", 0)
    )
    draft_config.tree_loss_max_nodes = int(getattr(model_args, "tree_loss_max_nodes", 32))
    draft_config.tree_rank_margin = float(getattr(model_args, "tree_rank_margin", 1.0))
    draft_config.tree_rank_hinge_cap = float(getattr(model_args, "tree_rank_hinge_cap", 4.0))
    draft_config.tree_rank_top_j = int(getattr(model_args, "tree_rank_top_j", 0))
    return draft_config


__all__ = [
    "build_draft_config",
]
