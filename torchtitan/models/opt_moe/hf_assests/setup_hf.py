# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import shutil


def _native_to_hf_rope_scaling(rope_cfg):
    """Convert a native RoPE.Config's scaling fields to an HF rope_scaling dict.

    Returns None if scaling == "none" (the common case), or a dict matching the
    HF rope_scaling format for "llama3" / "yarn" types.
    """
    if rope_cfg is None or rope_cfg.scaling == "none":
        return None
    if rope_cfg.scaling == "llama":
        return {
            "rope_type": "llama3",
            "factor": rope_cfg.scaling_factor,
            "low_freq_factor": rope_cfg.low_freq_factor,
            "high_freq_factor": rope_cfg.high_freq_factor,
            "original_max_position_embeddings": rope_cfg.original_max_position_embeddings,
        }
    if rope_cfg.scaling == "yarn":
        return {
            "rope_type": "yarn",
            "factor": rope_cfg.rope_factor,
            "original_max_position_embeddings": rope_cfg.original_seq_len,
        }
    return None  # unknown scaling type — leave as null


def copy_and_overwrite_model_config(model, model_config, dst_path: str):
    """Copy HF config/modeling files to dst_path and overwrite config.json
    with parameters extracted from the trained model and model_config.

    Args:
        model: The trained OPTMoEModel instance (used to inspect actual sizes).
        model_config: OPTMoEModel.Config instance with training hyperparameters.
        dst_path: Destination directory for HF checkpoint.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_config_path = os.path.join(current_dir, "configuration_opt_moe.py")
    src_modeling_path = os.path.join(current_dir, "modeling_opt_moe.py")

    dst_config_py_path = os.path.join(dst_path, "configuration_opt_moe.py")
    dst_modeling_py_path = os.path.join(dst_path, "modeling_opt_moe.py")
    dst_config_json_path = os.path.join(dst_path, "config.json")

    new_config = overwrite_config(model, model_config)

    if os.path.exists(dst_path):
        shutil.copy(src_config_path, dst_config_py_path)
        shutil.copy(src_modeling_path, dst_modeling_py_path)
        with open(dst_config_json_path, "w") as f:
            json.dump(new_config, f, indent=4)


def overwrite_config(model, model_config):
    """Build a config.json dict from the model and model_config.

    Args:
        model: The trained OPTMoEModel instance.
        model_config: OPTMoEModel.Config instance.

    Returns:
        dict suitable for writing as config.json.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(current_dir, "config.json")) as f:
        default_config = json.load(f)

    attn_cfg = model_config.layer.attention

    # Inspect the first layer's attention module for runtime sizes
    attention = model.layers["0"].attention

    default_config["num_hidden_layers"] = model_config.n_layers
    default_config["vocab_size"] = model_config.vocab_size
    default_config["rms_norm_eps"] = model_config.norm_eps
    default_config["qk_norm"] = attn_cfg.qk_norm
    default_config["norm_everywhere"] = attn_cfg.norm_everywhere
    default_config["max_position_embeddings"] = model_config.rope.max_seq_len

    default_config["num_attention_heads"] = attention.n_heads
    default_config["num_key_value_heads"] = attention.n_kv_heads
    default_config["head_dim"] = attention.head_dim
    default_config["rope_theta"] = model_config.rope.theta
    default_config["rope_scaling"] = _native_to_hf_rope_scaling(model_config.rope)

    default_config["hidden_size"] = model.tok_embeddings.weight.shape[1]

    default_config["n_dense_layers"] = model_config.layer.n_dense_layers

    # Gated attention type and SWA config
    default_config["gated_attention_type"] = getattr(
        attn_cfg, "gated_attention_type", None
    )
    default_config["gate_only"] = bool(getattr(attn_cfg, "gate_only", False))
    default_config["mid_norm_position"] = getattr(
        attn_cfg, "mid_norm_position", "after"
    )
    default_config["use_rope"] = bool(getattr(attn_cfg, "use_rope", True))
    default_config["sliding_window_size"] = getattr(attn_cfg, "sliding_window_size", -1)
    default_config["qk_rope_dim"] = getattr(
        attention, "qk_rope_dim", getattr(attn_cfg, "qk_rope_dim", attention.head_dim)
    )
    default_config["partial_rotary_factor"] = (
        default_config["qk_rope_dim"] / default_config["head_dim"]
    )

    default_config["residual_scale"] = getattr(
        model_config.layer, "residual_scale", "identity"
    )

    # Per-layer patterns (store as-is: None, str, or list)
    default_config["rope_pattern"] = getattr(model_config, "rope_pattern", None)
    default_config["swa_pattern"] = getattr(model_config, "swa_pattern", None)

    # Separate RoPE config for SWA layers (native model's rope_of_swa).
    # rope_theta_swa and rope_scaling_swa are fully independent from the primary rope.
    rope_of_swa = getattr(model_config, "rope_of_swa", None)
    default_config["rope_theta_swa"] = (
        float(rope_of_swa.theta) if rope_of_swa is not None else None
    )
    default_config["rope_scaling_swa"] = _native_to_hf_rope_scaling(rope_of_swa)

    if model_config.layer.n_dense_layers > 0:
        default_config["intermediate_size"] = model_config.layer.feed_forward.hidden_dim

    if len(model.layers) > model_config.layer.n_dense_layers:
        moe = model.layers[str(len(model.layers) - 1)].moe
        default_config["moe_intermediate_size"] = model_config.layer.moe.hidden_dim
        default_config["n_active_experts"] = model_config.layer.moe.top_k
        default_config["n_total_experts"] = model_config.layer.moe.num_experts
        default_config["moe_scaling_factor"] = moe.scaling_factor
        default_config["n_shared_experts"] = model_config.layer.moe.num_shared_experts

    return default_config
