# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import json

import os, shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import save_torch_state_dict
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP


def _normalize_layer_pattern_for_validation(pattern):
    """Match HF export normalization for per-layer pattern fields."""
    if pattern is None:
        return None
    if isinstance(pattern, tuple):
        pattern = list(pattern)
    if isinstance(pattern, list):
        if len(pattern) == 1 and isinstance(pattern[0], str):
            return pattern[0]
        if pattern and all(isinstance(x, str) and len(x) == 1 for x in pattern):
            return "".join(pattern)
    return pattern


def _validate_exported_hf_config(
    *,
    model_name: str,
    model_config,
    output_dir: Path,
):
    if model_name != "opt_moe":
        return

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected HF config at {config_path}, but it was not created."
        )

    exported_config = json.loads(config_path.read_text())
    attention_config = model_config.layer.attention
    expected_qk_rope_dim = getattr(attention_config, "qk_rope_dim", None)
    if expected_qk_rope_dim is None:
        expected_qk_rope_dim = (
            getattr(attention_config, "head_dim", None)
            or model_config.dim // attention_config.n_heads
        )
    expected_fields = {
        "gate_only": bool(getattr(attention_config, "gate_only", False)),
        "mid_norm_position": getattr(attention_config, "mid_norm_position", "after"),
        "qk_rope_dim": expected_qk_rope_dim,
        # HF router is always fp32 by design.
        "force_router_on_fp32": True,
        "rope_pattern": _normalize_layer_pattern_for_validation(
            getattr(model_config, "rope_pattern", None)
        ),
        "swa_pattern": _normalize_layer_pattern_for_validation(
            getattr(model_config, "swa_pattern", None)
        ),
    }

    mismatches = {
        field: (expected_value, exported_config.get(field))
        for field, expected_value in expected_fields.items()
        if exported_config.get(field) != expected_value
    }
    if mismatches:
        mismatch_text = ", ".join(
            f"{field}: expected {expected!r}, got {actual!r}"
            for field, (expected, actual) in mismatches.items()
        )
        raise ValueError(
            f"HF config export lost opt_moe attention settings: {mismatch_text}."
        )


def try_to_copy_tokenizer(output_dir, hf_assets_path):
    """
    if these files exist in the hf_assets_path, then copy them to the output_dir
    """
    if hf_assets_path is None:
        return

    tokenizer_assests_lists = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    ]
    for asset in tokenizer_assests_lists:
        if os.path.exists(os.path.join(hf_assets_path, asset)):
            shutil.copy(
                os.path.join(hf_assets_path, asset), os.path.join(output_dir, asset)
            )


@torch.inference_mode()
def convert_to_hf(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    model_flavor: str,
    hf_assets_path: "Path | None",
    export_dtype: str,
):
    """Convert a DCP checkpoint to HuggingFace safetensors format.

    Steps:
      1. Load ModelSpec from the model registry.
      2. Build an empty CPU model and wrap it.
      3. Create a state dict adapter.
      4. Load the DCP checkpoint.
      5. Convert native → HF state dict.
      6. Optionally cast dtype.
      7. Write HF safetensors.
      8. Copy HF config/modeling files and generate config.json.
    """
    # 1. Get ModelSpec from the model registry
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)

    # 2. Build empty model on CPU
    model_config = model_spec.model
    with torch.device("cpu"):
        actual_model = model_config.build()
    model_config = getattr(actual_model, "config", model_config)
    model = ModelWrapper(actual_model)

    # 3. Create state dict adapter (new API: model_config, not model_args)
    assert model_spec.state_dict_adapter is not None, (
        "state_dict_adapter is required for HF checkpoint conversion. "
        f"Model '{model_name}/{model_flavor}' has none registered."
    )
    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)

    # 4. Load DCP checkpoint into empty state dict
    state_dict = model._get_state_dict()
    dcp.load(state_dict, checkpoint_id=str(input_dir))

    # 5. Convert native → HF state dict
    hf_state_dict = sd_adapter.to_hf(state_dict)

    # 6. Apply export dtype if requested
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    # 7. Write HF safetensors
    output_dir.mkdir(parents=True, exist_ok=True)
    save_torch_state_dict(
        hf_state_dict,
        output_dir,
        max_shard_size="5GB",
        safe_serialization=True,
        metadata={"format": "pt"},
    )

    # 8. Copy HF config/modeling files and generate config.json
    if model_spec.hf_assets_setup_fn is not None:
        model_spec.hf_assets_setup_fn(actual_model, model_config, str(output_dir))
        _validate_exported_hf_config(
            model_name=model_name,
            model_config=model_config,
            output_dir=output_dir,
        )
    else:
        print(
            f"[WARNING] No hf_assets_setup_fn registered for '{model_name}/{model_flavor}'. "
            "Skipping config.json generation."
        )

    # hf_assets_setup_fn will create a dummy chat-template.jinja,
    # we need to copy the tokenizer files to the output_dir
    # to maybe override the dummy chat-template.jinja
    try_to_copy_tokenizer(output_dir, hf_assets_path)

    print(f"model is saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a DCP checkpoint to HuggingFace safetensors format."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory containing the DCP checkpoint.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the HF checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="opt_moe",
        help="Model module name under torchtitan.models (default: opt_moe).",
    )
    parser.add_argument(
        "--model_flavor",
        type=str,
        default="bsc-1B-7B-opt-g",
        help="Model flavor / config key (default: bsc-1B-7B-opt-g).",
    )
    parser.add_argument(
        "--hf_assets_path",
        type=Path,
        default=None,
        help="Path to a pre-existing HF assets directory containing "
        "model.safetensors.index.json for fqn_to_index_mapping.",
    )
    parser.add_argument(
        "--export_dtype",
        type=str,
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Export dtype for HF checkpoint (default: float32).",
    )
    args = parser.parse_args()

    convert_to_hf(
        args.input_dir,
        args.output_dir,
        args.model_name,
        args.model_flavor,
        args.hf_assets_path,
        args.export_dtype,
    )
