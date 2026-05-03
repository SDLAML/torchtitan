# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import json
import os
import shutil
from dataclasses import fields, is_dataclass, replace as dc_replace
from pathlib import Path
from typing import get_args

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


def _resolve_dataclass_type(annotation):
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    for candidate in get_args(annotation):
        if isinstance(candidate, type) and is_dataclass(candidate):
            return candidate
    return None


def _build_dataclass_from_dict(dataclass_type, values: dict, *, field_path: str):
    if not isinstance(values, dict):
        raise ValueError(f"Expected {field_path} to be a JSON object.")

    kwargs = {}
    field_map = {f.name: f for f in fields(dataclass_type)}
    for key, value in values.items():
        if key not in field_map:
            raise ValueError(f"Unknown key '{field_path}.{key}' in job_config.")
        field_info = field_map[key]
        nested_type = _resolve_dataclass_type(field_info.type)
        if isinstance(value, dict):
            if nested_type is None:
                raise ValueError(f"Expected '{field_path}.{key}' to be a scalar value.")
            kwargs[key] = _build_dataclass_from_dict(
                nested_type,
                value,
                field_path=f"{field_path}.{key}",
            )
        else:
            kwargs[key] = value
    return dataclass_type(**kwargs)


def _apply_dataclass_overrides(target_obj, overrides: dict, *, field_path: str):
    if not is_dataclass(target_obj):
        raise ValueError(f"Expected {field_path} to be a dataclass instance.")
    if not isinstance(overrides, dict):
        raise ValueError(f"Expected {field_path} to be a JSON object.")

    field_map = {f.name: f for f in fields(type(target_obj))}
    for key, value in overrides.items():
        if key not in field_map:
            raise ValueError(f"Unknown key '{field_path}.{key}' in job_config.")
        current_value = getattr(target_obj, key)
        field_info = field_map[key]
        nested_type = _resolve_dataclass_type(field_info.type)
        if isinstance(value, dict):
            if is_dataclass(current_value):
                _apply_dataclass_overrides(
                    current_value,
                    value,
                    field_path=f"{field_path}.{key}",
                )
            elif nested_type is not None:
                setattr(
                    target_obj,
                    key,
                    _build_dataclass_from_dict(
                        nested_type,
                        value,
                        field_path=f"{field_path}.{key}",
                    ),
                )
            else:
                raise ValueError(f"Expected '{field_path}.{key}' to be a scalar value.")
        else:
            setattr(target_obj, key, value)


def _load_job_config(job_config_path: Path) -> dict:
    if not job_config_path.exists():
        raise FileNotFoundError(f"job_config file does not exist: {job_config_path}")
    try:
        return json.loads(job_config_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed job_config JSON at {job_config_path}: {e}") from e


def _resolve_model_spec_for_conversion(
    *,
    model_name: str,
    model_flavor: str | None,
    job_config_path: "Path | None",
):
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    default_flavor = "bsc-1B-7B-opt-g"

    if model_name != "opt_moe":
        resolved_flavor = model_flavor or default_flavor
        return model_module.model_registry(resolved_flavor)

    if job_config_path is None:
        raise ValueError(
            "--job_config is required when converting opt_moe checkpoints to HF."
        )

    job_config = _load_job_config(job_config_path)
    model_spec_data = job_config.get("model_spec")
    if not isinstance(model_spec_data, dict):
        raise ValueError(
            f"Malformed job_config at {job_config_path}: missing object 'model_spec'."
        )

    job_config_flavor = model_spec_data.get("flavor")
    if not isinstance(job_config_flavor, str) or not job_config_flavor:
        raise ValueError(
            f"Malformed job_config at {job_config_path}: missing string "
            "'model_spec.flavor'."
        )

    if model_flavor is not None and model_flavor != job_config_flavor:
        raise ValueError(
            "opt_moe conversion flavor mismatch: "
            f"--model_flavor={model_flavor!r} but "
            f"job_config.model_spec.flavor={job_config_flavor!r}."
        )

    model_overrides = model_spec_data.get("model")
    if not isinstance(model_overrides, dict):
        raise ValueError(
            f"Malformed job_config at {job_config_path}: missing object "
            "'model_spec.model'."
        )

    model_spec = model_module.model_registry(job_config_flavor)
    _apply_dataclass_overrides(
        model_spec.model,
        model_overrides,
        field_path="model_spec.model",
    )
    # Sync training.seq_len → rope.max_seq_len so max_position_embeddings in the
    # exported HF config reflects the actual training context length, not the default.
    # Mirrors OPTMoEModel.Config.update_from_config (model.py:283-285).
    training_seq_len = job_config.get("training", {}).get("seq_len")
    if isinstance(training_seq_len, int) and training_seq_len > 0:
        if getattr(model_spec.model, "rope", None) is not None:
            model_spec.model.rope = dc_replace(
                model_spec.model.rope, max_seq_len=training_seq_len
            )
        if getattr(model_spec.model, "rope_of_swa", None) is not None:
            model_spec.model.rope_of_swa = dc_replace(
                model_spec.model.rope_of_swa, max_seq_len=training_seq_len
            )
    return model_spec


def _validate_exported_hf_config(
    *,
    model_name: str,
    model_config,
    output_dir: Path,
):
    if model_name != "opt_moe":
        return

    from torchtitan.models.opt_moe.hf_assests.setup_hf import (
        get_hf_config_overrides_from_model_config,
    )

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected HF config at {config_path}, but it was not created."
        )

    exported_config = json.loads(config_path.read_text())
    expected_fields = get_hf_config_overrides_from_model_config(None, model_config)
    expected_fields["rope_pattern"] = _normalize_layer_pattern_for_validation(
        expected_fields.get("rope_pattern")
    )
    expected_fields["swa_pattern"] = _normalize_layer_pattern_for_validation(
        expected_fields.get("swa_pattern")
    )

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
            f"HF config export lost opt_moe runtime config settings: {mismatch_text}."
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
    model_flavor: "str | None",
    hf_assets_path: "Path | None",
    export_dtype: str,
    job_config: "Path | None" = None,
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
    model_spec = _resolve_model_spec_for_conversion(
        model_name=model_name,
        model_flavor=model_flavor,
        job_config_path=job_config,
    )
    model_flavor = model_spec.flavor

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
        default=None,
        help="Model flavor / config key. For opt_moe this must match job_config.",
    )
    parser.add_argument(
        "--job_config",
        type=Path,
        default=None,
        help="Path to the saved job_config_*.json. Required for opt_moe.",
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
        args.job_config,
    )
