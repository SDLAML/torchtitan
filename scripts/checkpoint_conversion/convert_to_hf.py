# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torchtitan.components.checkpoint import ModelWrapper
from torchtitan.config import TORCH_DTYPE_MAP


def _apply_config_overrides(model_config, config_path: Path):
    """Load JSON config overrides and apply them to model_config in-place.

    This is used to match the exact hyperparameters of a saved checkpoint
    instead of the default values registered in model_registry.
    """
    overrides = json.loads(config_path.read_text())

    # Zero out expensive init functions so CPU model construction is cheap
    def _zero_init_fns(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if "init_fn_type" in k:
                    d[k] = "zeros"
                elif isinstance(v, dict):
                    _zero_init_fns(v)

    _zero_init_fns(overrides)

    from torchtitan.tools.config_utils import update_dataclass_from_dict

    update_dataclass_from_dict(model_config, overrides)


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
            "HF config export lost opt_moe attention settings: " f"{mismatch_text}."
        )


@torch.inference_mode()
def convert_to_hf(
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    model_flavor: str,
    hf_assets_path: "Path | None",
    export_dtype: str,
    model_config_path: "Path | None" = None,
):
    """Convert a DCP checkpoint to HuggingFace safetensors format.

    Steps:
      1. Load ModelSpec from the model registry.
      2. Optionally apply config overrides from a JSON file.
      3. Build an empty CPU model and wrap it.
      4. Create a state dict adapter.
      5. Load the DCP checkpoint.
      6. Convert native → HF state dict.
      7. Optionally cast dtype.
      8. Write HF safetensors via HuggingFaceStorageWriter.
      9. Copy HF config/modeling files and generate config.json.
    """
    # 1. Get ModelSpec from the model registry
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_spec = model_module.model_registry(model_flavor)

    # 2. Optionally apply config overrides
    model_config = model_spec.model
    if model_config_path is not None:
        _apply_config_overrides(model_config, model_config_path)

    # 3. Build empty model on CPU
    with torch.device("cpu"):
        model = model_config.build()
    model = ModelWrapper(model)

    # 4. Create state dict adapter (new API: model_config, not model_args)
    assert model_spec.state_dict_adapter is not None, (
        "state_dict_adapter is required for HF checkpoint conversion. "
        f"Model '{model_name}/{model_flavor}' has none registered."
    )
    sd_adapter = model_spec.state_dict_adapter(model_config, hf_assets_path)

    # 5. Load DCP checkpoint into empty state dict
    state_dict = model._get_state_dict()
    dcp.load(state_dict, checkpoint_id=str(input_dir))

    # 6. Convert native → HF state dict
    hf_state_dict = sd_adapter.to_hf(state_dict)

    # 7. Apply export dtype if requested
    target_dtype = TORCH_DTYPE_MAP[export_dtype]
    if target_dtype != torch.float32:
        hf_state_dict = {k: v.to(target_dtype) for k, v in hf_state_dict.items()}

    # 8. Write HF safetensors
    output_dir.mkdir(parents=True, exist_ok=True)
    storage_writer = HuggingFaceStorageWriter(
        path=str(output_dir),
        save_distributed=True,
        fqn_to_index_mapping=sd_adapter.fqn_to_index_mapping,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    dcp.save(hf_state_dict, storage_writer=storage_writer)

    # 9. Copy HF config/modeling files and generate config.json
    if model_spec.hf_assets_setup_fn is not None:
        model_spec.hf_assets_setup_fn(model.module, model_config, str(output_dir))
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
        "--model_config_path",
        type=Path,
        default=None,
        help="Optional JSON file with model config overrides (e.g. saved "
        "checkpoint config). Used to match the exact training hyperparameters.",
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
        args.model_config_path,
    )
