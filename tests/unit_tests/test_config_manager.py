# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import textwrap
import unittest

import pytest
from torchtitan.config import ConfigManager
from torchtitan.models.opt_moe import model_registry as opt_moe_model_registry
from torchtitan.trainer import Trainer


class TestConfigManager(unittest.TestCase):
    def test_model_config_args(self):
        """--module and --config together load the correct config."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"
        assert config.training.steps == 10

    def test_model_config_args_equals_form(self):
        """--module=X --config=Y form works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module=llama3", "--config=llama3_debugmodel"]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_model_without_config_errors(self):
        """--module alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--config is required"):
            config_manager.parse_args(["--module", "llama3"])

    def test_config_without_model_errors(self):
        """--config alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args(["--config", "llama3_debugmodel"])

    def test_external_config_file_without_module(self):
        """External config files can be loaded without --module."""
        config_manager = ConfigManager()
        with tempfile.TemporaryDirectory() as d:
            config_path = os.path.join(d, "external_cfg.py")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        from torchtitan.models.llama3.config_registry import llama3_debugmodel

                        def make_config():
                            cfg = llama3_debugmodel()
                            cfg.training.steps = 12
                            return cfg
                        """
                    ).strip()
                )

            config = config_manager.parse_args(["--config", config_path])
            assert config.model_spec.name == "llama3"
            assert config.training.steps == 12

    def test_external_config_file_with_explicit_function(self):
        """External config supports '/path/to/file.py:function_name' form."""
        config_manager = ConfigManager()
        with tempfile.TemporaryDirectory() as d:
            config_path = os.path.join(d, "external_cfg.py")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        from torchtitan.models.llama3.config_registry import llama3_debugmodel

                        def alpha():
                            cfg = llama3_debugmodel()
                            cfg.training.steps = 13
                            return cfg

                        def beta():
                            cfg = llama3_debugmodel()
                            cfg.training.steps = 14
                            return cfg
                        """
                    ).strip()
                )

            config = config_manager.parse_args(["--config", f"{config_path}:beta"])
            assert config.model_spec.name == "llama3"
            assert config.training.steps == 14

    def test_external_config_normalizes_scalar_list_fields(self):
        """Scalar assignments for list-typed config fields are normalized early."""
        config_manager = ConfigManager()
        with tempfile.TemporaryDirectory() as d:
            config_path = os.path.join(d, "external_cfg.py")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        from torchtitan.models.opt_moe.config_registry import opt_moe_1b_7b_proxy

                        def make_config():
                            cfg = opt_moe_1b_7b_proxy()
                            cfg.dataloader.dataset = "simple_custom"
                            cfg.dataloader.dataset_path = "/tmp/some_dataset_path"
                            cfg.dataloader.dataset_split = "train"
                            cfg.dataloader.dataset_key = "text"
                            return cfg
                        """
                    ).strip()
                )

            config = config_manager.parse_args(["--config", config_path])
            assert config.dataloader.dataset == ["simple_custom"]
            assert config.dataloader.dataset_path == ["/tmp/some_dataset_path"]
            assert config.dataloader.dataset_split == ["train"]
            assert config.dataloader.dataset_key == ["text"]

    def test_missing_both_errors(self):
        """No --module or --config raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args([])

    def test_invalid_model_errors(self):
        """--module with unknown module name raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="Unknown module"):
            config_manager.parse_args(["--module", "nonexistent", "--config", "foo"])

    def test_invalid_config_errors(self):
        """--config with unknown function name lists available functions."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="Available config functions"):
            config_manager.parse_args(["--module", "llama3", "--config", "nonexistent"])

    def test_cli_overrides(self):
        """CLI args override config defaults."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.steps",
                "5",
            ]
        )
        assert config.training.steps == 5

    def test_cli_override_dump_folder(self):
        """CLI args override config defaults for nested fields."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--dump_folder",
                "/tmp/test_tt/",
            ]
        )
        assert config.dump_folder == "/tmp/test_tt/"

    def test_parse_module_fqns_per_model_part(self):
        """module_fqns_per_model_part defaults to None."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.parallelism.module_fqns_per_model_part is None

    def test_parse_exclude_from_loading(self):
        """exclude_from_loading defaults to [] and can be overridden."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.checkpoint.exclude_from_loading == []

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--checkpoint.exclude_from_loading",
                "optimizer,lr_scheduler",
            ]
        )
        assert config.checkpoint.exclude_from_loading == [
            "optimizer",
            "lr_scheduler",
        ]

    def test_trainer_config_model_converters_default(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.model_converters.converters == []

    # TODO: remove this test when we remove the merge functionality
    def test_extend_trainer_config_directly(self):
        """Test that _merge_configs works to extend config types."""
        from dataclasses import dataclass

        @dataclass
        class CustomCheckpoint:
            convert_path: str = "/custom/path"
            fake_model: bool = True

        @dataclass
        class CustomTrainerConfig:
            checkpoint: CustomCheckpoint

        MergedTrainerConfig = ConfigManager._merge_configs(
            Trainer.Config, CustomTrainerConfig
        )

        # Verify the merged type has both base and custom fields
        merged = MergedTrainerConfig()
        assert hasattr(merged, "checkpoint")
        assert hasattr(merged.checkpoint, "convert_path")
        assert merged.checkpoint.convert_path == "/custom/path"
        assert merged.checkpoint.fake_model is True
        assert hasattr(merged, "model_spec")

    def test_flux_config_via_cli(self):
        """Test that --module flux --config flux_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "flux", "--config", "flux_debugmodel"]
        )
        assert config.model_spec.name == "flux"
        assert hasattr(config, "encoder")

    def test_deepseek_config(self):
        """Test that --module deepseek_v3 --config deepseek_v3_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "deepseek_v3", "--config", "deepseek_v3_debugmodel"]
        )
        assert config.model_spec.name == "deepseek_v3"
        assert config.model_spec.flavor == "debugmodel"

    def test_model_flavor_override_flag(self):
        """--model-flavor overrides model_spec via model_registry."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "opt_moe",
                "--config",
                "opt_moe_1b_7b_proxy",
                "--model-flavor",
                "1B-7B-Proxy-8layers",
            ]
        )
        assert config.model_spec.name == "opt_moe"
        assert config.model_spec.flavor == "1B-7B-Proxy-8layers"
        assert config.model_spec.model.n_layers == 8

    def test_model_flavor_override_dot_alias(self):
        """--model.flavor alias behaves the same as --model-flavor."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "opt_moe",
                "--config",
                "opt_moe_1b_7b_proxy",
                "--model.flavor=1B-7B-Proxy",
            ]
        )
        assert config.model_spec.name == "opt_moe"
        assert config.model_spec.flavor == "1B-7B-Proxy"

    def test_model_flavor_override_preserves_external_model_overrides(self):
        """When flavor is overridden, external model field overrides are retained."""
        config_manager = ConfigManager()
        with tempfile.TemporaryDirectory() as d:
            config_path = os.path.join(d, "external_cfg.py")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(
                    textwrap.dedent(
                        """
                        from torchtitan.models.opt_moe.config_registry import opt_moe_1b_7b_proxy

                        def make_config():
                            cfg = opt_moe_1b_7b_proxy()
                            cfg.model_spec.model.layer.depth_init = "relative_depth"
                            cfg.model_spec.model.layer.residual_scale = "complete_p"
                            return cfg
                        """
                    ).strip()
                )

            config = config_manager.parse_args(
                [
                    "--config",
                    config_path,
                    "--model-flavor",
                    "1B-7B-Proxy-8layers",
                ]
            )

            assert config.model_spec.flavor == "1B-7B-Proxy-8layers"
            assert config.model_spec.model.layer.depth_init == "relative_depth"
            assert config.model_spec.model.layer.residual_scale == "complete_p"

    def test_opt_moe_model_registry_returns_independent_config_objects(self):
        """Mutating one returned model config must not affect later registry calls."""
        spec_a = opt_moe_model_registry("1B-7B-Proxy")
        spec_a.model.layer.depth_init = "relative_depth"
        spec_a.model.layer.residual_scale = "complete_p"

        spec_b = opt_moe_model_registry("1B-7B-Proxy")
        assert spec_b.model.layer.depth_init == "total_depth"
        assert spec_b.model.layer.residual_scale == "identity"


if __name__ == "__main__":
    unittest.main()
