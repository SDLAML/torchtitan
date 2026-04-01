# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.rope import RoPE
from torchtitan.models.opt_moe.gated_norm_swattention import GatedNormSWAttention
from torchtitan.models.opt_moe.hf_assests.configuration_opt_moe import OptMoEConfig
from torchtitan.models.opt_moe.hf_assests.modeling_opt_moe import OptMoEAttention
from torchtitan.models.opt_moe.hf_assests.setup_hf import overwrite_config
from torchtitan.models.opt_moe.model import OPTMoEModel, OPTMoETransformerBlock
from torchtitan.models.opt_moe.norm_ffn import FeedForward
from torchtitan.models.opt_moe.norm_moe import MoE
from torchtitan.models.opt_moe.state_dict_adapter import OPTMoEStateDictAdapter


def _build_native_model_config(
    *,
    gate_only: bool = False,
    mid_norm_position: str = "after",
    qk_rope_dim: int | None = None,
) -> OPTMoEModel.Config:
    head_dim = 4
    return OPTMoEModel.Config(
        dim=8,
        n_layers=1,
        vocab_size=32,
        norm_eps=1e-30,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=1,
            norm_eps=1e-30,
            attention=GatedNormSWAttention.Config(
                n_heads=2,
                n_kv_heads=2,
                head_dim=head_dim,
                qk_norm=False,
                norm_everywhere=True,
                gated_attention_type="head-wise",
                gate_only=gate_only,
                mid_norm_position=mid_norm_position,
                use_rope=False,
                qk_rope_dim=qk_rope_dim,
                attn_backend="sdpa",
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=16,
                norm_everywhere=False,
            ),
        ),
        rope=RoPE.Config(
            dim=qk_rope_dim or head_dim,
            max_seq_len=8,
            theta=10000.0,
            backend="cos_sin",
        ),
    )


def _build_hf_attention(
    *,
    gate_only: bool = False,
    mid_norm_position: str = "after",
    qk_rope_dim: int | None = None,
) -> OptMoEAttention:
    config = OptMoEConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=8,
        qk_norm=False,
        norm_everywhere=True,
        gated_attention_type="head-wise",
        gate_only=gate_only,
        mid_norm_position=mid_norm_position,
        use_rope=False,
        qk_rope_dim=qk_rope_dim,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    attention = OptMoEAttention(config=config, layer_idx=0)
    attention.use_rope = False
    return attention


def _build_native_moe_model_config() -> OPTMoEModel.Config:
    return OPTMoEModel.Config(
        dim=12,
        n_layers=4,
        vocab_size=64,
        norm_eps=1e-5,
        rope_pattern="RRNR",
        swa_pattern="SSFS",
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=0,
            norm_eps=1e-5,
            residual_scale="depth_scale",
            attention=GatedNormSWAttention.Config(
                n_heads=3,
                n_kv_heads=1,
                head_dim=4,
                qk_norm=True,
                norm_everywhere=True,
                gated_attention_type="head-wise",
                gate_only=True,
                mid_norm_position="before",
                use_rope=True,
                qk_rope_dim=2,
                sliding_window_size=16,
                attn_backend="flex",
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=18,
                norm_everywhere=False,
                activation_type="silu",
            ),
            moe=MoE.Config(
                hidden_dim=24,
                num_experts=16,
                num_shared_experts=2,
                top_k=4,
                scaling_factor=1.0,
                norm_everywhere=True,
                activation_type="silu",
            ),
        ),
        rope=RoPE.Config(
            dim=2,
            max_seq_len=32,
            theta=10000.0,
            backend="cos_sin",
        ),
        rope_of_swa=RoPE.Config(
            dim=2,
            max_seq_len=32,
            theta=5000.0,
            backend="cos_sin",
        ),
    )


class _ConfigOnlyModel:
    __slots__ = ("config",)

    def __init__(self, config):
        self.config = config


def _copy_attention_weights(
    native_attention: GatedNormSWAttention,
    hf_attention: OptMoEAttention,
) -> None:
    with torch.no_grad():
        hf_attention.q_proj.weight.copy_(native_attention.wq.weight)
        hf_attention.k_proj.weight.copy_(native_attention.wk.weight)
        hf_attention.v_proj.weight.copy_(native_attention.wv.weight)
        hf_attention.o_proj.weight.copy_(native_attention.wo.weight)
        hf_attention.gate_proj.weight.copy_(native_attention.gate_proj.weight)


class TestOptMoEHFAssets(unittest.TestCase):
    def test_overwrite_config_preserves_opt_moe_attention_flags(self):
        model_config = _build_native_model_config(
            gate_only=True,
            mid_norm_position="before",
            qk_rope_dim=2,
        )
        model = model_config.build()

        exported = overwrite_config(model)

        self.assertTrue(exported["gate_only"])
        self.assertEqual(exported["mid_norm_position"], "before")
        self.assertEqual(exported["qk_rope_dim"], 2)
        self.assertEqual(exported["partial_rotary_factor"], 0.5)

    def test_hf_attention_matches_native_for_new_attention_modes(self):
        for gate_only, mid_norm_position in ((True, "after"), (False, "before")):
            with self.subTest(
                gate_only=gate_only,
                mid_norm_position=mid_norm_position,
            ):
                torch.manual_seed(0)
                native_attention = _build_native_model_config(
                    gate_only=gate_only,
                    mid_norm_position=mid_norm_position,
                ).layer.attention.build(dim=8)
                hf_attention = _build_hf_attention(
                    gate_only=gate_only,
                    mid_norm_position=mid_norm_position,
                )

                _copy_attention_weights(native_attention, hf_attention)

                hidden_states = torch.randn(2, 1, 8, dtype=torch.float32)

                native_output = native_attention(
                    hidden_states,
                    rope_cache=torch.empty(0),
                    attention_masks=None,
                )
                hf_output, _ = hf_attention(
                    hidden_states=hidden_states,
                    position_embeddings=None,
                    attention_mask=None,
                )

                torch.testing.assert_close(native_output, hf_output)

    def test_state_dict_adapter_rejects_unsupported_norm_tensors(self):
        adapter = OPTMoEStateDictAdapter(_build_native_model_config(), None)

        with self.assertRaisesRegex(ValueError, "learned norm tensors"):
            adapter.to_hf({"layers.0.attention.mid_norm.weight": torch.ones(4)})

    def test_overwrite_config_uses_runtime_model_config_without_module_inspection(self):
        model_config = _build_native_moe_model_config()

        exported = overwrite_config(_ConfigOnlyModel(model_config))

        self.assertEqual(exported["hidden_size"], 12)
        self.assertEqual(exported["num_hidden_layers"], 4)
        self.assertEqual(exported["num_attention_heads"], 3)
        self.assertEqual(exported["num_key_value_heads"], 1)
        self.assertEqual(exported["head_dim"], 4)
        self.assertEqual(exported["qk_rope_dim"], 2)
        self.assertEqual(exported["partial_rotary_factor"], 0.5)
        self.assertEqual(exported["intermediate_size"], 18)
        self.assertEqual(exported["moe_intermediate_size"], 24)
        self.assertEqual(exported["moe_scaling_factor"], 1.0)
        self.assertEqual(exported["n_active_experts"], 4)
        self.assertEqual(exported["n_total_experts"], 16)
        self.assertEqual(exported["n_shared_experts"], 2)
        self.assertEqual(exported["rope_pattern"], "RRNR")
        self.assertEqual(exported["swa_pattern"], "SSFS")
        self.assertEqual(exported["rope_theta_swa"], 5000.0)
        self.assertEqual(exported["hidden_act"], "silu")
        self.assertEqual(exported["residual_scale"], "depth_scale")
