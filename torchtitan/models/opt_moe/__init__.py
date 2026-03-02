# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import RoPE

from torchtitan.protocols.model_spec import ModelSpec
from .gated_norm_swattention import GatedNormSWAttention
from .hf_assests import setup_hf

from .model import OPTMoEModel, OPTMoETransformerBlock
from .norm_ffn import FeedForward
from .norm_moe import MoE

from .parallelize import parallelize_opt_moe
from .state_dict_adapter import OPTMoEStateDictAdapter

__all__ = ["parallelize_opt_moe", "OPTMoEModel", "moe_opt_moe_configs"]


moe_opt_moe_configs = {
    "dense-1B-Proxy-8layers-test": OPTMoEModel.Config(
        n_layers=8,
        dim=256,
        rope_of_swa=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=1000000.0,
            backend="cos_sin",
        ),
        rope_pattern="NNRRRRNN",
        swa_pattern="SSSSFFFF",
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=8,
            feed_forward=FeedForward.Config(
                hidden_dim=704,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
            attention=GatedNormSWAttention.Config(
                n_heads=2,
                n_kv_heads=1,
                head_dim=128,
                qk_norm=True,
                norm_everywhere=True,
                norm_eps=1e-30,
                sliding_window_size=128,
                attn_backend="flex",
                gated_attention_type="head-wise",
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=10000.0,
            backend="cos_sin",
        ),
    ),
    "dense-1B-Proxy-8layers": OPTMoEModel.Config(
        n_layers=8,
        dim=256,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=8,
            feed_forward=FeedForward.Config(
                hidden_dim=704,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
            attention=GatedNormSWAttention.Config(
                n_heads=2,
                n_kv_heads=1,
                head_dim=128,
                qk_norm=True,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=10000.0,
            backend="cos_sin",
        ),
    ),
    "dense-1B": OPTMoEModel.Config(
        n_layers=24,
        dim=2048,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=24,
            feed_forward=FeedForward.Config(
                hidden_dim=4096,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
            attention=GatedNormSWAttention.Config(
                n_heads=16,
                n_kv_heads=8,
                head_dim=128,
                qk_norm=True,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=10000.0,
            backend="cos_sin",
        ),
    ),
    "bsc-1B-7B-opt-g": OPTMoEModel.Config(
        n_layers=24,
        dim=2048,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=1,
            feed_forward=FeedForward.Config(
                hidden_dim=5120,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
            moe=MoE.Config(
                hidden_dim=640,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                norm_everywhere=True,
                scaling_factor=2.8232,
            ),
            attention=GatedNormSWAttention.Config(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=10000.0,
            backend="cos_sin",
        ),
    ),
    "bsc-1B-7B-opt-g-proxy": OPTMoEModel.Config(
        n_layers=24,
        dim=512,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=1,
            feed_forward=FeedForward.Config(
                hidden_dim=1280,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
            moe=MoE.Config(
                hidden_dim=160,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                norm_everywhere=True,
                scaling_factor=2.8232,
            ),
            attention=GatedNormSWAttention.Config(
                n_heads=8,
                n_kv_heads=1,
                head_dim=128,
                qk_norm=True,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=10000.0,
            backend="cos_sin",
        ),
    ),
    "bsc-1B-7B-opt-g-proxy-8layers": OPTMoEModel.Config(
        n_layers=8,
        dim=512,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=1,
            feed_forward=FeedForward.Config(
                hidden_dim=1280,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
            moe=MoE.Config(
                hidden_dim=160,
                num_experts=64,
                num_shared_experts=1,
                top_k=8,
                norm_everywhere=True,
                scaling_factor=2.8232,
            ),
            attention=GatedNormSWAttention.Config(
                n_heads=8,
                n_kv_heads=1,
                head_dim=128,
                qk_norm=True,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=10000.0,
            backend="cos_sin",
        ),
    ),
    "qwen30b-a3b-8layers": OPTMoEModel.Config(
        n_layers=8,
        dim=2048,
        layer=OPTMoETransformerBlock.Config(
            n_dense_layers=0,
            moe=MoE.Config(
                hidden_dim=768,
                num_experts=128,
                num_shared_experts=1,
                top_k=8,
                norm_everywhere=True,
                scaling_factor=2.8232,
            ),
            attention=GatedNormSWAttention.Config(
                n_heads=32,
                n_kv_heads=4,
                head_dim=128,
                qk_norm=True,
                norm_everywhere=True,
                norm_eps=1e-30,
            ),
        ),
        rope=RoPE.Config(
            dim=128,
            max_seq_len=4096,
            theta=10000.0,
            backend="cos_sin",
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="opt_moe",
        flavor=flavor,
        model=copy.deepcopy(moe_opt_moe_configs[flavor]),
        parallelize_fn=parallelize_opt_moe,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=OPTMoEStateDictAdapter,
        hf_assets_setup_fn=setup_hf.copy_and_overwrite_model_config,
    )
