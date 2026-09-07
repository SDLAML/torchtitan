# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Declarative sharding for OPT MoE.

Upstream replaced the imperative ``parallelize_module``/``ExpertParallel``
plans with ``ShardingConfig`` objects attached to each sub-config, resolved at
``Module.parallelize`` time. Root modules (embedding, final norm, lm_head) and
the per-layer norms are stock upstream modules, so they use the shared helpers.

OPT MoE's attention (``GatedNormSWAttention``) and MoE (``norm_moe.MoE``) are
custom modules, not ``GQAttention``/``common.moe.MoE``, so the upstream
per-module TP/EP sharding helpers do not apply to them. Those axes are rejected
explicitly rather than silently producing a wrong plan -- see the guard in
``set_opt_moe_sharding_config``.
"""

from typing import TYPE_CHECKING

from torchtitan.models.common.decoder_sharding import (
    norm_config,
    set_decoder_sharding_config,
)

if TYPE_CHECKING:
    from torchtitan.models.opt_moe.model import OPTMoEModel


def set_opt_moe_sharding_config(
    config: "OPTMoEModel.Config",
    *,
    enable_sp: bool,
    enable_tp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on the OPT MoE sub-configs that support it.

    Args:
        config: The expanded model config (``_expand_layers`` must have run).
        enable_sp: Whether sequence parallelism is on (implies TP > 1).
        enable_tp: Whether tensor parallelism is on.
        enable_ep: Whether expert parallelism is on.
    """
    if enable_tp or enable_ep:
        raise NotImplementedError(
            "OPT MoE on the 0.5.0 base does not yet carry tensor/expert "
            "parallel sharding. Upstream's set_gqa_attention_sharding and "
            "set_moe_sharding_config target GQAttention and common.moe.MoE; "
            "GatedNormSWAttention and norm_moe.MoE need their own sharding "
            "plans before TP/EP can be enabled. Run with "
            "tensor_parallel_degree=1 and expert_parallel_degree=1 (FSDP/HSDP "
            "and CP are unaffected)."
        )

    set_decoder_sharding_config(config, enable_sp=enable_sp)

    norm = norm_config(enable_sp=enable_sp)
    for layer_cfg in config.layers:
        layer_cfg.attention_norm.sharding_config = norm
        layer_cfg.ffn_norm.sharding_config = norm
