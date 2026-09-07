# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch import nn

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Identity
from torchtitan.protocols.module import Module
from .utils.activations import build_activation

from .utils.inits import make_param_init
from .utils.norms import build_norm


class FeedForward(Module):
    """SwiGLU feed-forward module shared across models.

    Config takes the **final** hidden_dim (no internal 2/3 scaling).
    Use compute_ffn_hidden_dim() for Llama3/4-style dim computation.
    Runtime ``dim`` is passed as a build() kwarg.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_dim: int
        norm_everywhere: bool = False
        norm_type: str = "np_rmsnorm"
        norm_eps: float = 1e-30
        activation_type: str = "silu"

        w1_init_fn_type: str = "scaled_orthogonal"
        w2_init_fn_type: str = "scaled_orthogonal"
        w3_init_fn_type: str = "scaled_orthogonal"

        w1_init_std: float = 1.0
        w2_init_std: float = 1.0
        w3_init_std: float = 1.0

        # Stamped per layer by OPTMoEModel.Config expansion; these used to be
        # build()/init_weights() arguments.
        dim: int = 0
        residual_div: float = 1.0
        init_gate_as_residual: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dim = config.dim
        assert dim > 0, (
            "FeedForward.Config.dim must be stamped by the model's config "
            "expansion before build()."
        )
        # Configurable Linear rather than nn.Linear: every submodule has to
        # satisfy the Module protocol, and param_init then lives on the config.
        w3_residual_div = config.residual_div if config.init_gate_as_residual else 1.0
        self.w1 = Linear.Config(
            in_features=dim,
            out_features=config.hidden_dim,
            param_init={
                "weight": make_param_init(config.w1_init_fn_type, config.w1_init_std)
            },
        ).build()
        self.w2 = Linear.Config(
            in_features=config.hidden_dim,
            out_features=dim,
            param_init={
                "weight": make_param_init(
                    config.w2_init_fn_type, config.w2_init_std, config.residual_div
                )
            },
        ).build()
        self.w3 = Linear.Config(
            in_features=dim,
            out_features=config.hidden_dim,
            param_init={
                "weight": make_param_init(
                    config.w3_init_fn_type, config.w3_init_std, w3_residual_div
                )
            },
        ).build()
        self.act_fn = build_activation(config.activation_type)

        if config.norm_everywhere:
            self.mid_norm = build_norm(
                config.norm_type, dim=config.hidden_dim, eps=config.norm_eps
            )
        else:
            self.mid_norm = Identity.Config().build()

    def forward(self, x):
        return self.w2(self.mid_norm(self.act_fn(self.w1(x)) * self.w3(x)))
