# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch import nn

from torchtitan.protocols.module import Module
from .utils.activations import build_activation

from .utils.inits import build_init_fn
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

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, config.hidden_dim, bias=False)
        self.act_fn = build_activation(config.activation_type)

        if config.norm_everywhere:
            self.mid_norm = build_norm(
                config.norm_type, dim=config.hidden_dim, eps=config.norm_eps
            )
        else:
            self.mid_norm = nn.Identity()

    def forward(self, x):
        return self.w2(self.mid_norm(self.act_fn(self.w1(x)) * self.w3(x)))

    def init_weights(
        self,
        residual_div: float,
        init_gate_as_residual: bool,
        skip_init: bool = False,
    ):
        if not isinstance(self.mid_norm, nn.Identity):
            self.mid_norm.reset_parameters()
        if skip_init:
            return

        w1_init_fn = build_init_fn(self.config.w1_init_fn_type)
        w2_init_fn = build_init_fn(self.config.w2_init_fn_type)
        w3_init_fn = build_init_fn(self.config.w3_init_fn_type)

        w1_init_fn(self.w1.weight, mean=0.0, std=self.config.w1_init_std)
        w2_init_fn(self.w2.weight, mean=0.0, std=self.config.w2_init_std / residual_div)

        w3_init_std = (
            self.config.w3_init_std / residual_div
            if init_gate_as_residual
            else self.config.w3_init_std
        )
        w3_init_fn(self.w3.weight, mean=0.0, std=w3_init_std)
