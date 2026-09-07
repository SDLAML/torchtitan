# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torch import nn

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

        self._param_init = self._build_param_init()

    def forward(self, x):
        return self.w2(self.mid_norm(self.act_fn(self.w1(x)) * self.w3(x)))

    def _build_param_init(self) -> dict:
        """Per-parameter initializers, replacing the old init_weights cascade."""
        cfg = self.config
        w3_residual_div = cfg.residual_div if cfg.init_gate_as_residual else 1.0
        return {
            "w1.weight": make_param_init(cfg.w1_init_fn_type, cfg.w1_init_std),
            "w2.weight": make_param_init(
                cfg.w2_init_fn_type, cfg.w2_init_std, cfg.residual_div
            ),
            "w3.weight": make_param_init(
                cfg.w3_init_fn_type, cfg.w3_init_std, w3_residual_div
            ),
        }

    def _init_self_parameters(self) -> None:
        for name, param in self.named_parameters(recurse=True):
            init_fn = self._param_init.get(name)
            if init_fn is not None:
                init_fn(param)
