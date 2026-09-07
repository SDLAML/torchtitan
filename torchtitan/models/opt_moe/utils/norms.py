# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import numbers
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.common.nn_modules import LayerNorm, RMSNorm
from torchtitan.protocols.module import Module


class SingleScaleRMSNorm(Module):
    """AKA SSNorm, from https://arxiv.org/abs/2506.19697."""

    __constants__ = ["normalized_shape", "eps"]
    normalized_shape: tuple[int, ...]
    eps: float | None

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-6

    def __init__(self, config: Config) -> None:
        super().__init__()
        normalized_shape = config.normalized_shape
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = config.eps

        self.ssnorm_scale = nn.Parameter(torch.empty((1,), dtype=torch.float32))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = F.rms_norm(input, self.normalized_shape, eps=self.eps)
        return self.ssnorm_scale * out

    def reset_parameters(self) -> None:
        nn.init.ones_(self.ssnorm_scale)

    def extra_repr(self) -> str:
        return f"normalized_shape={self.normalized_shape}, eps={self.eps}"


# Every OPT MoE norm type maps onto a configurable upstream module, so norms are
# first-class ``Module``s: ``Module.parallelize`` can shard them and
# ``init_states`` initializes them, neither of which worked when they were bare
# ``nn.Module`` instances built by a lambda.
NORM_CONFIGS = {
    "layernorm": lambda dim, eps: LayerNorm.Config(normalized_shape=dim, eps=eps),
    "np_layernorm": lambda dim, eps: LayerNorm.Config(
        normalized_shape=dim, eps=eps, elementwise_affine=False
    ),
    "rmsnorm": lambda dim, eps: RMSNorm.Config(normalized_shape=dim, eps=eps),
    "np_rmsnorm": lambda dim, eps: RMSNorm.Config(
        normalized_shape=dim, eps=eps, elementwise_affine=False
    ),
    "ss_rmsnorm": lambda dim, eps: SingleScaleRMSNorm.Config(
        normalized_shape=dim, eps=eps
    ),
}


def build_norm_config(norm_type: str, dim: int, eps: float = 1e-6):
    """Return the ``Module.Config`` for the requested norm type.

    Args:
        norm_type: One of ``layernorm``, ``np_layernorm``, ``rmsnorm``,
            ``np_rmsnorm``, ``ss_rmsnorm``.
        dim: Normalized dimension.
        eps: Epsilon for numerical stability.

    Raises:
        NotImplementedError: If an unknown ``norm_type`` is provided.
    """
    norm_config_fn = NORM_CONFIGS.get(norm_type.lower())
    if norm_config_fn is None:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")
    return norm_config_fn(dim, eps)


def build_norm(norm_type: str, dim: int, eps: float = 1e-6):
    """Build a norm module directly. Prefer ``build_norm_config`` where the
    config tree is available, so the module participates in sharding."""
    return build_norm_config(norm_type, dim, eps).build()
