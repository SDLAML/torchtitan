# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Slow reference layers for the square-polar scale-invariance experiment.

Raw parameters keep their usual ``weight`` names. Optimizer metrics therefore
measure the raw weights, not the normalized matrices used by the forward pass.
"""

import math

import torch
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn import functional as F


class _SquarePolar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
            raise ValueError("The polar forward parameterization requires a square matrix")
        with torch.autocast(device_type=weight.device.type, enabled=False):
            # A custom backward avoids the differences of squared singular values
            # in generic SVD autograd, which are singular at orthogonal initialization.
            kwargs = {"driver": "gesvd"} if weight.is_cuda else {}
            u, s, vh = torch.linalg.svd(weight.double(), full_matrices=False, **kwargs)
            ctx.save_for_backward(u, s, vh)
            ctx.weight_dtype = weight.dtype
            return (u @ vh).to(weight.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        u, s, vh = ctx.saved_tensors
        with torch.autocast(device_type=grad_output.device.type, enabled=False):
            # If B = U.T @ dW @ V, then U.T @ dQ @ V has entries
            # (B_ij - B_ji) / (s_i + s_j). This derivative is self-adjoint
            # in these coordinates and remains regular when singular values tie.
            c = u.mT @ grad_output.double() @ vh.mT
            grad = u @ ((c - c.mT) / (s[:, None] + s[None, :])) @ vh
            return grad.to(ctx.weight_dtype)


def square_polar(weight: torch.Tensor) -> torch.Tensor:
    """Polar factor of a full-rank square weight, with an FP64 SVD/backward."""
    return _SquarePolar.apply(weight)


class PolarLinear(nn.Linear):
    """Use Q(weight) in forward; retain an unconstrained square raw weight."""

    def __init__(self, in_features, out_features, bias=False, **kwargs):
        if in_features != out_features or bias:
            raise ValueError("PolarLinear requires square weights and bias=False")
        super().__init__(in_features, out_features, bias=False, **kwargs)

    def forward(self, input):
        dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=input.device.type, enabled=False):
            return F.linear(input.to(dtype), square_polar(self.weight).to(dtype))


class CosineLinear(nn.Linear):
    """Normalized output rows and inputs, with a learned positive temperature."""

    def __init__(self, in_features, out_features, *, initial_logit_scale=1.0):
        if not math.isfinite(initial_logit_scale) or initial_logit_scale <= 0:
            raise ValueError("initial_logit_scale must be finite and positive")
        super().__init__(in_features, out_features, bias=False)
        self.initial_logit_scale = initial_logit_scale
        self.logit_scale = nn.Parameter(torch.empty(()))
        self.reset_logit_scale()

    def reset_logit_scale(self):
        # Called again after Trainer materializes modules constructed on meta.
        nn.init.constant_(self.logit_scale, math.log(self.initial_logit_scale))

    def forward(self, input):
        dtype = torch.float64 if self.weight.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=input.device.type, enabled=False):
            weight = F.normalize(self.weight.to(dtype), dim=-1, eps=1e-30)
            hidden = F.normalize(input.to(dtype), dim=-1, eps=1e-30)
            return F.linear(hidden, weight) * self.logit_scale.to(dtype).exp()
