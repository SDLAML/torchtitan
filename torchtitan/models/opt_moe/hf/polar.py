# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Square polar and cosine projections shared by the native and HF models."""

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

# D and I are the output and input dimensions of each square matrix.
# T is a token, V a vocabulary row. Leading dimensions may batch matrices.


class _SquarePolar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_DI):  # pyrefly: ignore[bad-override]
        if weight_DI.ndim < 2 or weight_DI.shape[-2] != weight_DI.shape[-1]:
            raise ValueError("The polar parameterization requires square matrices")
        with torch.autocast(device_type=weight_DI.device.type, enabled=False):
            driver = {"driver": "gesvd"} if weight_DI.is_cuda else {}
            u, s, vh = torch.linalg.svd(
                weight_DI.double(), full_matrices=False, **driver
            )
            ctx.save_for_backward(u, s, vh)
            ctx.weight_dtype = weight_DI.dtype
            return (u @ vh).to(weight_DI.dtype)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_DI):  # pyrefly: ignore[bad-override]
        u, s, vh = ctx.saved_tensors
        with torch.autocast(device_type=grad_DI.device.type, enabled=False):
            # The polar derivative divides by sums of singular values. Unlike
            # generic SVD autograd, it is regular at orthogonal initialization.
            c = u.mT @ grad_DI.double() @ vh.mT
            grad = u @ ((c - c.mT) / (s.unsqueeze(-1) + s.unsqueeze(-2))) @ vh
            return grad.to(ctx.weight_dtype)


def square_polar(weight_DI: torch.Tensor) -> torch.Tensor:
    """Polar factor of each full-rank square matrix, with an FP64 derivative."""
    return _SquarePolar.apply(weight_DI)


def cosine_linear(input_TD, weight_VD, logit_scale):
    dtype = torch.float64 if weight_VD.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=input_TD.device.type, enabled=False):
        rows_VD = F.normalize(weight_VD.to(dtype), dim=-1, eps=1e-30)
        hidden_TD = F.normalize(input_TD.to(dtype), dim=-1, eps=1e-30)
        # Scale the hidden state so backward never saves a second logits tensor.
        hidden_TD = hidden_TD * logit_scale.to(dtype).exp()
    return F.linear(hidden_TD.to(weight_VD.dtype), rows_VD.to(weight_VD.dtype))
