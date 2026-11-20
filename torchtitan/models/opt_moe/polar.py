# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OPT MoE projections and gated activations for the symmetry studies."""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from torchtitan.models.common.activation import BinaryActivationFn
from torchtitan.models.common.linear import Linear
from .hf.polar import cosine_linear, square_polar

# D = model width, H = heads, K = head width.
# I and O are the logical input and output dimensions of a linear projection.


class PolarLinear(Linear):
    """Apply the square polar factor of each logical projection to the input."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        qkv_head_dim: int | None = None
        inplace_wgrad_accum: bool = False

        def __post_init__(self):
            width = self.in_features
            qkv = self.qkv_head_dim
            if (
                self.bias
                or self.inplace_wgrad_accum
                or self.out_features != (1 if qkv is None else 3) * width
                or (
                    qkv is not None
                    and (qkv <= 0 or width % qkv or self.num_linears != 1)
                )
            ):
                raise ValueError(
                    "PolarLinear needs square, bias-free logical projections and ordinary autograd"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.qkv_head_dim = config.qkv_head_dim

    def _linear(self, input, weight, bias):
        if self.qkv_head_dim is None:
            # w13 stores two contiguous square matrices. wo and w2 store one.
            polar_OI = square_polar(weight).flatten(0, -2)
        else:
            # QKV rows interleave q, k and v within each head. Take three
            # independent polar factors, then restore exactly that row order.
            k = self.qkv_head_dim
            d = self.in_features
            qkv_H3KD = weight.view(-1, 3, k, d)
            matrices = qkv_H3KD.transpose(0, 1).reshape(3, d, d)
            polar_OI = (
                square_polar(matrices)
                .view(3, -1, k, d)
                .transpose(0, 1)
                .reshape_as(weight)
            )
        return F.linear(input.to(weight.dtype), polar_OI)


class CosineLinear(Linear):
    """L2-normalized input and vocabulary rows with a positive logit scale."""

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        initial_logit_scale: float
        logit_scale_trainable: bool
        inplace_wgrad_accum: bool = False

        def __post_init__(self):
            if (
                not math.isfinite(self.initial_logit_scale)
                or self.initial_logit_scale <= 0
            ):
                raise ValueError("initial_logit_scale must be finite and positive")
            if self.bias or self.num_linears != 1 or self.inplace_wgrad_accum:
                raise ValueError(
                    "CosineLinear needs one bias-free projection and ordinary autograd"
                )

    def __init__(self, config: Config):
        super().__init__(config)
        self.initial_log_scale = math.log(config.initial_logit_scale)
        # fully_shard wraps the decoder head even on one device and needs a dimension to shard.
        self.logit_scale = nn.Parameter(
            torch.full((1,), self.initial_log_scale),
            requires_grad=config.logit_scale_trainable,
        )

    def _init_self_parameters(self):
        if self._param_init is None:
            self.reset_parameters()
        else:
            self._init_param("weight", self.get_parameter("weight"))
        nn.init.constant_(self.logit_scale, self.initial_log_scale)

    def _linear(self, input, weight, bias):
        return cosine_linear(input, weight, self.logit_scale)


class ReGLU(BinaryActivationFn):
    @dataclass(kw_only=True, slots=True)
    class Config(BinaryActivationFn.Config):
        pass

    def __init__(self, config: Config):
        pass

    def _activation_fn(self, gate, up, /):
        return F.relu(gate) * up
