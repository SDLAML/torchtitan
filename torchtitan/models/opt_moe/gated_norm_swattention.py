# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.models.common.rope import (
    apply_rotary_emb_complex,
    apply_rotary_emb_cos_sin,
)
from .utils.inits import build_init_fn
from .utils.norms import build_norm


class GatedNormSWAttention(BaseAttention):
    """Gated Norm Sliding Window Attention module shared across OPT MoE."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int | None = None
        head_dim: int | None = None
        qk_norm: bool = False
        norm_everywhere: bool = False
        gated_attention_type: str | None = None  # "none", "head-wise", "element-wise"
        norm_eps: float = 1e-30
        norm_type: str = "np_rmsnorm"
        use_rope: bool = True
        attn_backend: str = "sdpa"
        attn_mask_type: str = "causal"

        rope_backend: str = "cos_sin"  # "complex" or "cos_sin"
        sliding_window_size: int = -1

        wq_init_fn_type: str = "scaled_orthogonal"
        wk_init_fn_type: str = "scaled_orthogonal"
        wv_init_fn_type: str = "scaled_orthogonal"
        wo_init_fn_type: str = "scaled_orthogonal"
        w_gate_init_fn_type: str = "scaled_orthogonal"

        wq_init_std: float = 1.0
        wk_init_std: float = 1.0
        wv_init_std: float = 1.0
        wo_init_std: float = 1.0
        w_gate_init_std: float = 1.0

    def __init__(self, config: Config, *, dim: int):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        self.head_dim = (
            config.head_dim if config.head_dim is not None else dim // config.n_heads
        )
        self.enable_gqa = self.n_heads > self.n_kv_heads
        self.use_rope = config.use_rope
        self.rope_backend = config.rope_backend

        self.gated_attention_type = config.gated_attention_type
        self.sliding_window_size = config.sliding_window_size

        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.v_norm = nn.Identity()
        self.mid_norm = nn.Identity()
        self.gate_proj = nn.Identity()

        build_attention_norm = partial(
            build_norm, norm_type=config.norm_type, eps=config.norm_eps
        )

        if config.qk_norm or config.norm_everywhere:
            self.q_norm = build_attention_norm(dim=self.head_dim)
            self.k_norm = build_attention_norm(dim=self.head_dim)
        if config.norm_everywhere:
            self.v_norm = build_attention_norm(dim=self.head_dim)
            self.mid_norm = build_attention_norm(dim=self.n_heads * self.head_dim)

        if self.gated_attention_type == "head-wise":
            # G1-style: one gate per attention head.
            self.gate_proj = nn.Linear(dim, self.n_heads, bias=False)
        elif self.gated_attention_type == "element-wise":
            # Dense gate over the full attention output channel dimension.
            self.gate_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)

        # Scaling factor (needed when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.attn_backend = config.attn_backend
        self.inner_attention: nn.Module
        match self.attn_backend:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_backend}")

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # Optional QK normalization (before RoPE, per Qwen3)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xv = self.v_norm(xv)

        # Apply rotary embeddings
        if self.use_rope:
            if self.rope_backend == "cos_sin":
                xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)
            else:
                xq, xk = apply_rotary_emb_complex(
                    xq, xk, freqs_cis=rope_cache, positions=positions
                )

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)  # (bs, n_kv_heads, seqlen, head_dim)

        scale_kwargs = {"scale": self.scaling} if self.scaling is not None else {}

        match self.attn_backend:
            case "flex":
                assert isinstance(attention_masks, BlockMask), attention_masks
                block_mask = attention_masks
                output = (
                    self.inner_attention(
                        xq,
                        xk,
                        xv,
                        block_mask=block_mask,
                        enable_gqa=self.enable_gqa,
                        **scale_kwargs,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq, xk, xv, attention_masks, **scale_kwargs
                )
                output = output.view(bs, seqlen, -1, self.head_dim)
            case "sdpa":
                assert attention_masks is None
                output = (
                    self.inner_attention(
                        xq,
                        xk,
                        xv,
                        enable_gqa=self.enable_gqa,
                        **scale_kwargs,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_backend}")

        if self.gated_attention_type is not None:
            gate = torch.sigmoid(self.gate_proj(x)).to(output.dtype)
            if self.gated_attention_type == "head-wise":
                # gate: [bs, seqlen, n_local_heads]
                output = output * gate.unsqueeze(-1)
            elif self.gated_attention_type == "element-wise":
                # gate: [bs, seqlen, n_local_heads * head_dim]
                output_flat = output.reshape(bs, seqlen, -1)
                output = output_flat * gate

        output = output.reshape(bs, seqlen, -1)
        output = self.mid_norm(output)
        return self.wo(output)

    def init_weights(self, residual_div: float):

        wq_init_fn = build_init_fn(self.config.wq_init_fn_type)
        wk_init_fn = build_init_fn(self.config.wk_init_fn_type)
        wv_init_fn = build_init_fn(self.config.wv_init_fn_type)

        wq_init_fn(self.wq.weight, mean=0.0, std=self.config.wq_init_std)
        wk_init_fn(self.wk.weight, mean=0.0, std=self.config.wk_init_std)
        wv_init_fn(self.wv.weight, mean=0.0, std=self.config.wv_init_std)

        wo_init_fn = build_init_fn(self.config.wo_init_fn_type)
        wo_init_fn(self.wo.weight, mean=0.0, std=self.config.wo_init_std / residual_div)

        for norm in (self.q_norm, self.k_norm, self.v_norm, self.mid_norm):
            if not isinstance(norm, nn.Identity):
                norm.reset_parameters()

        if self.gated_attention_type is not None:
            w_gate_init_fn = build_init_fn(self.config.w_gate_init_fn_type)
            w_gate_init_fn(
                self.gate_proj.weight,
                mean=0.0,
                std=self.config.w_gate_init_std / residual_div,
            )
