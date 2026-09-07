# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from functools import partial

import torch
from torch import nn

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    FlexAttention,
    ScaledDotProductAttention,
    VarlenAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Identity
from torchtitan.models.common.rope import CosSinRoPE, RoPE
from torchtitan.protocols.module import Module
from .utils.inits import make_param_init
from .utils.norms import build_norm


class GatedNormSWAttention(BaseAttention):
    """Gated Norm Sliding Window Attention module shared across OPT MoE.

    Token-flat: inputs are ``[T, D]`` and Q/K/V are ``[T, H, K]``, matching
    upstream's post-0.3 batch contract. There is no batch dimension; document
    boundaries are carried by ``positions`` and by the flex ``BlockMask``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        # ``dim`` used to arrive as a build kwarg from the block. Upstream builds
        # every layer config with a bare ``build()``, so the model's config
        # expansion stamps it onto each per-layer copy instead.
        dim: int = 0
        n_kv_heads: int | None = None
        head_dim: int | None = None
        qk_norm: bool = False
        mid_norm: bool = False
        v_norm: bool = False
        norm_everywhere: bool = False
        gated_attention_type: str | None = None  # "none", "head-wise", "element-wise"
        gate_only: bool = False
        norm_eps: float = 1e-30
        norm_type: str = "np_rmsnorm"
        mid_norm_position: str = "after"  # "after" or "before"
        head_wise_mid_norm: bool = False
        use_rope: bool = True
        attn_backend: str = "sdpa"
        attn_mask_type: str = "causal"

        rope_backend: str = "cos_sin"  # "complex" or "cos_sin"
        qk_rope_dim: int | None = None
        # Number of head dimensions to apply RoPE to.  None (default) means full
        # head_dim (standard RoPE).  Set to a smaller value for partial RoPE where
        # only the first qk_rope_dim dims are rotated and the rest pass through.
        sliding_window_size: int = -1

        # Filled in by OPTMoEModel.Config expansion from ``attn_backend`` and the
        # model-level rope/rope_of_swa configs. Declared with defaults so the
        # compact flavor registry in __init__.py does not have to spell them out.
        inner_attention: Module.Config = field(
            default_factory=ScaledDotProductAttention.Config
        )
        rope: RoPE.Config | None = None

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

        # Depth-init divisor for the output projections, stamped per layer by the
        # model's config expansion (was passed to init_weights(residual_div=...)).
        residual_div: float = 1.0

        def build_inner_attention(self) -> Module.Config:
            """Map ``attn_backend`` onto an upstream inner-attention config."""
            match self.attn_backend:
                case "flex":
                    return FlexAttention.Config()
                case "varlen":
                    return VarlenAttention.Config()
                case "sdpa":
                    return ScaledDotProductAttention.Config()
                case _:
                    raise ValueError(f"Unknown attention type: {self.attn_backend}")

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        dim = config.dim
        assert dim > 0, (
            "GatedNormSWAttention.Config.dim must be stamped by the model's "
            "config expansion before build()."
        )
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
        self.qk_rope_dim: int = (
            config.qk_rope_dim if config.qk_rope_dim is not None else self.head_dim
        )
        assert (
            self.qk_rope_dim <= self.head_dim
        ), f"qk_rope_dim ({self.qk_rope_dim}) must be less than or equal to head_dim ({self.head_dim})"
        self.partial_rope = self.qk_rope_dim != self.head_dim

        self.gated_attention_type = config.gated_attention_type
        self.gate_only = config.gate_only
        self.sliding_window_size = config.sliding_window_size
        self.mid_norm_position = config.mid_norm_position
        assert self.mid_norm_position in [
            "after",
            "before",
        ], f"mid_norm_position ({self.mid_norm_position}) must be either 'after' or 'before'"
        self.head_wise_mid_norm = config.head_wise_mid_norm

        identity = Identity.Config()
        self.q_norm = identity.build()
        self.k_norm = identity.build()
        self.v_norm = identity.build()
        self.mid_norm = identity.build()
        self.gate_proj = identity.build()

        build_attention_norm = partial(
            build_norm, norm_type=config.norm_type, eps=config.norm_eps
        )

        if config.qk_norm or config.norm_everywhere:
            self.q_norm = build_attention_norm(dim=self.head_dim)
            self.k_norm = build_attention_norm(dim=self.head_dim)
        if config.v_norm or config.norm_everywhere:
            self.v_norm = build_attention_norm(dim=self.head_dim)
        if config.mid_norm or config.norm_everywhere:
            if self.mid_norm_position == "after" and not self.head_wise_mid_norm:
                self.mid_norm = build_attention_norm(dim=self.n_heads * self.head_dim)
            else:
                self.mid_norm = build_attention_norm(dim=self.head_dim)

        gate_init = {
            "weight": make_param_init(
                config.w_gate_init_fn_type, config.w_gate_init_std, config.residual_div
            )
        }
        if self.gated_attention_type == "head-wise":
            # G1-style: one gate per attention head.
            self.gate_proj = Linear.Config(
                in_features=dim, out_features=self.n_heads, param_init=gate_init
            ).build()
        elif self.gated_attention_type == "element-wise":
            # Dense gate over the full attention output channel dimension.
            self.gate_proj = Linear.Config(
                in_features=dim,
                out_features=self.n_heads * self.head_dim,
                param_init=gate_init,
            ).build()

        # Scaling factor (needed when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

        self.wq = Linear.Config(
            in_features=dim,
            out_features=self.n_heads * self.head_dim,
            param_init={
                "weight": make_param_init(config.wq_init_fn_type, config.wq_init_std)
            },
        ).build()
        self.wk = Linear.Config(
            in_features=dim,
            out_features=self.n_kv_heads * self.head_dim,
            param_init={
                "weight": make_param_init(config.wk_init_fn_type, config.wk_init_std)
            },
        ).build()
        self.wv = Linear.Config(
            in_features=dim,
            out_features=self.n_kv_heads * self.head_dim,
            param_init={
                "weight": make_param_init(config.wv_init_fn_type, config.wv_init_std)
            },
        ).build()
        self.wo = Linear.Config(
            in_features=self.n_heads * self.head_dim,
            out_features=dim,
            param_init={
                "weight": make_param_init(
                    config.wo_init_fn_type, config.wo_init_std, config.residual_div
                )
            },
        ).build()

        self.attn_backend = config.attn_backend
        self.inner_attention = config.inner_attention.build()

        # RoPE is now owned by the attention module (upstream removed freqs_cis
        # from the block forward signature). For partial RoPE the cache is built
        # at qk_rope_dim so only the leading slice is rotated.
        self.rope: RoPE | None = None
        if self.use_rope:
            assert config.rope is not None, (
                "use_rope=True requires a rope config; the model's config "
                "expansion assigns the global or SWA-local cache per layer."
            )
            self.rope = config.rope.build()

    def _apply_rope(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.rope is not None
        if not self.partial_rope:
            return self.rope(xq_THK, xk_THK, positions)
        # Partial RoPE: rotate only the leading qk_rope_dim dims, pass the rest
        # through untouched.
        xq_rot, xq_pass = (
            xq_THK[..., : self.qk_rope_dim],
            xq_THK[..., self.qk_rope_dim :],
        )
        xk_rot, xk_pass = (
            xk_THK[..., : self.qk_rope_dim],
            xk_THK[..., self.qk_rope_dim :],
        )
        xq_rot, xk_rot = self.rope(xq_rot, xk_rot, positions)
        return (
            torch.cat([xq_rot, xq_pass], dim=-1),
            torch.cat([xk_rot, xk_pass], dim=-1),
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        xq, xk, xv = self.wq(x_TD), self.wk(x_TD), self.wv(x_TD)

        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(num_tokens, -1, self.head_dim)
        xk = xk.view(num_tokens, -1, self.head_dim)
        xv = xv.view(num_tokens, -1, self.head_dim)

        # Optional QK/V normalization (before RoPE, per Qwen3)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xv = self.v_norm(xv)

        if self.use_rope:
            xq, xk = self._apply_rope(xq, xk, positions)

        scale_kwargs = {"scale": self.scaling} if self.scaling is not None else {}

        if self.attn_backend == "sdpa":
            # Upstream's SDPA module still works in (B, L, H, K); add and drop
            # the singleton batch dim at the kernel boundary. Flex and varlen
            # are natively token-flat.
            assert attention_masks is None
            output = self.inner_attention(
                xq.unsqueeze(0),
                xk.unsqueeze(0),
                xv.unsqueeze(0),
                enable_gqa=self.enable_gqa,
                **scale_kwargs,
            ).squeeze(0)
        else:
            output = self.inner_attention(
                xq,
                xk,
                xv,
                attention_masks=attention_masks,
                enable_gqa=self.enable_gqa,
                **scale_kwargs,
            )
        output = output.contiguous()

        # "before" mid-norm: per-head norm applied before gating.
        # output shape here: [T, n_local_heads, head_dim]
        if self.mid_norm_position == "before" and not self.gate_only:
            output = self.mid_norm(output)

        if self.gated_attention_type is not None:
            # Compute gate and multiply in float32 for numeric stability, then cast back.
            orig_dtype = output.dtype
            gate = torch.sigmoid(self.gate_proj(x_TD).float())
            if self.gated_attention_type == "head-wise":
                # gate: [T, n_local_heads] -> broadcast over head_dim
                output = (output.float() * gate.unsqueeze(-1)).to(orig_dtype)
            elif self.gated_attention_type == "element-wise":
                # gate: [T, n_local_heads * head_dim]
                output_flat = output.reshape(num_tokens, -1).float()
                output = (output_flat * gate).to(orig_dtype)

        # "after" mid-norm: norm applied either per-head or over the full
        # concatenated head output, depending on head_wise_mid_norm.
        if self.mid_norm_position == "after" and not self.gate_only:
            if self.head_wise_mid_norm:
                output = output.reshape(num_tokens, -1, self.head_dim)
                output = self.mid_norm(output)
                output = output.reshape(num_tokens, -1)
            else:
                output = output.reshape(num_tokens, -1)
                output = self.mid_norm(output)
        else:
            output = output.reshape(num_tokens, -1)
        return self.wo(output)
