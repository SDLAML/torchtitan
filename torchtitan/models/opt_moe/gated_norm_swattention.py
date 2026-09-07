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
    FusedQKVLinear,
    GQAttention,
    QKVLinear,
    ScaledDotProductAttention,
    VarlenAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Identity
from torchtitan.models.common.rope import CosSinRoPE, RoPE
from torchtitan.protocols.module import Module
from .utils.inits import make_param_init
from .utils.norms import build_norm, build_norm_config


class GatedNormSWAttention(GQAttention):
    """Gated Norm Sliding Window Attention, built on upstream's GQAttention.

    Token-flat: inputs are ``[T, D]`` and Q/K/V are ``[T, H, K]``. There is no
    batch dimension; document boundaries travel in ``positions`` and in the flex
    ``BlockMask``.

    Q/K/V projection, output projection, RoPE, QK-norm and the inner attention
    backend all come from ``GQAttention`` -- including its pluggable
    ``qkv_linear``, so fused QKV is available by setting ``fuse_qkv``. Only what
    upstream has no equivalent for is added here: V-norm, mid-norm, sigmoid
    gating and partial RoPE.

    The config keeps the flat, string-valued authoring surface the flavor
    registry uses (``attn_backend="flex"``, ``qk_norm=True``, ...) and
    synthesizes upstream's structured ``GQAttention.Config`` in
    ``to_gqa_config``. Upstream's field of the same name, ``qk_norm``, is a norm
    config rather than a bool, which is why the two are not merged.
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

        fuse_qkv: bool = False
        """Project Q/K/V with one fused GEMM instead of three.

        Off by default, which is the historical behaviour and keeps wq/wk/wv as
        separate checkpoint entries. Upstream's FusedQKVLinear still saves and
        loads in the split layout, so this is a pure throughput switch.
        """

        def to_gqa_config(self) -> "GQAttention.Config":
            """Synthesize upstream's GQAttention.Config from these flat fields."""
            assert self.dim > 0, (
                "GatedNormSWAttention.Config.dim must be stamped by the model's "
                "config expansion before build()."
            )
            n_kv = self.n_heads if self.n_kv_heads is None else self.n_kv_heads
            head_dim = (
                self.head_dim if self.head_dim is not None else self.dim // self.n_heads
            )
            wq_init = {
                "weight": make_param_init(self.wq_init_fn_type, self.wq_init_std)
            }
            wkv_init = {
                "weight": make_param_init(self.wk_init_fn_type, self.wk_init_std)
            }
            if (self.wk_init_fn_type, self.wk_init_std) != (
                self.wv_init_fn_type,
                self.wv_init_std,
            ):
                # QKVLinear builds wk and wv from one shared config, so it cannot
                # express different initializers for them. Fail loudly rather
                # than silently applying wk's to both.
                raise ValueError(
                    "wk and wv initializers must match "
                    f"(got wk={self.wk_init_fn_type}/{self.wk_init_std}, "
                    f"wv={self.wv_init_fn_type}/{self.wv_init_std}): upstream's "
                    "QKVLinear shares one config between them."
                )

            if self.fuse_qkv:
                if (self.wq_init_fn_type, self.wq_init_std) != (
                    self.wk_init_fn_type,
                    self.wk_init_std,
                ):
                    # The fused init applies one initializer to q, k and v, so
                    # fusing would silently change wq's or wk's initialization.
                    raise ValueError(
                        "fuse_qkv requires wq, wk and wv to share an initializer "
                        f"(got wq={self.wq_init_fn_type}/{self.wq_init_std}, "
                        f"wk={self.wk_init_fn_type}/{self.wk_init_std}). Leave "
                        "fuse_qkv off to keep them independent."
                    )
                # Upstream's _fused_qkv_param_init reproduces the exact draws the
                # split wq/wk/wv would make, so fusing is a pure throughput
                # change and does not perturb initialization.
                from torchtitan.models.common.config_utils import (
                    _fused_qkv_param_init,
                )

                qkv = FusedQKVLinear.Config(
                    head_dim=head_dim,
                    n_heads=self.n_heads,
                    n_kv_heads=n_kv,
                    wqkv=Linear.Config(
                        in_features=self.dim,
                        out_features=(self.n_heads + 2 * n_kv) * head_dim,
                        param_init=_fused_qkv_param_init(
                            wq_init,
                            n_heads=self.n_heads,
                            n_kv_heads=n_kv,
                            head_dim=head_dim,
                        ),
                    ),
                )
            else:
                qkv = QKVLinear.Config(
                    head_dim=head_dim,
                    wq=Linear.Config(
                        in_features=self.dim,
                        out_features=self.n_heads * head_dim,
                        param_init=wq_init,
                    ),
                    wkv=Linear.Config(
                        in_features=self.dim,
                        out_features=n_kv * head_dim,
                        param_init=wkv_init,
                    ),
                )

            qk_norm = None
            if self.qk_norm or self.norm_everywhere:
                qk_norm = build_norm_config(self.norm_type, head_dim, self.norm_eps)

            return GQAttention.Config(
                n_heads=self.n_heads,
                n_kv_heads=n_kv,
                head_dim=self.head_dim,
                dim=self.dim,
                qkv_linear=qkv,
                wo=Linear.Config(
                    in_features=self.n_heads * head_dim,
                    out_features=self.dim,
                    param_init={
                        "weight": make_param_init(
                            self.wo_init_fn_type, self.wo_init_std, self.residual_div
                        )
                    },
                ),
                qk_norm=qk_norm,
                inner_attention=self.inner_attention,
                rope=self.rope,
            )

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
        # GQAttention builds qkv_linear, wo, rope, inner_attention, q/k norm and
        # the scaling factor. Only the pieces upstream has no equivalent for are
        # added below.
        super().__init__(config.to_gqa_config())
        self.config = config

        if self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
            )
        self.use_rope = config.use_rope
        self.qk_rope_dim: int = (
            config.qk_rope_dim if config.qk_rope_dim is not None else self.head_dim
        )
        assert self.qk_rope_dim <= self.head_dim, (
            f"qk_rope_dim ({self.qk_rope_dim}) must be <= head_dim ({self.head_dim})"
        )
        self.partial_rope = self.qk_rope_dim != self.head_dim
        if not self.use_rope:
            # GQAttention always builds a RoPE; drop it so NoPE layers carry no
            # unused cache and cannot accidentally apply it.
            self.rope = None

        self.gated_attention_type = config.gated_attention_type
        self.gate_only = config.gate_only
        self.sliding_window_size = config.sliding_window_size
        self.mid_norm_position = config.mid_norm_position
        assert self.mid_norm_position in ("after", "before"), (
            f"mid_norm_position ({self.mid_norm_position}) must be 'after' or 'before'"
        )
        self.head_wise_mid_norm = config.head_wise_mid_norm
        self.attn_backend = config.attn_backend

        identity = Identity.Config()
        self.v_norm = identity.build()
        self.mid_norm = identity.build()
        self.gate_proj = identity.build()

        dim = config.dim
        if config.v_norm or config.norm_everywhere:
            self.v_norm = build_norm_config(
                config.norm_type, self.head_dim, config.norm_eps
            ).build()
        if config.mid_norm or config.norm_everywhere:
            mid_dim = (
                self.n_heads * self.head_dim
                if self.mid_norm_position == "after" and not self.head_wise_mid_norm
                else self.head_dim
            )
            self.mid_norm = build_norm_config(
                config.norm_type, mid_dim, config.norm_eps
            ).build()

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
        # Q/K/V projection comes from GQAttention's pluggable qkv_linear, so
        # fused and split projections are both available.
        xq, xk, xv = self.qkv_linear(x_TD)

        # QK norm before RoPE (per Qwen3); V norm has no upstream equivalent.
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
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
            # Gate in float32 for numeric stability, then cast back.
            orig_dtype = output.dtype
            gate = torch.sigmoid(self.gate_proj(x_TD).float())
            if self.gated_attention_type == "head-wise":
                # gate: [T, n_local_heads] -> broadcast over head_dim
                output = (output.float() * gate.unsqueeze(-1)).to(orig_dtype)
            elif self.gated_attention_type == "element-wise":
                # gate: [T, n_local_heads * head_dim]
                output = (output.reshape(num_tokens, -1).float() * gate).to(orig_dtype)

        # "after" mid-norm: per-head or over the concatenated head output.
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
