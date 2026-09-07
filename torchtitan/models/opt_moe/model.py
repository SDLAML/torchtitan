# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses as _dc
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks

from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttention,
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Identity
from torchtitan.models.common.rope import RoPE
from torchtitan.models.utils import (
    get_nparams_and_active_nparams,
    quadratic_attention_flops_per_token,
)
from torchtitan.tools.logging import logger
from .gated_norm_swattention import GatedNormSWAttention
from .utils.inits import (
    build_init_fn,
    make_param_init,
    parse_depth_init,
    setup_depth_init,
    setup_residual_scale,
)
from .utils.norms import build_norm_config


def _parse_layer_pattern(
    pattern: "str | list[bool] | list[str] | None",
    n_layers: int,
    true_char: str,
    false_char: str,
    default_true: bool = False,
) -> "list[bool]":
    """Parse a per-layer pattern into a flat list of booleans.

    Args:
        pattern: A string (one char per layer, e.g. ``'SSSF'`` or ``'RRRN'``),
                 a ``list[bool]``, a list-wrapped string (e.g. ``['SSSF']``),
                 or ``None``.
        n_layers: Expected number of layers; length is validated.
        true_char: Character that maps to ``True`` (case-insensitive).
        false_char: Character that maps to ``False`` (case-insensitive).
        default_true: Value returned for every layer when ``pattern`` is ``None``.
    """
    if pattern is None:
        return [default_true] * n_layers
    if isinstance(pattern, list):
        # Some config frontends pass single string values as one-item lists.
        # Accept both ['SSSF'] and ['S','S','S','F'] in addition to list[bool].
        if len(pattern) == 1 and isinstance(pattern[0], str):
            pattern = pattern[0]
        elif pattern and all(isinstance(x, str) and len(x) == 1 for x in pattern):
            pattern = "".join(pattern)
        else:
            if len(pattern) != n_layers:
                raise ValueError(
                    f"Pattern list length {len(pattern)} != n_layers {n_layers}"
                )
            if not all(isinstance(x, bool) for x in pattern):
                raise ValueError(
                    "Pattern list must be list[bool], ['PATTERN'], or list of single-character strings."
                )
            return list(pattern)
    # String path
    allowed = {true_char.upper(), false_char.upper()}
    pattern_up = pattern.upper()
    if len(pattern_up) != n_layers:
        raise ValueError(
            f"Pattern string length {len(pattern_up)} != n_layers {n_layers}"
        )
    invalid = set(pattern_up) - allowed
    if invalid:
        raise ValueError(
            f"Invalid characters {invalid!r} in pattern. Expected only {allowed!r}."
        )
    return [c == true_char.upper() for c in pattern_up]


class OPTMoETransformerBlock(TransformerBlock):
    """OPT MoE TransformerBlock.

    Token-flat throughout: ``x`` is ``[T, D]``. Returns ``(output, lbl_loss)``
    so the model can accumulate per-layer MoE load-balance losses without
    threading a running tensor through every layer signature.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # ``TransformerBlock.Config`` requires these; OPT MoE builds its norms
        # from ``norm_type``/``norm_eps`` instead, so they stay unset.
        attention_norm: Any = None
        ffn_norm: Any = None

        n_dense_layers: int = 0
        init_gate_as_residual: bool = False
        depth_init: bool | str = "total_depth"
        residual_scale: str = "identity"
        norm_eps: float = 1e-30
        norm_type: str = "np_rmsnorm"

        # Stamped per layer by OPTMoEModel.Config._expand_layers(). Upstream
        # builds each layer with a bare build(), so these can no longer be
        # build() kwargs.
        dim: int = 0
        layer_id: int = 0
        n_layers: int = 1

    def __init__(self, config: Config):
        super().__init__()
        dim = config.dim
        assert dim > 0, (
            "OPTMoETransformerBlock.Config.dim must be stamped by "
            "OPTMoEModel.Config._expand_layers() before build()."
        )
        self.layer_id = config.layer_id
        self.attention = config.attention.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

        # Per-layer attention-mode flags (derived from the per-layer attention config).
        assert isinstance(config.attention, GatedNormSWAttention.Config)
        self.use_swa: bool = config.attention.sliding_window_size > 0
        self.attn_backend: str = config.attention.attn_backend

        # Pre-compute the mask dict key used in forward() so we avoid string
        # comparisons and conditional logic on every training step.
        #   None  → SDPA (no mask; PyTorch applies causal masking internally)
        #   "swa" → FlexAttention with sliding-window mask
        #   "full"→ FlexAttention with full-causal mask
        if self.attn_backend == "sdpa":
            self._mask_key: str | None = None
        elif self.use_swa:
            self._mask_key = "swa"
        else:
            self._mask_key = "full"

        # Per-layer debug metadata surfaced in __repr__/print(model).
        self._repr_use_rope: bool = config.attention.use_rope
        self._repr_swa_window_size: int = (
            config.attention.sliding_window_size if self.use_swa else -1
        )
        self._repr_rope_theta: float = (
            float(config.attention.rope.theta)
            if config.attention.use_rope and config.attention.rope is not None
            else -1.0
        )

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        # x = identity_scale * x + block_scale * block(x)
        self.block_scale, self.identity_scale = setup_residual_scale(
            config.residual_scale, config.n_layers
        )

    def extra_repr(self) -> str:
        return (
            f"layer_id={self.layer_id}, "
            f"rope_theta={self._repr_rope_theta}, "
            f"use_rope={self._repr_use_rope}, "
            f"qk_rope_dim={self.attention.qk_rope_dim}, "
            f"swa_window_size={self._repr_swa_window_size}"
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """Forward pass through the block.

        Args:
            x: Input tensor ``[T, D]``.
            attention_masks: Mask(s) for this layer; a dict keyed ``full``/``swa``
                when layers mix backends, or a single mask.
            positions: Optional position indices ``[T]``.
            loss_mask: Optional token loss mask for the MoE load-balance loss.

        Returns:
            ``(output, lbl_loss)`` -- ``lbl_loss`` is this layer's load-balance
            loss on MoE layers and ``None`` on dense layers.
        """
        # _mask_key is pre-computed at init; no string comparisons or tensor ops here.
        if self._mask_key is None:
            layer_mask = None  # SDPA: causal masking is handled internally by PyTorch
        elif isinstance(attention_masks, dict):
            layer_mask = attention_masks[self._mask_key]
        else:
            layer_mask = attention_masks  # single BlockMask (backward compat)

        h = self.identity_scale * x + self.block_scale * self.attention(
            self.attention_norm(x), layer_mask, positions
        )

        if self.moe_enabled:
            mlp_output, lbl_loss = self.moe(self.ffn_norm(h), loss_mask)
        else:
            mlp_output = self.feed_forward(self.ffn_norm(h))
            lbl_loss = None

        return self.identity_scale * h + self.block_scale * mlp_output, lbl_loss


class OPTMoEModel(Decoder):
    """OPT MoE Transformer model with attention and feed-forward layers."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        dim: int = 2048
        n_layers: int = 24
        vocab_size: int = 201088
        layer: TransformerBlock.Config

        # Upstream's Decoder.Config requires these outright. OPT MoE describes a
        # model with one `layer` template plus per-layer patterns, so they are
        # derived in _expand_layers() instead of being spelled out per flavor.
        layers: list = field(default_factory=list)
        lm_head: Any = None
        tok_embeddings: Any = None
        norm: Any = None

        rope: RoPE.Config
        norm_eps: float = 1e-30
        norm_type: str = "np_rmsnorm"

        first_in_init_fn_type: str = "scion_normal_input"
        first_in_init_std: float = 1.0

        final_out_init_fn_type: str = "scion_normal_output"
        final_out_init_std: float = 1.0

        use_embeddings_norm: bool = False

        # Mirrored from training config in update_from_config; consumed by
        # preprocess_inputs to build the MoE load-balance token mask.
        enable_token_mask_for_moe: bool = False

        rope_of_swa: RoPE.Config | None = None
        # RoPE cache for SWA layers (typically a lower theta for local context).
        # The global ``rope`` config applies to full-attention layers.
        # None → all layers share the global ``rope`` cache.
        # Ignored for NoPE SWA layers.

        rope_pattern: str | list[bool] | None = None
        # Per-layer RoPE vs NoPE selection.
        # String: one char per layer — 'R' = RoPE, 'N' = NoPE.  E.g. ``"RRRN"``.
        # list[bool]: True = RoPE, False = NoPE.
        # None keeps the uniform behaviour from ``layer.attention.use_rope``.

        swa_pattern: str | list[bool] | None = None
        # Per-layer sliding-window vs full-attention selection.
        # String: one char per layer — 'S' = SWA, 'F' = Full attention.
        # None keeps the uniform behaviour from ``layer.attention``.
        # SWA layers are automatically assigned attn_backend="flex".

        def _expand_layers(self) -> None:
            """Expand the ``layer`` template into upstream's per-layer config list.

            Upstream builds each block from a fully-specified entry in
            ``Decoder.Config.layers`` via a bare ``build()``, so everything the
            old code passed as a build kwarg or an ``init_weights`` argument --
            dim, layer_id, n_layers, the depth-init divisors, the RoPE cache and
            the inner-attention backend -- has to be stamped onto a per-layer
            copy here.
            """
            n_layers = self.n_layers
            base_attn = self.layer.attention
            assert isinstance(base_attn, GatedNormSWAttention.Config)

            use_rope = _parse_layer_pattern(
                self.rope_pattern, n_layers, "R", "N", default_true=base_attn.use_rope
            )
            use_swa = _parse_layer_pattern(
                self.swa_pattern,
                n_layers,
                "S",
                "F",
                default_true=base_attn.sliding_window_size > 0,
            )
            swa_window = base_attn.sliding_window_size
            depth_init = parse_depth_init(self.layer.depth_init)

            layers: list = []
            for layer_id in range(n_layers):
                residual_div_attn, residual_div_ffn = setup_depth_init(
                    depth_init, layer_id, n_layers
                )
                # SWA layers use the local RoPE cache when one is configured.
                layer_rope = (
                    self.rope_of_swa
                    if (use_swa[layer_id] and self.rope_of_swa is not None)
                    else self.rope
                )
                attn_backend = "flex" if use_swa[layer_id] else base_attn.attn_backend
                attn_cfg = _dc.replace(
                    base_attn,
                    dim=self.dim,
                    use_rope=use_rope[layer_id],
                    sliding_window_size=swa_window if use_swa[layer_id] else -1,
                    # SWA requires FlexAttention; non-SWA keeps the configured backend.
                    attn_backend=attn_backend,
                    rope=layer_rope if use_rope[layer_id] else None,
                    residual_div=residual_div_attn,
                )
                attn_cfg.inner_attention = attn_cfg.build_inner_attention()

                is_dense = layer_id < self.layer.n_dense_layers
                moe_cfg = None
                ff_cfg = None
                if is_dense:
                    assert self.layer.feed_forward is not None, (
                        f"layer {layer_id} is dense (n_dense_layers="
                        f"{self.layer.n_dense_layers}) but no feed_forward config is set"
                    )
                    ff_cfg = _dc.replace(
                        self.layer.feed_forward,
                        dim=self.dim,
                        residual_div=residual_div_ffn,
                        init_gate_as_residual=self.layer.init_gate_as_residual,
                    )
                else:
                    assert self.layer.moe is not None, (
                        f"layer {layer_id} is an MoE layer but no moe config is set"
                    )
                    moe_cfg = _dc.replace(
                        self.layer.moe,
                        dim=self.dim,
                        layer_id=layer_id,
                        residual_div=residual_div_ffn,
                        init_gate_as_residual=self.layer.init_gate_as_residual,
                    )

                layers.append(
                    _dc.replace(
                        self.layer,
                        attention=attn_cfg,
                        moe=moe_cfg,
                        feed_forward=ff_cfg,
                        attention_norm=build_norm_config(
                            self.layer.norm_type, self.dim, self.layer.norm_eps
                        ),
                        ffn_norm=build_norm_config(
                            self.layer.norm_type, self.dim, self.layer.norm_eps
                        ),
                        dim=self.dim,
                        layer_id=layer_id,
                        n_layers=n_layers,
                    )
                )
            self.layers = layers
            self.norm = build_norm_config(self.norm_type, self.dim, self.norm_eps)

            # Decoder.__init__ builds these directly; OPT MoE overrides `norm`
            # with its own build_norm, but tok_embeddings/lm_head are stock.
            self.tok_embeddings = Embedding.Config(
                num_embeddings=self.vocab_size,
                embedding_dim=self.dim,
                param_init={
                    "weight": make_param_init(
                        self.first_in_init_fn_type, self.first_in_init_std
                    )
                },
            )
            self.lm_head = Linear.Config(
                in_features=self.dim,
                out_features=self.vocab_size,
                param_init={
                    "weight": make_param_init(
                        self.final_out_init_fn_type, self.final_out_init_std
                    )
                },
            )

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            max_context_length = config.training.max_context_length
            if max_context_length > self.rope.max_context_length:
                logger.warning(
                    f"Context length {max_context_length} exceeds original "
                    f"maximum {self.rope.max_context_length}."
                )

            # Sync rope length (both global and SWA-local caches)
            self.rope = _dc.replace(self.rope, max_context_length=max_context_length)
            if self.rope_of_swa is not None:
                self.rope_of_swa = _dc.replace(
                    self.rope_of_swa, max_context_length=max_context_length
                )

            self.enable_token_mask_for_moe = getattr(
                config.training, "enable_token_mask_for_moe", False
            )

            base_attn = self.layer.attention
            assert isinstance(base_attn, GatedNormSWAttention.Config)
            use_swa = _parse_layer_pattern(self.swa_pattern, self.n_layers, "S", "F")
            uses_swa = any(use_swa) or (
                self.swa_pattern is None and base_attn.sliding_window_size > 0
            )
            if uses_swa and base_attn.sliding_window_size <= 0:
                raise ValueError(
                    "SWA is enabled but layer.attention.sliding_window_size "
                    "is not set (must be > 0)."
                )
            if uses_swa and base_attn.attn_backend == "varlen":
                raise ValueError("SWA is not supported with varlen attention.")

            # Expand before delegating: Decoder.Config.update_from_config walks
            # self.layers for TP/EP validation and token-dispatcher setup, so the
            # per-layer list has to exist by then.
            self._expand_layers()
            Decoder.Config.update_from_config(self, config=config, **kwargs)

            from torchtitan.models.opt_moe.sharding import set_opt_moe_sharding_config

            # enable_sequence_parallel defaults to True regardless of TP, so
            # gate on the actual degree -- otherwise a plain FSDP run would look
            # like a TP run to the sharding plan.
            set_opt_moe_sharding_config(
                self,
                enable_sp=(
                    parallelism.enable_sequence_parallel
                    and parallelism.tensor_parallel_degree > 1
                ),
                enable_tp=parallelism.tensor_parallel_degree > 1,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams, active_nparams = get_nparams_and_active_nparams(model)
            base_attn = self.layer.attention
            assert isinstance(base_attn, GatedNormSWAttention.Config)
            head_dim = (
                base_attn.head_dim
                if base_attn.head_dim is not None
                else self.dim // base_attn.n_heads
            )
            attention_op_flops = self.n_layers * quadratic_attention_flops_per_token(
                num_heads=base_attn.n_heads,
                qk_head_dim=head_dim,
                v_head_dim=head_dim,
                seq_len=seq_len,
            )
            return nparams, 6 * active_nparams + attention_op_flops

    def __init__(self, config: Config):
        super().__init__(config)
        if config.use_embeddings_norm:
            self.embeddings_norm = build_norm_config(
                config.norm_type, config.dim, config.norm_eps
            ).build()
        else:
            self.embeddings_norm = Identity.Config().build()

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        loss_mask: torch.Tensor | None = None,
        accumulated_load_balance_loss: torch.Tensor | None = None,
    ):
        """Forward pass.

        Args:
            tokens: Input token indices ``[T]``, or hidden states when this is
                not the first pipeline stage.
            positions: Position indices ``[T]``.
            attention_masks: Per-layer masks (dict) or a single mask.
            loss_mask: Token mask for the MoE load-balance loss.
            accumulated_load_balance_loss: Load-balance loss carried in from a
                prior pipeline stage.

        Returns:
            ``(output, total_lbl_loss)``.
        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        h = self.embeddings_norm(h)

        # Collect per-layer load-balance losses; accumulation happens after the
        # loop so we never thread a running tensor through every layer signature.
        local_lbl_loss: torch.Tensor | None = None
        for layer in self.layers.values():
            h, lbl = layer(h, attention_masks, positions, loss_mask)
            if lbl is not None:
                local_lbl_loss = lbl if local_lbl_loss is None else local_lbl_loss + lbl

        if accumulated_load_balance_loss is not None:
            total_lbl_loss = (
                accumulated_load_balance_loss + local_lbl_loss
                if local_lbl_loss is not None
                else accumulated_load_balance_loss
            )
        elif local_lbl_loss is not None:
            total_lbl_loss = local_lbl_loss
        else:
            total_lbl_loss = torch.zeros((), device=h.device, dtype=torch.float32)

        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h, total_lbl_loss
        output = self.lm_head(h) if self.lm_head is not None else h
        return output, total_lbl_loss

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> "AttentionMasksType | None":
        """Return attention masks appropriate for the mix of layer backends.

        Returns:
            ``None``                                  — all layers use SDPA.
            ``{"full": BlockMask}``                   — flex layers, no SWA.
            ``{"full": BlockMask, "swa": BlockMask}`` — mixed flex full + SWA.
            Delegates to ``super()`` for varlen.
        """
        has_flex = any(
            getattr(layer, "attn_backend", "sdpa") == "flex"
            for layer in self.layers.values()
        )
        has_swa = any(
            getattr(layer, "use_swa", False) for layer in self.layers.values()
        )
        has_varlen = any(
            getattr(layer, "attn_backend", "sdpa") == "varlen"
            for layer in self.layers.values()
        )

        if has_varlen and has_swa:
            raise ValueError("SWA is not supported with varlen attention.")
        if has_varlen:
            return super().get_attention_masks(positions)
        if not has_flex:
            # All SDPA — PyTorch handles causal masking internally.
            return None

        # Document boundaries now come from `positions` resetting to 0 rather
        # than from scanning for eos_id, so the mask no longer needs the
        # tokenizer or the raw token ids.
        attn_config = self.config.first_attention
        assert attn_config is not None
        base_attn = self.config.layer.attention
        assert isinstance(base_attn, GatedNormSWAttention.Config)
        mask_mods = [get_causal_mask_mod()]
        if base_attn.attn_mask_type == "block_causal":
            mask_mods.append(
                get_efficient_causal_mask_mod_for_packed_document(positions)
            )
        elif base_attn.attn_mask_type != "causal":
            raise ValueError(
                f"Unknown attn_mask_type: {base_attn.attn_mask_type!r}"
            )
        full_mask = self._create_flex_attention_mask(positions, attn_config, mask_mods)
        if not has_swa:
            return {"full": full_mask}

        swa_mask = self._create_flex_attention_mask(
            positions,
            attn_config,
            mask_mods + [get_sliding_window_mask_mod(base_attn.sliding_window_size)],
        )
        return {"full": full_mask, "swa": swa_mask}

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims,
        parallelism,
    ):
        """Build masks/CP shards, then add the MoE token mask.

        The MoE loss mask is derived from the CP-sharded labels so its token
        layout matches the hidden states the router sees; building it before CP
        would leave it full-length and break indexing inside the router.
        """
        inputs, labels, extra_kwargs = super().preprocess_inputs(
            input_dict, parallel_dims=parallel_dims, parallelism=parallelism
        )
        if self.config.enable_token_mask_for_moe:
            from torchtitan.components.loss import IGNORE_INDEX

            extra_kwargs["loss_mask"] = labels != IGNORE_INDEX
        return inputs, labels, extra_kwargs
