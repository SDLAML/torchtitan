# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn.attention.flex_attention import and_masks
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    create_attention_mask,
    create_varlen_metadata_for_document,
    get_causal_mask_mod,
    get_document_mask_mod,
)

from torchtitan.models.inits import (
    build_init_fn,
    setup_depth_init,
    setup_residual_scale,
)
from torchtitan.models.llama3.model.model import (
    # apply_rotary_emb,
    Attention,
    precompute_freqs_cis,
    # repeat_kv,
)
from torchtitan.models.norms import build_norm
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol
from torchtitan.tools.logging import logger

from .args import MoEModelArgs
from .moe import FeedForward, MoE
from .moe_deepep import DeepEPMoE
from .moe_utils import calc_gate_scaling_factor


class TransformerBlock(nn.Module):
    """
    TransformerBlock Module

    Args:
        layer_id (int): Identifier for the layer.
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, layer_id: int, model_args: MoEModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.attention = Attention(model_args)

        moe_args = model_args.moe_args
        # TODO(JSC): Need ablation, feels like this does not really matter
        if moe_args.scaling_factor is None:
            scaling_factor = calc_gate_scaling_factor(
                moe_args.num_routed_experts,
                moe_args.activate_experts,
                iter_times=10_000,
            )
            if layer_id == 0:
                logger.info(f"Auto-computed router_scaling_factor: {scaling_factor}")
        else:
            scaling_factor = moe_args.scaling_factor
            if layer_id == 0:
                logger.info(
                    f"Using manually set router_scaling_factor: {scaling_factor}"
                )
        moe_args.scaling_factor = scaling_factor

        self.moe_enabled = layer_id >= model_args.n_dense_layers

        if self.moe_enabled:
            match_dim_with_dense = True
            if match_dim_with_dense:
                ratio = 1.0 / (moe_args.top_k + moe_args.num_shared_experts)
            else:
                ratio = 1.0

            hidden_dim = 2 * 4 * model_args.dim / 3
            if model_args.ffn_dim_multiplier is not None:
                hidden_dim = model_args.ffn_dim_multiplier * hidden_dim
            hidden_dim = int(hidden_dim * ratio)
            hidden_dim = hidden_dim - hidden_dim % model_args.multiple_of

            if model_args.moe_intermediate_size is not None:
                hidden_dim = model_args.moe_intermediate_size

            if model_args.moe_impl == "deepep":
                moe_cls = DeepEPMoE
            else:
                moe_cls = MoE

            self.moe = moe_cls(
                layer_id,
                model_args.dim,
                hidden_dim,
                moe_args,
                activation_type=model_args.activation_type,
                norm_everywhere=model_args.norm_everywhere,
                norm_type=model_args.norm_type,
                norm_eps=model_args.norm_eps,
            )
        else:
            hidden_dim = 2 * 4 * model_args.dim / 3
            if model_args.ffn_dim_multiplier is not None:
                hidden_dim = model_args.ffn_dim_multiplier * hidden_dim
            hidden_dim = int(hidden_dim - hidden_dim % model_args.multiple_of)

            if model_args.intermediate_size is not None:
                hidden_dim = model_args.intermediate_size

            self.feed_forward = FeedForward(
                dim=model_args.dim,
                hidden_dim=hidden_dim,
                activation_type=model_args.activation_type,
                norm_everywhere=model_args.norm_everywhere,
                norm_type=model_args.norm_type,
                norm_eps=model_args.norm_eps,
            )

        self.attention_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )

        model_init_args = model_args.model_init_args
        self.weight_init_fn_type = model_init_args.intermediate_init_fn_type
        self.weight_init_std = (
            model_init_args.intermediate_init_std
            * model_args.dim**model_init_args.intermediate_exp
        )
        self.router_init_fn_type = model_init_args.router_init_fn_type
        self.init_gate_as_residual = model_init_args.init_gate_as_residual

        # x  = identity_scale * x + block_scale * block(x)

        self.residual_div_attn, self.residual_div_ffn = setup_depth_init(
            model_init_args.depth_init, layer_id, model_args.n_layers
        )
        self.block_scale, self.identity_scale = setup_residual_scale(
            model_init_args.residual_scale, model_args.n_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        accumulated_load_balance_loss: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """

        h = self.identity_scale * x + self.block_scale * self.attention(
            self.attention_norm(x), freqs_cis, attention_masks, positions
        )

        if self.moe_enabled:
            mlp_output, load_balance_loss = self.moe(self.ffn_norm(h), loss_mask)
            accumulated_load_balance_loss = (
                accumulated_load_balance_loss + load_balance_loss
            )

        else:
            mlp_output = self.feed_forward(self.ffn_norm(h))

        out = self.identity_scale * h + self.block_scale * mlp_output

        return out, accumulated_load_balance_loss

    def init_weights(self):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(
            self.weight_init_std,
            residual_div=self.residual_div_attn,
            init_fn_type=self.weight_init_fn_type,
        )
        if self.moe_enabled:
            self.moe.init_weights(
                self.weight_init_std,
                residual_div=self.residual_div_attn,
                init_gate_as_residual=self.init_gate_as_residual,
                init_fn_type=self.weight_init_fn_type,
                router_init_fn_type=self.router_init_fn_type,
            )
        else:
            self.feed_forward.init_weights(
                self.weight_init_std,
                residual_div=self.residual_div_ffn,
                init_gate_as_residual=self.init_gate_as_residual,
                init_fn_type=self.weight_init_fn_type,
            )


class Transformer(ModelProtocol):
    """
    Transformer Module

    Args:
        model_args (TransformerModelArgs): Model configuration arguments.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
        vocab_size (int): Vocabulary size.
        n_layers (int): Number of layers in the model.
        tok_embeddings (ParallelEmbedding): Token embeddings.
        layers (torch.nn.ModuleList): List of Transformer blocks.
        norm (RMSNorm): Layer normalization for the model output.
        output (Linear): Linear layer for final output.
        freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

    """

    transformer_block_cls = TransformerBlock

    def __init__(self, model_args: MoEModelArgs):
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.pad_id = model_args.pad_id

        self.tok_embeddings = nn.Embedding(
            model_args.vocab_size,
            model_args.dim,
            padding_idx=self.pad_id if self.pad_id >= 0 else None,
        )

        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = self.transformer_block_cls(
                layer_id, model_args
            )
        self.norm = build_norm(
            model_args.norm_type, dim=model_args.dim, eps=model_args.norm_eps
        )
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        self.init_weights()

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        model_init_args = self.model_args.model_init_args
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()
        first_in_init_fn = build_init_fn(model_init_args.first_in_init_fn_type)
        first_in_std = (
            model_init_args.first_in_init_std
            * self.model_args.vocab_size**model_init_args.first_in_exp
        )
        if self.tok_embeddings is not None:
            if model_init_args.first_in_init_fn_type == "scion_normal":
                # catch cases when axis=1 is sharded
                assert self.tok_embeddings.weight.size(1) == self.model_args.dim, (
                    f"Input embedding last dim does not match model dim. "
                    f"Got shape: {self.tok_embeddings.weight.shape}"
                )
            first_in_init_fn(
                self.tok_embeddings.weight,
                mean=0.0,
                std=first_in_std,
            )
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_init_fn = build_init_fn(model_init_args.final_out_init_fn_type)
        final_out_std = (
            model_init_args.final_out_init_std
            * self.model_args.dim**model_init_args.final_out_exp
        )
        cutoff_factor = 3
        if self.output is not None:
            extra_kwargs = {}
            if model_init_args.final_out_init_fn_type == "trunc_normal":
                extra_kwargs["a"] = -cutoff_factor * final_out_std
                extra_kwargs["b"] = cutoff_factor * final_out_std
            if model_init_args.final_out_init_fn_type == "scion_normal":
                # catch cases when axis=1 is sharded
                assert self.output.weight.size(1) == self.model_args.dim, (
                    f"Output last dim does not match model dim. "
                    f"Got shape: {self.output.weight.shape}"
                )
            final_out_init_fn(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                **extra_kwargs,
            )

    def _precompute_freqs_cis(self) -> torch.Tensor:
        dim_per_head = self.model_args.dim // self.model_args.n_heads
        if self.model_args.head_dim is not None:
            dim_per_head = self.model_args.head_dim
        return precompute_freqs_cis(
            dim_per_head,
            # Need to compute until at least the max token limit for generation
            # TODO: explain in docs/composability.md why we removed the 2x
            # relaxing in our CP enablement PR
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
            self.model_args.rope_scaling_args,
        )

    def _get_flex_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        mask_mods = [get_causal_mask_mod()]

        match self.model_args.attn_mask_type:
            case "causal":
                B = 1
            case "block_causal":
                B = input_batch.shape[0]
                assert tokenizer.eos_id is not None
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )

        return create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.model_args.attn_type:
            case "flex":
                return self._get_flex_attention_masks(
                    input_batch, tokenizer, extra_inputs
                )
            case "varlen":
                if self.model_args.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal \
                        attention mask type, got {self.model_args.attn_mask_type}"
                    )
                assert tokenizer.eos_id is not None
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise TypeError("Only varlen and flex attn masks are supported")

    def forward(
        self,
        tokens: torch.Tensor,
        accumulated_load_balance_loss: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
            accumulated_load_balance_loss (torch.Tensor | None): Accumulated load balance loss.
            attention_masks (AttentionMasksType | None): Attention masks.
            positions (torch.Tensor | None): Positions.
            loss_mask (torch.Tensor | None): Loss mask.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        accumulated_load_balance_loss = (
            torch.zeros((), device=h.device, dtype=torch.float32)
            if accumulated_load_balance_loss is None
            else accumulated_load_balance_loss
        )

        for layer in self.layers.values():
            h, accumulated_load_balance_loss = layer(
                h,
                self.freqs_cis,
                accumulated_load_balance_loss,
                attention_masks,
                positions,
                loss_mask,
            )

        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output, accumulated_load_balance_loss
