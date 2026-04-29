# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.common.moe.utils import (
    indices_padding_wrapper,
    need_indices_padding,
)
from torchtitan.ops.scatter_add import deterministic_scatter_add
from torchtitan.protocols.module import Module

from torchtitan.tools.logging import logger

from .norm_ffn import FeedForward

from .utils.activations import build_activation
from .utils.inits import build_init_fn
from .utils.moe_utils import calc_gate_scaling_factor
from .utils.norms import build_norm


def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    activation: Callable,
    mid_norm: nn.Module,
) -> torch.Tensor:
    # NOTE: this would incur a synchronization between device and host
    num_tokens_per_expert = num_tokens_per_expert.tolist()

    # side-effect code due to the usage of generate_permute_indices
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    # a tuple of tensors indexed by experts
    # each with shape (tokens_per_expert(varying), dim)
    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = activation(torch.matmul(x_expert, w1[expert_idx].transpose(-2, -1)))
        h = h * torch.matmul(x_expert, w3[expert_idx].transpose(-2, -1))
        h = mid_norm(h)
        h = torch.matmul(h, w2[expert_idx].transpose(-2, -1))
        # h shape (tokens_per_expert(varying), dim)
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)

    # side-effect code due to the usage of generate_permute_indices
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))

    return out


def _run_experts_grouped_mm(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    activation: Callable,
    mid_norm: nn.Module,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

    h = activation(
        torch._grouped_mm(x.bfloat16(), w1.bfloat16().transpose(-2, -1), offs=offsets)
    )
    h = h * torch._grouped_mm(
        x.bfloat16(), w3.bfloat16().transpose(-2, -1), offs=offsets
    )
    out = torch._grouped_mm(
        mid_norm(h), w2.bfloat16().transpose(-2, -1), offs=offsets
    ).type_as(x)

    return out


class GroupedExperts(nn.Module):
    def __init__(
        self,
        *,
        layer_id: int,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        use_grouped_mm: bool = True,
        activation_type: str = "silu",
        norm_everywhere: bool = False,
        norm_type: str | None = "np_rmsnorm",
        norm_eps: float | None = 1e-30,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.use_grouped_mm = use_grouped_mm

        self.act_fn = build_activation(activation_type)

        if norm_everywhere:
            assert (
                norm_type is not None
            ), "`norm_type` needs to be passed when `norm_everywhere=True`"
            assert (
                norm_eps is not None
            ), "`norm_eps` needs to be passed when `norm_everywhere=True`"
            self.mid_norm = build_norm(norm_type, dim=hidden_dim, eps=norm_eps)
        else:
            self.mid_norm = nn.Identity()

    def __repr__(self):
        model_str = f"GroupedExperts(dim={self.dim}, hidden_dim={self.hidden_dim},\n"
        # model_str += (
        #     f"\tnum_experts={self.num_experts}, local_experts={self.expert_per_rank}, "
        # )
        # model_str += f"ep_size={self.ep_size}, \n"
        model_str += f"\tup_proj={self.w1.shape}, \n"
        model_str += f"\tgate_proj={self.w3.shape}, \n"
        model_str += f"\tdown_proj={self.w2.shape}, \n"
        model_str += f"\tmid_norm={self.mid_norm}, \n"
        model_str += ")"
        return model_str

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.w1, DTensor):
            # Convert parameters from DTensors to plain Tensors, to work with
            # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
            w1 = self.w1.to_local()
            w2 = self.w2.to_local()
            w3 = self.w3.to_local()
        else:
            w1 = self.w1
            w2 = self.w2
            w3 = self.w3

        if self.use_grouped_mm:
            # NOTE: If EP is not used, we need to pad the indices
            #       to prepare for grouped_mm;
            #       otherwise, EP will handle the padding.
            if need_indices_padding() and (
                not isinstance(self.w1, DTensor)
                or "ep" not in self.w1.device_mesh.mesh_dim_names
            ):
                run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
            else:
                run_experts_fn = _run_experts_grouped_mm
            return run_experts_fn(
                w1,
                w2,
                w3,
                x,
                num_tokens_per_expert,
                activation=self.act_fn,
                mid_norm=self.mid_norm,
            )
        else:
            return _run_experts_for_loop(
                w1,
                w2,
                w3,
                x,
                num_tokens_per_expert,
                activation=self.act_fn,
                mid_norm=self.mid_norm,
            )

    def init_weights(
        self,
        residual_div: float,
        init_gate_as_residual: bool,
        weights_init_stds: tuple[float, float, float],
        init_fn_types: tuple[str, str, str],
        skip_init: bool = False,
    ):
        if not isinstance(self.mid_norm, nn.Identity):
            self.mid_norm.reset_parameters()
        if skip_init:
            return

        w1_init_std, w2_init_std, w3_init_std = weights_init_stds
        w1_init_fn_type, w2_init_fn_type, w3_init_fn_type = init_fn_types

        w1_init_fn = build_init_fn(w1_init_fn_type)
        w2_init_fn = build_init_fn(w2_init_fn_type)
        w3_init_fn = build_init_fn(w3_init_fn_type)

        w3_init_std = (
            w3_init_std / residual_div if init_gate_as_residual else w3_init_std
        )

        # lets always use different experts
        expert_init_fn = init_all_experts_different

        expert_init_fn(
            w1_init_fn, self.w1.data, w1_init_std, slot=0, layer_id=self.layer_id
        )
        expert_init_fn(
            w3_init_fn, self.w3.data, w3_init_std, slot=2, layer_id=self.layer_id
        )
        expert_init_fn(
            w2_init_fn,
            self.w2.data,
            w2_init_std / residual_div,
            slot=1,
            layer_id=self.layer_id,
        )


def make_seed_from_global(
    layer_id: int, slot: int, expert_id: int, total_experts: int
) -> int:
    # 3 slots per expert: w1, w2, w3
    return layer_id * (3 * total_experts) + slot * total_experts + expert_id


def init_all_experts_different(init_fn, w, init_std, slot, layer_id):
    # we should expect the experts to have same norms, rather than same weights
    assert layer_id is not None, "layer_id must be set "
    total_experts = w.shape[0]
    if isinstance(w, torch.distributed.tensor.DTensor):
        # we assume the DTensor is already sharded on dim 0
        local_tensor = w.to_local()
        shard_chunk = w.__create_chunk_list__()[0]
        offsets = shard_chunk.offsets[0]
    else:
        local_tensor = w
        offsets = 0

    for e in range(local_tensor.shape[0]):
        expert_id = e + offsets
        seed = make_seed_from_global(layer_id, slot, expert_id, total_experts)
        if local_tensor.device.type == "meta":
            rng = None
        else:
            rng = torch.Generator(device=local_tensor.device)
            rng.manual_seed(seed)

        init_fn(local_tensor[e], mean=0.0, std=init_std, generator=rng)

    if isinstance(w, torch.distributed.tensor.DTensor):
        w.to_local().copy_(local_tensor)
    else:
        w.copy_(local_tensor)


class TokenChoiceTopKRouter(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        route_scale: float,
        _debug_force_load_balance: bool = False,
        force_router_fp32_matmul: bool = False,
    ):
        super().__init__()

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self._debug_force_load_balance = _debug_force_load_balance
        self.force_router_fp32_matmul = force_router_fp32_matmul

    def __repr__(self):
        return (
            f"Gate(experts={self.num_experts}, topk={self.top_k} | "
            f"DEBUG_FORCE_LOAD_BALANCED: {self._debug_force_load_balance}) | "
            f"FORCE_ROUTER_FP32_MATMUL: {self.force_router_fp32_matmul}"
        )

    def init_weights(self, init_std: float, init_fn_type: str, skip_init: bool = False):
        if skip_init:
            return

        # nn.init.xavier_uniform_(self.expert_embeddings)
        init_fn = build_init_fn(init_fn_type)
        init_fn(self.gate.weight, mean=0.0, std=init_std)

    def _debug_force_load_balance_routing(
        self, scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Balanced round-robin expert assignment.
        Returns (selected_experts_indices [N, K] LongTensor, top_scores [N, K] FloatTensor).
        """
        n_tokens = scores.size(0)
        # Round-robin indices with exact balance
        selected_experts_indices = (
            torch.arange(
                n_tokens * self.top_k, device=scores.device, dtype=torch.int64
            ).reshape(n_tokens, self.top_k)
            % self.num_experts
        )
        top_scores = scores.gather(dim=1, index=selected_experts_indices)  # [N,K]
        return selected_experts_indices, top_scores

    def forward(
        self,
        x: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
        need_aux_loss: bool = False,
        loss_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        # scores shape (bs*slen, num_experts)
        if self.force_router_fp32_matmul:
            with torch.autocast(device_type=x.device.type, dtype=torch.float32):
                scores = self.gate(x)
        else:
            scores = self.gate(x).float()

        scores = torch.sigmoid(scores)

        scores_for_choice = scores if expert_bias is None else scores + expert_bias
        _, selected_experts_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=1, index=selected_experts_indices)

        if need_aux_loss:
            indices_for_load_balance = torch.topk(scores, k=self.top_k, dim=1)[1]
        else:
            indices_for_load_balance = None

        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        top_scores = top_scores / (top_scores.sum(dim=-1, keepdim=True) + 1e-20)

        # TODO(JSC):  entropy - Do we want the entropy on the top-K experts or all experts?

        # ========================================
        # BELOW IS OLD IMPLEMENTATION OF ENTROPY CALCULATION
        detached_top_scores = top_scores.detach()
        experts_entropy = (
            -(detached_top_scores * detached_top_scores.log()).sum(dim=-1).mean()
        )

        if loss_mask is None:
            num_tokens_per_expert = torch.histc(
                selected_experts_indices.view(-1),
                bins=self.num_experts,
                min=0,
                max=self.num_experts,
            )
        else:
            mask_t = loss_mask.view(-1)  # (T,)
            mask_tk = mask_t[:, None].expand(-1, self.top_k)  # (T, K)

            idx = selected_experts_indices.reshape(-1)  # (T*K,)
            m = mask_tk.reshape(-1).bool()  # (T*K,)

            # masked token counts
            num_tokens_per_expert = torch.bincount(
                idx[m], minlength=self.num_experts
            ).to(device=idx.device)

        # ===== END
        top_scores = top_scores * self.route_scale

        return (
            top_scores,
            scores,
            selected_experts_indices,
            num_tokens_per_expert,
            experts_entropy,
            indices_for_load_balance,
        )


# NOTE: the reason we make this a stateless module is to support
#       expert_tensor_parallel_degree=1 with consistent TP/EP APIs.
class TokenReorderer(nn.Module):
    """
    This module reorders token indices to match the order of experts, enabling
    efficient parallel processing of tokens by experts.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of experts each token will be routed to.
    """

    def __init__(self, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        top_scores: torch.Tensor,
        selected_experts_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reorders token indices to match the order of experts for MoE routing.

        Args:
            top_scores (torch.Tensor): Routing scores for selected experts,
                shape (batch_size * seq_len, top_k)
            selected_experts_indices (torch.Tensor): Expert indices selected for each token,
                shape (batch_size*seq_len, top_k)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores_experts_sorted: Scores reordered to match expert ordering
                - token_indices_experts_sorted: Token indices reordered to match expert ordering
                - num_tokens_per_expert: Number of tokens assigned to each expert
        """
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )

        # Reorder the token indices to match the order of the experts
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        token_indices_experts_sorted = torch.argsort(
            selected_experts_indices.view(-1), stable=True
        )

        top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
        token_indices_experts_sorted = token_indices_experts_sorted // self.top_k

        return (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        )


def saint_check_scaling_factor(
    num_routed_experts: int,
    activate_experts: int,
    scaling_factor: float | None = None,
    iter_times: int = 10_000,
) -> float:
    if scaling_factor is not None:
        return scaling_factor

    return calc_gate_scaling_factor(
        num_routed_experts,
        activate_experts,
        iter_times=iter_times,
    )


class MoE(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_experts: int = 8
        num_shared_experts: int = 1
        top_k: int = 1

        scaling_factor: float | None = None
        score_before_experts: bool = False

        use_grouped_mm: bool = True
        load_balance_coeff: float | None = 1e-3
        load_balance_loss_weight: float = 0.0
        load_balance_loss_type: Literal["sequence_wise", "batch_wise"] = "sequence_wise"
        bias_update_norm_factor: str = "sign"

        _debug_force_load_balance: bool = False
        # if True, we force each experts get same amount of token via round-robin

        # Expert hidden dimension (replaces old moe_inter_dim)
        hidden_dim: int = 0

        force_router_on_fp32: bool = False

        norm_everywhere: bool = False
        norm_type: str | None = "np_rmsnorm"
        norm_eps: float | None = 1e-30
        activation_type: str = "silu"

        # init
        w1_init_fn_type: str = "scaled_orthogonal"
        w2_init_fn_type: str = "scaled_orthogonal"
        w3_init_fn_type: str = "scaled_orthogonal"
        w1_init_std: float = 1.0
        w2_init_std: float = 1.0
        w3_init_std: float = 1.0

        router_init_fn_type: str = "scion_normal_output"
        router_init_std: float = 1.0

    def __init__(self, config: Config, *, layer_id: int, dim: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.num_experts = config.num_experts
        self.top_k = config.top_k

        self.load_balance_loss_weight = (
            config.load_balance_loss_weight
        )  # Loss coefficient
        self.load_balance_coeff = config.load_balance_coeff
        self.bias_update_norm_factor = config.bias_update_norm_factor
        self.load_balance_loss_type = config.load_balance_loss_type

        self.score_before_experts = config.score_before_experts
        if self.score_before_experts is True:
            logger.warning("score_before_experts is True, Let's set it to False")
            raise ValueError("score_before_experts must be False")

        self.scaling_factor = saint_check_scaling_factor(
            num_routed_experts=self.num_experts,
            activate_experts=self.top_k,
            scaling_factor=config.scaling_factor,
        )
        logger.info(
            f"Checked router_scaling_factor: {self.scaling_factor} at layer {self.layer_id}"
        )

        hidden_dim = config.hidden_dim
        self.experts = GroupedExperts(
            layer_id=layer_id,
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=self.num_experts,
            use_grouped_mm=config.use_grouped_mm,
            activation_type=config.activation_type,
            norm_everywhere=config.norm_everywhere,
            norm_type=config.norm_type,
            norm_eps=config.norm_eps,
        )
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            route_scale=self.scaling_factor,
            _debug_force_load_balance=config._debug_force_load_balance,
            force_router_fp32_matmul=config.force_router_on_fp32,
        )
        self.reorderer = TokenReorderer(
            num_experts=self.num_experts, top_k=config.top_k
        )
        self.shared_experts = (
            FeedForward.Config(
                hidden_dim=hidden_dim * config.num_shared_experts,
                activation_type=config.activation_type,
                norm_everywhere=config.norm_everywhere,
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
                w1_init_fn_type=config.w1_init_fn_type,
                w2_init_fn_type=config.w2_init_fn_type,
                w3_init_fn_type=config.w3_init_fn_type,
                w1_init_std=config.w1_init_std,
                w2_init_std=config.w2_init_std,
                w3_init_std=config.w3_init_std,
            ).build(dim=dim)
            if config.num_shared_experts > 0
            else None
        )

        self.register_buffer(
            "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32)
        )
        self.register_buffer("load_balance_loss", torch.zeros(1, dtype=torch.float32))
        self.register_buffer(
            "tokens_per_expert", torch.zeros(self.num_experts, dtype=torch.int64)
        )
        self.register_buffer("router_entropy", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("acc_fwd_times", torch.zeros(1, dtype=torch.int64))
        # EMA state used by training-path moe_maxvio_ema metric.
        self.register_buffer(
            "tokens_per_expert_cumul",
            torch.zeros(self.num_experts, dtype=torch.float32),
        )

    def forward(
        self, x: torch.Tensor, loss_mask: torch.Tensor | None = None, **kwargs
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """
        Forward pass for the MoE layer.

        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.
            loss_mask (torch.Tensor | None): Loss mask tensor with shape ``(bs, slen)``.

        Returns:
            (out, load_balance_loss):
                ``out`` has shape ``(bs, slen, dim)`` and ``load_balance_loss`` is
                a scalar tensor when enabled, otherwise ``None``.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # top_scores and selected_experts_indices shape (bs*slen, top_k)
        # num_tokens_per_expert shape (num_experts,)
        need_lb_loss = self.training and self.load_balance_loss_weight > 0.0
        (
            top_scores,
            sigmoid_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            experts_entropy,
            indices_for_load_balance,
        ) = self.router(
            x,
            self.expert_bias,
            need_aux_loss=need_lb_loss,
            loss_mask=loss_mask,
        )
        # tokens_per_expert will be used to update the expert bias for load balancing.
        # and also to count the expert usage
        # TODO: Activation Checkpointing has the side effect of double counting tokens_per_expert --
        #       first in the forward pass, and then in the backward pass. However, this has no
        #       effect on the expert bias update thanks to the torch.sign() operator.
        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
            self.router_entropy.add_(experts_entropy)
            self.acc_fwd_times.add_(1)

        # top_scores_experts_sorted and token_indices_experts_sorted shape (bs*slen*top_k,)
        # num_tokens_per_expert shape (num_experts,)
        # NOTE: the reason we need to compute num_tokens_per_expert again is:
        #       1st computation in router is to update self.tokens_per_expert
        #       which would be the same across all TP ranks.
        #       2nd computation in reorderer is for the actual routing and experts computation
        #       which would be sharded over TP ranks if expert_tensor_parallel_degree==1.
        #       If tensor_paralllel_degree == expert_tensor_parallel_degree, they agree.
        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        if need_lb_loss:
            if self.load_balance_loss_type == "sequence_wise":
                load_balance_loss = MoE.sequence_wise_aux_loss(
                    sigmoid_scores,
                    indices_for_load_balance,
                    bs,
                    slen,
                    self.top_k,
                    self.load_balance_loss_weight,
                )
            elif self.load_balance_loss_type == "batch_wise":
                load_balance_loss = MoE.batch_wise_aux_loss(
                    sigmoid_scores,
                    num_tokens_per_expert,
                    self.top_k,
                    self.load_balance_loss_weight,
                )
            else:
                raise ValueError(
                    f"Invalid load_balance_loss_type: {self.load_balance_loss_type}"
                )
            with torch.no_grad():
                # for logging only
                self.load_balance_loss.add_(load_balance_loss.detach())
        else:
            load_balance_loss = None

        # ====

        # shape (bs*slen*top_k, dim)
        routed_input = x[token_indices_experts_sorted]

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # shared expert
        # Note: we execute the shared expert before scoring the output of the routed expert
        # to "implicitly" overlap the shared expert compute with token combine communication
        out = (
            self.shared_experts(x)
            if self.shared_experts is not None
            else torch.zeros_like(x)
        )

        if not self.score_before_experts:
            routed_output = (
                routed_output.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        out = deterministic_scatter_add(
            out,
            token_indices_experts_sorted.reshape(-1, 1).expand(-1, dim),
            routed_output,
        )

        out = out.reshape(bs, slen, dim)
        return out, load_balance_loss

    def init_weights(
        self,
        residual_div: float,
        init_gate_as_residual: bool,
        skip_init: bool = False,
    ):
        self.experts.init_weights(
            residual_div=residual_div,
            init_gate_as_residual=init_gate_as_residual,
            weights_init_stds=(
                self.config.w1_init_std,
                self.config.w2_init_std,
                self.config.w3_init_std,
            ),
            init_fn_types=(
                self.config.w1_init_fn_type,
                self.config.w2_init_fn_type,
                self.config.w3_init_fn_type,
            ),
            skip_init=skip_init,
        )
        if self.shared_experts is not None:
            self.shared_experts.init_weights(
                residual_div=residual_div,
                init_gate_as_residual=init_gate_as_residual,
                skip_init=skip_init,
            )
        self.router.init_weights(
            self.config.router_init_std,
            self.config.router_init_fn_type,
            skip_init=skip_init,
        )

        self.expert_bias.zero_()
        self.tokens_per_expert.zero_()
        self.tokens_per_expert_cumul.zero_()
        self.router_entropy.zero_()
        self.acc_fwd_times.zero_()
        self.load_balance_loss.zero_()

    @staticmethod
    def sequence_wise_aux_loss(
        scores: torch.Tensor,  # Shape: (B*S, N) - Raw Sigmoid Affinities (s_{i,t})
        indices: torch.Tensor | None,  # Shape: (B*S, K) - Selected Expert Indices
        B: int,  # Batch size
        S: int,  # Sequence length (T in the paper)
        top_k: int,  # K_r
        aux_loss_alpha: float,  # Alpha
    ) -> torch.Tensor:
        """
        Computes Sequence-Wise Auxiliary Loss (DeepSeek-V3 Equations 17-20).

        Args:
            scores: The dense affinity scores (s_{i,t}) for routed experts.
                    Should be the output of Sigmoid, shape (B*S, N).
            indices: The top-k selected expert indices. Shape (B*S, K).
        """
        if aux_loss_alpha <= 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # N_r: Total number of routed experts
        N = scores.size(-1)

        indices = indices.long()
        # 1. Reshape inputs to handle each sequence separately: (B, S, N)
        #    This ensures we calculate P_i and f_i per sequence (Eq 20 & 18).
        scores_per_seq = scores.view(B, S, N)
        indices_per_seq = indices.view(B, S, top_k)

        # 2. Eq 19: Normalize affinity scores s_{i,t} to get s'_{i,t}
        #    DeepSeek-V3 uses Sigmoid, so scores don't sum to 1.
        #    Eq 19 explicitly requires dividing by the sum of all affinities.
        #    denominator shape: (B, S, 1)
        denominator = scores_per_seq.sum(dim=-1, keepdim=True) + 1e-20
        probs_per_seq = scores_per_seq / denominator  # This is s'_{i,t}

        # 3. Eq 20: Calculate P_i (Average probability per expert for each sequence)
        #    P_i = (1/T) * sum_{t=1}^T (s'_{i,t})
        #    We average over the Sequence dimension (dim=1).
        #    P_i shape: (B, N)
        P_i = probs_per_seq.mean(dim=1)

        # 4. Eq 18: Calculate f_i (Fraction of tokens selecting expert i per sequence)
        #    f_i = (N / (K * T)) * count_i

        # Flatten the top-k dimension to count hits per sequence: (B, S*K)
        flat_indices_per_seq = indices_per_seq.view(B, -1)
        selection_counts = torch.zeros((B, N), device=scores.device, dtype=scores.dtype)
        src = torch.ones_like(flat_indices_per_seq, dtype=scores.dtype)
        selection_counts.scatter_add_(1, flat_indices_per_seq, src)

        # Calculate f_i for each sequence, T (tokens in sequence) is S
        f_i = selection_counts * (N / (top_k * S))

        # 5. Eq 17: Calculate Balance Loss
        loss_per_seq = (f_i * P_i).sum(dim=1) * aux_loss_alpha

        return loss_per_seq.mean()

    @staticmethod
    def batch_wise_aux_loss(
        scores: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        top_k: int,
        aux_loss_alpha: float,
    ) -> torch.Tensor:
        """
        Computes Batch-Wise Auxiliary Loss.
        Args:
            scores: Dense probabilities (BS, N).
            num_tokens_per_expert: Token counts (N).
            top_k: Number of experts selected per token.
            aux_loss_alpha: Scaling factor for the loss.
        """
        if aux_loss_alpha <= 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # Total number of routed experts (N)
        N = scores.size(1)
        # Total number of tokens (T = BS * S)
        T = scores.size(0)

        P_i = scores.mean(dim=0)

        f_i = num_tokens_per_expert.to(scores.dtype) * (N / (top_k * T))

        loss = (f_i * P_i).sum() * aux_loss_alpha

        return loss
