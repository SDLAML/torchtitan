# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE with DeepEP backend for efficient expert-parallel communication."""

from dataclasses import dataclass

import torch

from torchtitan.distributed.deepep import sync_combine

from .norm_moe import MoE


class DeepEPMoE(MoE):
    """
    Mixture of Experts with DeepEP communication.

    Inherits from MoE but overrides forward() to pass routing info to experts,
    letting DeepEPExpertParallel hooks handle dispatch/combine.

    The forward pass is structured to overlap shared_experts computation with
    the DeepEP combine communication:
    1. Router computes expert assignments
    2. DeepEP dispatches tokens to experts (sync)
    3. Experts process tokens
    4. DeepEP combine starts (async) - returns immediately
    5. shared_experts runs IN PARALLEL with combine communication
    6. sync_combine() waits for combine to complete
    7. Addition of shared_experts output and routed_output
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MoE.Config):
        pass

    def __init__(self, config: Config, *, layer_id: int, dim: int):
        super().__init__(config, layer_id=layer_id, dim=dim)
        # DeepEP doesn't use reorderer - routing handled by DeepEPExpertParallel
        self.reorderer = None  # pyrefly: ignore [bad-assignment]

    def forward(
        self, x: torch.Tensor, loss_mask: torch.Tensor | None = None, **kwargs
    ) -> "tuple[torch.Tensor, torch.Tensor | None]":
        """
        Forward pass with DeepEP communication.

        DeepEPExpertParallel hooks intercept experts() call and handle
        dispatch/combine via deepep functions.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

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

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
            self.router_entropy.add_(experts_entropy)
            self.acc_fwd_times.add_(1)

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

        # Call experts with routing info - hooks handle DeepEP dispatch/combine
        routed_output = self.experts(
            x,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
        )

        out = self.shared_experts(x) if self.shared_experts is not None else None

        sync_combine()
        if out is None:
            return routed_output.reshape(bs, slen, dim), load_balance_loss
        return (out + routed_output).reshape(bs, slen, dim), load_balance_loss
