# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE with DeepEP backend for efficient expert-parallel communication."""

from typing import Optional

import torch

from .moe import MoE, MoEArgs


class DeepEPMoE(MoE):
    """
    Mixture of Experts with DeepEP communication.

    Inherits from MoE but overrides forward() to pass routing info to experts,
    letting DeepEPExpertParallel hooks handle dispatch/combine.
    """

    def __init__(
        self,
        layer_id: int,
        dim: int,
        hidden_dim,
        moe_args: MoEArgs,
        activation_type: str = "silu",
        norm_everywhere: bool = False,
        norm_type: Optional[str] = None,
        norm_eps: Optional[float] = None,
    ):
        super().__init__(
            layer_id,
            dim,
            hidden_dim,
            moe_args,
            activation_type,
            norm_everywhere,
            norm_type,
            norm_eps,
        )
        # DeepEP doesn't use reorderer - routing handled by DeepEPExpertParallel
        self.reorderer = None  # pyrefly: ignore [bad-assignment]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with DeepEP communication.

        DeepEPExpertParallel hooks intercept experts() call and handle
        dispatch/combine via deepep functions.
        """
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        (
            top_scores,
            sigmoid_scores,
            selected_experts_indices,
            num_tokens_per_expert,
            experts_entropy,
            indices_for_load_balance,
        ) = self.router(
            x, self.expert_bias, need_aux_loss=self.load_balance_coeff > 0.0
        )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)
            self.router_entropy.add_(experts_entropy)
            self.acc_fwd_times.add_(1)

        if self.training:
            if self.load_balance_loss_type == "sequence_wise":
                load_balance_loss = MoE.sequence_wise_aux_loss(
                    sigmoid_scores,
                    indices_for_load_balance.long(),
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
            load_balance_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # Call experts with routing info - hooks handle DeepEP dispatch/combine
        routed_output = self.experts(
            x,
            num_tokens_per_expert,
            selected_experts_indices,
            top_scores,
            self.experts.num_experts,
        )

        out = self.shared_experts(x) if self.shared_experts is not None else None

        if out is None:
            return routed_output.reshape(bs, slen, dim), load_balance_loss
        return (out + routed_output).reshape(bs, slen, dim), load_balance_loss
