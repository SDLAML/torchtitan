# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, TypeAlias

import torch

from torchtitan.config import JobConfig
from torchtitan.models.inputs import MoEInputsDict
from torchtitan.tools.logging import logger

# PyTorch's default ignore index for cross-entropy loss
IGNORE_INDEX = -100

LossFunction: TypeAlias = Callable[..., torch.Tensor]

IGNORE_INDEX = -100
# Pytorch's default for F.cross_entropy
# Used in VLM and SFT training


def cross_entropy_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss with sum reduction for token-based normalization."""
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(),
        labels.flatten(0, 1),
        reduction="sum",
        ignore_index=IGNORE_INDEX,
    )


def build_cross_entropy_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn


def moe_loss(
    pred: MoEInputsDict,
    labels: torch.Tensor,
    loss_fn: LossFunction,
    grad_accumulation_steps: int = 1,
) -> torch.Tensor:
    """Sequence-wise auxiliary loss-enhanced loss function for MoE Transformer
    model training.
    """
    if isinstance(pred, dict) and "load_balance_loss" in pred:
        loss = loss_fn(pred["tokens_list"][0], labels)
        aux_loss = pred["load_balance_loss"] / grad_accumulation_steps
        # USE STE to make the magnitude of loss remain the same
        loss = loss + (aux_loss - aux_loss.detach())
    elif isinstance(pred, tuple):
        pred, aux_loss = pred
        loss = loss_fn(pred, labels)
        aux_loss = aux_loss / grad_accumulation_steps
        loss = loss + (aux_loss - aux_loss.detach())
    else:
        loss = loss_fn(pred, labels)
    return loss


def mse_loss(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Common MSE loss function with sum reduction for Transformer models training."""
    return torch.nn.functional.mse_loss(
        pred.float(), labels.float().detach(), reduction="sum"
    )


def build_mse_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    loss_fn = mse_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn
