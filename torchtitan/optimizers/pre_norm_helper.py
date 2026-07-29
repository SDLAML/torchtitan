# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pre-norm: a step applied to the effective gradient (raw grad, or
momentum-blended buffer if momentum>0) before any communication for LMO --
so it runs on the raw tensor in its original dtype, before it gets cast down
to the communication dtype and gathered/reconstructed.

Config values look like "identity" (no-op, default), "row-l2", "col-l2",
"mat-l2": the prefix before the first "-" (row/col/mat) selects the
communication strategy, the full string selects the formula. Three
variants, by what a full 2-D weight/grad matrix's dim=0 (row, the FSDP-
sharded dimension) vs dim=-1 (col) reduction needs:

  - row: reduces along dim=-1 (never the FSDP-sharded dim) -- every local
    shard already holds complete rows, so this works with zero
    communication. Applied directly to whatever the effective gradient
    already is (DTensor row-shard or plain Tensor) at the point it's
    fetched -- see PRE_NORM_ROW_FUNCTIONS usage in disco.py. No pass, no
    unwrap needed: dim=-1 reduction dispatches locally per DTensor shard.

  - col: reduces along dim=-2 (the row/FSDP-sharded dimension) -- a local
    shard only sees some rows, so the true per-column norm needs combining
    across ranks: one all-reduce of a [..., cols]-shaped partial
    sum-of-squares.

  - mat: reduces over the last two dims entirely -- same idea, one
    all-reduce of a [...]-shaped (one scalar per matrix) partial
    sum-of-squares.

All functions are shape-agnostic over an optional leading batch dim (used
when disco.py stacks several same-shape params from one shape group into
one [N, rows, cols] tensor for a single batched op instead of N separate
ones) -- dim=-1/-2 addressing means they work identically on a plain
[rows, cols] matrix or a [N, rows, cols] stack.
"""

from typing import Callable

import torch


def _row_l2(g: torch.Tensor, eps: float) -> torch.Tensor:
    g32 = g.float()
    norm = g32.pow(2).sum(dim=-1, keepdim=True).sqrt()
    return (g32 / (norm + eps)).to(g.dtype)


# name -> (g, eps) -> normalized g. g may be a DTensor (row-sharded or not)
# or a plain Tensor -- applied directly wherever the effective grad is
# already fetched, no unwrap/pass/cache needed.
PRE_NORM_ROW_FUNCTIONS: dict[str, Callable[[torch.Tensor, float], torch.Tensor]] = {
    "row-l2": _row_l2,
}


def _col_l2_full(g: torch.Tensor, eps: float) -> torch.Tensor:
    g32 = g.float()
    norm = g32.pow(2).sum(dim=-2, keepdim=True).sqrt()
    return (g32 / (norm + eps)).to(g.dtype)


def _mat_l2_full(g: torch.Tensor, eps: float) -> torch.Tensor:
    g32 = g.float()
    norm = g32.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt()
    return (g32 / (norm + eps)).to(g.dtype)


# name -> (g_full, eps) -> normalized g_full. Used whenever the full matrix
# is already available with no extra work: ddp/experts (always), non-sharded
# embed params, and the gather_to_local=True norm-logging path (which
# already gathers to full for its own reasons).
PRE_NORM_FULL_FUNCTIONS: dict[str, Callable[[torch.Tensor, float], torch.Tensor]] = {
    "col-l2": _col_l2_full,
    "mat-l2": _mat_l2_full,
}


def _col_l2_partial(g_local: torch.Tensor) -> torch.Tensor:
    return g_local.float().pow(2).sum(dim=-2)


def _mat_l2_partial(g_local: torch.Tensor) -> torch.Tensor:
    return g_local.float().pow(2).sum(dim=(-2, -1))


# name -> local partial-sum function, called once per shape group (on a
# torch.stack-ed batch of same-shape FSDP-row-sharded local shards, not per
# param) -- see _apply_reduce_pre_norm_pass in disco.py.
PRE_NORM_PARTIAL_FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "col-l2": _col_l2_partial,
    "mat-l2": _mat_l2_partial,
}


def _col_l2_apply(
    g_local: torch.Tensor, reduced_sq: torch.Tensor, eps: float
) -> torch.Tensor:
    norm = reduced_sq.sqrt().unsqueeze(-2)
    return (g_local.float() / (norm + eps)).to(g_local.dtype)


def _mat_l2_apply(
    g_local: torch.Tensor, reduced_sq: torch.Tensor, eps: float
) -> torch.Tensor:
    norm = reduced_sq.sqrt().view(*reduced_sq.shape, 1, 1)
    return (g_local.float() / (norm + eps)).to(g_local.dtype)


# name -> (g_local_stacked, all_reduced_partial, eps) -> normalized stacked
# local shards, same shape-grouped batch as PRE_NORM_PARTIAL_FUNCTIONS.
PRE_NORM_SHARDED_APPLY_FUNCTIONS: dict[
    str, Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
] = {
    "col-l2": _col_l2_apply,
    "mat-l2": _mat_l2_apply,
}


def pre_norm_category(pre_norm: str) -> str:
    return pre_norm.split("-", 1)[0]  # "identity" / "row" / "col" / "mat"
