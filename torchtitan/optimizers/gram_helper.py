# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Scaffold for metrics that are a function of BOTH a weight matrix `W` and its
associated update matrix `U` (same shape) -- e.g. alignment/similarity between
a weight and its update -- mirroring norm_helper.calculate_norm's contract but
kept in its own file since gram metrics operate on a *pair* of tensors instead
of one. No formulas are implemented yet: `GRAM_METRIC_FUNCTIONS` is empty, so
`calculate_gram_metrics` currently always returns `{}`. Every call site in
disco.py already treats an empty dict as a valid, cheap no-op.
"""

from typing import Callable

import torch
from torch.distributed.tensor import DTensor

# name -> callable(W, U) -> 0-d torch.Tensor.
GRAM_METRIC_FUNCTIONS: dict[
    str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {}


def _prep(X: torch.Tensor, transpose: bool) -> torch.Tensor:
    if isinstance(X, torch.nn.Parameter):
        X = X.data
    if isinstance(X, DTensor):
        X = X.to_local()
    if X.ndim == 1 and X.numel() > 1:
        X = torch.diag_embed(X)
    if transpose:
        X = X.transpose(0, 1)
    return X


def calculate_gram_metrics(
    W: torch.Tensor,
    U: torch.Tensor,
    gram_metrics_to_log: list[str] | None = None,
    transpose: bool = False,
) -> dict[str, torch.Tensor]:
    """
    It is important to note that the order of the metrics is the same
    as the order of `gram_metrics_to_log`.

    Mirrors `norm_helper.calculate_norm`'s unwrapping contract for both `W`
    and `U`: unwraps `torch.nn.Parameter`/`DTensor` to a plain local tensor,
    expands a 1-D tensor to a diagonal matrix, and applies `transpose`.
    """
    if gram_metrics_to_log is None:
        gram_metrics_to_log = list(GRAM_METRIC_FUNCTIONS.keys())

    W = _prep(W, transpose)
    U = _prep(U, transpose)

    return {name: GRAM_METRIC_FUNCTIONS[name](W, U) for name in gram_metrics_to_log}
