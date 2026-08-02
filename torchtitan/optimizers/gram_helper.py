# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Gram-based metrics comparing a weight's before/after-update geometry against
its raw momentum -- see gram_matrix.md (repo root) for the full math
writeup this ports; formulas below follow that spec's reference
`get_gram_metrics` implementation directly.

Three tensors, all the same shape:
  - `W_before` -- the weight before this step's update.
  - `V_raw`    -- the RAW effective gradient/momentum (whatever's fed into
                  `AbstractDiSCO.lmo()`), not the LMO-processed update.
  - `W_after`  -- the weight after this step's update. disco.py passes the
                  cheap `pseudo_w` approximation here (the same tensor
                  already used for track_param_* norms), not a fresh real
                  post-update read.

`U = W_after - W_before` (the exact realised displacement) and `A = -U` are
derived internally and used throughout -- see gram_matrix.md's "which tensor
answers which question" table for the intuition (raw momentum `V` studies
emergent optimizer-state geometry; the realised displacement `U` studies the
actual trajectory that moved the weights).

Three cumulative levels, gated by a single `level: int` argument:
  0: nothing (returns {} immediately -- the cheap no-op every disco.py call
     site relies on).
  1: O(m^2) entrywise geometry (row-dominance ratios, off-diagonal
     correlation stats, weight/momentum/update self- and cross-alignment,
     exact weight-Gram change).
  2: adds O(m^3) spectral structure (Gram/correlation eigenspectra for all
     4 tensors, subspace overlaps, weight-eigenbasis energy).
  3: adds whitened/generalised geometry (relative-motion and relative-flow
     eigenspectra, cross-Gram singular values, canonical correlations) --
     most expensive, most numerically sensitive. The reference's optional
     `include_cross_svd` extras (cross-Gram singular values, canonical
     correlations) are folded in unconditionally here rather than exposed
     as a separate flag -- both only ever run at level 3.

Every vector-valued metric has length `m` (the row-count of the
prepped/transposed matrix) -- every Gram/correlation matrix here is square
`m x m` (built as `X @ Y.T`, never `Y.T @ X`).

Where a full vector is itself returned, its generic percentile/mean/min/max
summary is intentionally NOT also returned (redundant, reconstructable
post-hoc from the logged vector); named *nonlinear* reductions are kept
alongside their vector. Two vectors from the reference are dropped as
trivially derivable from an already-logged vector: `raw_trace_normalised`/
`actual_trace_normalised` (a simple `/ sum()` of `K_V_eigenvalues`/
`K_U_eigenvalues`, both already returned).
"""

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor

_DEFAULT_GRAM_EPS: float = 1e-6
_DEFAULT_GRAM_TOPK: int = 8


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


def gram_vector_len(shape: tuple[int, ...], transpose: bool = False) -> int:
    """
    Static, shape-derived length of every vector-valued gram metric for a
    parameter of this local shape -- mirrors `_prep`'s diag_embed/transpose
    handling without needing an actual tensor, for disco.py's per-param
    offset-table precompute (DDP/FSDP/experts). Always `m`, never
    `min(m, n)` -- see module docstring.
    """
    if len(shape) == 1:
        return max(int(shape[0]), 1)
    return int(shape[-1]) if transpose else int(shape[-2])


def _gram(X: torch.Tensor) -> torch.Tensor:
    return X @ X.T


def _relative_floor(reference: torch.Tensor, eps_rel: float) -> torch.Tensor:
    # Scale-relative floor for a denominator that's structurally unbounded
    # relative to its numerator (can be exactly 0 while the numerator is
    # positive -- e.g. perfectly orthogonal rows, or a rank-deficient Gram
    # matrix) -- keeps the resulting *ratio* capped at a fixed,
    # scale-independent ceiling (1 / eps_rel) instead of blowing up
    # arbitrarily as the true denominator approaches 0. An absolute eps
    # can't do this: it doesn't scale with the input, so it either swamps
    # small-but-legitimate values (business-scale eps) or lets the ratio
    # blow up unpredictably near 0 (any fixed small eps). Backstopped with
    # an absolute machine-tiny floor for the fully-degenerate case where
    # `reference` itself is exactly 0 (avoids a literal 0/0).
    tiny = torch.finfo(reference.dtype).tiny
    return (eps_rel * reference).clamp_min(tiny)


def _row_normalise(X: torch.Tensor) -> torch.Tensor:
    # Floor each row by its OWN norm only, never a whole-matrix reference:
    # a matrix-wide floor under-normalises any row that's disproportionately
    # smaller than the rest of the matrix (e.g. a near-dead neuron sitting
    # alongside normal-scale rows), since the floor would then reflect the
    # OTHER rows' scale, not this row's. A row's norm is either exactly 0
    # (undefined direction) or some positive value that normalises
    # correctly on its own terms regardless of other rows' scale, so a bare
    # machine-tiny floor is both correct and sufficient.
    norm = X.norm(dim=1, keepdim=True)
    tiny = torch.finfo(X.dtype).tiny
    normalised = X / norm.clamp_min(tiny)
    return torch.where(norm > 0, normalised, torch.zeros_like(normalised))


def _corr(X: torch.Tensor) -> torch.Tensor:
    X_hat = _row_normalise(X)
    return X_hat @ X_hat.T


def _cross_corr(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return _row_normalise(X) @ _row_normalise(Y).T


def _offdiag(A: torch.Tensor) -> torch.Tensor:
    return A - torch.diag_embed(torch.diagonal(A))


def _dominance_ratio(diagonal: torch.Tensor, off_mean: torch.Tensor) -> torch.Tensor:
    # diagonal / off_mean, meant to be scale-invariant (both are reductions
    # of the same matrix) -- shared by _row_dominance and _cross_summary's
    # specificity, which were the same formula duplicated inline.
    eps_rel = torch.finfo(diagonal.dtype).eps
    denominator = torch.maximum(off_mean, _relative_floor(diagonal, eps_rel))
    result = torch.zeros_like(diagonal)
    nonzero = diagonal > 0
    result[nonzero] = diagonal[nonzero] / denominator[nonzero]
    return result


def _row_dominance(A: torch.Tensor, m: int) -> torch.Tensor:
    # Sum the off-diagonal entries directly (mask the diagonal out first)
    # rather than `row_sum - diagonal` -- the subtraction form suffers
    # catastrophic cancellation exactly when the matrix is highly
    # diagonal-dominant (the regime this metric is meant to detect):
    # row_sum ~= diagonal there, so their difference loses most of its
    # precision instead of correctly coming out small.
    abs_A = A.abs()
    diagonal = torch.diagonal(abs_A)
    off_A = abs_A.clone()
    off_A.fill_diagonal_(0)
    off_mean = off_A.sum(dim=1) / max(m - 1, 1)
    return _dominance_ratio(diagonal, off_mean)


def _offdiag_stats(
    C: torch.Tensor, m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    off_mask = ~torch.eye(m, dtype=torch.bool, device=C.device)
    x = C[off_mask]
    ax = x.abs()
    return ax.mean(), x.square().mean().sqrt(), ax.max()


def _cross_summary(C_XY: torch.Tensor, m: int) -> dict[str, torch.Tensor]:
    diagonal = torch.diagonal(C_XY)
    # Same off-diagonal-masking fix as _row_dominance -- avoid
    # `row_sum - diagonal`'s catastrophic cancellation under high diagonal
    # dominance.
    abs_C = C_XY.abs()
    off_C = abs_C.clone()
    off_C.fill_diagonal_(0)
    off_mean = off_C.sum(dim=1) / max(m - 1, 1)
    specificity = _dominance_ratio(diagonal.abs(), off_mean)
    indices = torch.arange(m, device=C_XY.device)
    # diagonal_energy_fraction's numerator is a strict energy subset of its
    # denominator (diagonal^2 <= sum of all entries^2), so it's already
    # bounded in [0, 1] -- a small absolute floor (not a relative one) is
    # enough to avoid 0/0 without risking any blow-up.
    tiny = torch.finfo(C_XY.dtype).tiny
    return {
        "diagonal": diagonal,
        "row_specificity": specificity,
        "row_top1_identity": (
            (C_XY.abs().argmax(dim=1) == indices).to(C_XY.dtype).mean()
        ),
        "column_top1_identity": (
            (C_XY.abs().argmax(dim=0) == indices).to(C_XY.dtype).mean()
        ),
        "diagonal_energy_fraction": (
            diagonal.square().sum() / C_XY.square().sum().clamp_min(tiny)
        ),
    }


def _matrix_cosine(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    # Cauchy-Schwarz bounds this in [-1, 1] regardless of A/B's scale, so a
    # tiny absolute floor (not a relative one) is enough to avoid 0/0.
    tiny = torch.finfo(A.dtype).tiny
    return (A * B).sum() / (A.norm() * B.norm()).clamp_min(tiny)


def _effective_rank(eigenvalues: torch.Tensor) -> torch.Tensor:
    values = eigenvalues.clamp_min(0)
    total = values.sum()
    # The probability sum is bounded (each value <= the sum of all
    # non-negative values), so a tiny absolute floor suffices here -- an
    # absolute business-scale eps would otherwise stop `probabilities` from
    # summing to ~1 whenever the whole spectrum is uniformly small,
    # corrupting the entropy below. `xlogy` handles the p=0 entropy term
    # exactly (0 * log(0) := 0 by definition) with no eps-in-the-log fudge
    # needed, and without an eps distorting log(p) for small-but-positive p
    # the way `log(p + eps)` would.
    tiny = torch.finfo(values.dtype).tiny
    probabilities = values / total.clamp_min(tiny)
    entropy = -torch.xlogy(probabilities, probabilities).sum()
    effective_rank = torch.exp(entropy)
    # A fully zero spectrum spans no directions -- effective_rank should be
    # 0, not exp(0)=1 (which the formula above would otherwise give: every
    # probability is 0/tiny=0, xlogy(0,0)=0, so entropy=0).
    return torch.where(total > 0, effective_rank, torch.zeros_like(effective_rank))


def _spectral_summary(eigenvalues: torch.Tensor, topk: int) -> dict[str, torch.Tensor]:
    # Expects descending-sorted, non-negative-clamped eigenvalues.
    eigenvalues = eigenvalues.clamp_min(0)
    total = eigenvalues.sum()
    k = min(topk, eigenvalues.numel())
    eps_rel = torch.finfo(eigenvalues.dtype).eps
    tiny = torch.finfo(eigenvalues.dtype).tiny
    return {
        "effective_rank": _effective_rank(eigenvalues),
        "largest": eigenvalues[0],
        "smallest": eigenvalues[-1],
        # Condition number is genuinely unbounded (smallest eigenvalue can
        # be exactly 0 for a rank-deficient matrix) -- same relative-floor
        # treatment as row_dominance, not a tiny absolute floor.
        "condition_regularized": eigenvalues[0]
        / torch.maximum(eigenvalues[-1], _relative_floor(eigenvalues[0], eps_rel)),
        "topk_energy_fraction": eigenvalues[:k].sum() / total.clamp_min(tiny),
    }


def _inverse_sqrt_from_eigh(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    matrix_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Scale-relative floor (NOT a bare eps clamp) -- keeps this
    # scale-invariant across params of very different magnitude. The old
    # `matrix_scale.clamp_min(eps)` broke exactly that for matrix_scale <
    # eps: clamping matrix_scale itself up to eps first made the floor
    # collapse to a fixed eps^2 regardless of how much smaller matrix_scale
    # actually was -- the same absolute-floor swamping bug fixed elsewhere
    # in this file, hiding here too. Floor purely proportionally instead,
    # backstopped only by machine-tiny for the literal matrix_scale == 0
    # case.
    tiny = torch.finfo(eigenvalues.dtype).tiny
    scale = matrix_scale.clamp_min(0)
    floor = (eps * scale).clamp_min(tiny)
    # A fully zero-scale matrix has no direction to whiten relative to --
    # define its inverse-sqrt as the zero matrix rather than an arbitrary
    # huge value from flooring near-zero eigenvalues up to `tiny`.
    active = (scale > 0).to(eigenvalues.dtype)
    inv_sqrt_values = eigenvalues.clamp_min(floor).rsqrt() * active
    return (eigenvectors * inv_sqrt_values.unsqueeze(0)) @ eigenvectors.T


@dataclass
class _GramCore:
    m: int
    W_before: torch.Tensor
    V_raw: torch.Tensor
    W_after: torch.Tensor
    U_actual: torch.Tensor
    A_actual: torch.Tensor
    G_Wm: torch.Tensor
    G_Wp: torch.Tensor
    G_V: torch.Tensor
    G_U: torch.Tensor
    C_Wm: torch.Tensor
    C_Wp: torch.Tensor
    C_V: torch.Tensor
    C_U: torch.Tensor
    C_WV: torch.Tensor
    C_WU: torch.Tensor
    C_VA: torch.Tensor
    delta_GW: torch.Tensor
    delta_CW: torch.Tensor
    reconstructed_delta_GW: torch.Tensor


def _build_gram_core(
    W_before: torch.Tensor, V_raw: torch.Tensor, W_after: torch.Tensor
) -> _GramCore:
    m = W_before.shape[0]
    U_actual = W_after - W_before
    A_actual = -U_actual

    G_Wm, G_Wp, G_V, G_U = (
        _gram(W_before),
        _gram(W_after),
        _gram(V_raw),
        _gram(U_actual),
    )
    C_Wm, C_Wp, C_V, C_U = (
        _corr(W_before),
        _corr(W_after),
        _corr(V_raw),
        _corr(U_actual),
    )
    C_WV = _cross_corr(W_before, V_raw)
    C_WU = _cross_corr(W_before, U_actual)
    C_VA = _cross_corr(V_raw, A_actual)

    delta_GW = G_Wp - G_Wm
    delta_CW = C_Wp - C_Wm
    # Exact algebraic identity: G_Wp - G_Wm == W_before@U^T + U@W_before^T +
    # U@U^T (U = W_after - W_before) -- see gram_identity_residual below.
    reconstructed_delta_GW = W_before @ U_actual.T + U_actual @ W_before.T + G_U

    return _GramCore(
        m=m,
        W_before=W_before,
        V_raw=V_raw,
        W_after=W_after,
        U_actual=U_actual,
        A_actual=A_actual,
        G_Wm=G_Wm,
        G_Wp=G_Wp,
        G_V=G_V,
        G_U=G_U,
        C_Wm=C_Wm,
        C_Wp=C_Wp,
        C_V=C_V,
        C_U=C_U,
        C_WV=C_WV,
        C_WU=C_WU,
        C_VA=C_VA,
        delta_GW=delta_GW,
        delta_CW=delta_CW,
        reconstructed_delta_GW=reconstructed_delta_GW,
    )


def _level1_metrics(core: _GramCore) -> dict[str, torch.Tensor]:
    m = core.m
    # Cross-tensor ratios below (numerator/denominator from genuinely
    # different tensors, e.g. update norm vs. weight norm) have no natural
    # same-tensor relative reference -- a weight/momentum/etc. can be
    # legitimately exactly 0, so full scale-invariance isn't achievable.
    # Just avoid literal 0/0 with a tiny absolute floor, same as any other
    # structurally-bounded-elsewhere ratio.
    tiny = torch.finfo(core.W_before.dtype).tiny

    V_mean_abs, V_rms, V_max_abs = _offdiag_stats(core.C_V, m)
    U_mean_abs, U_rms, U_max_abs = _offdiag_stats(core.C_U, m)
    Wm_mean_abs, Wm_rms, Wm_max_abs = _offdiag_stats(core.C_Wm, m)
    Wp_mean_abs, Wp_rms, Wp_max_abs = _offdiag_stats(core.C_Wp, m)

    # Sign-convention note: VA is built from A_actual (= -U_actual, the
    # descent-oriented direction), but WU is built from U_actual itself (the
    # literal displacement W_after - W_before) -- these are DIFFERENT sign
    # conventions relative to each other by design (WU_diagonal answers "did
    # this row grow/shrink", VA_diagonal answers "does momentum point the
    # way weights actually moved"). If you want a descent-oriented
    # counterpart of WU (cos(w_before, a) instead of cos(w_before, u)), it's
    # a pure sign flip -- row_normalise(-X) == -row_normalise(X), so
    # cross_corr(W_before, A_actual) == -C_WU exactly (not just its
    # diagonal) -- WA_diagonal = -WU["diagonal"], WA_row_specificity ==
    # WU["row_specificity"] unchanged (abs()-based), etc. Not returned as a
    # separate metric since it's trivially derivable from what's already
    # logged (same "drop what's a linear/simple transform of an
    # already-logged vector" rule as principal_angles_radians before).
    VA = _cross_summary(core.C_VA, m)
    WV = _cross_summary(core.C_WV, m)
    WU = _cross_summary(core.C_WU, m)

    U_relative_step_fro = core.U_actual.norm() / core.W_before.norm().clamp_min(tiny)
    update_to_momentum_isotropy_ratio = U_rms / V_rms.clamp_min(tiny)
    gram_geometry_alignment = _matrix_cosine(_offdiag(core.C_V), _offdiag(core.C_U))
    global_direction_alignment = _matrix_cosine(core.V_raw, core.A_actual)

    gram_change_relative = core.delta_GW.norm() / core.G_Wm.norm().clamp_min(tiny)
    correlation_change_per_row = core.delta_CW.norm() / (m**0.5)
    relational_change_fraction = _offdiag(core.delta_GW).square().sum() / (
        core.delta_GW.square().sum().clamp_min(tiny)
    )
    gram_identity_residual = (
        core.delta_GW - core.reconstructed_delta_GW
    ).norm() / core.delta_GW.norm().clamp_min(tiny)

    return {
        "V_R_raw": _row_dominance(core.G_V, m),
        "V_R_cos": _row_dominance(core.C_V, m),
        "U_R_raw": _row_dominance(core.G_U, m),
        "U_R_cos": _row_dominance(core.C_U, m),
        "Wm_R_raw": _row_dominance(core.G_Wm, m),
        "Wm_R_cos": _row_dominance(core.C_Wm, m),
        "Wp_R_raw": _row_dominance(core.G_Wp, m),
        "Wp_R_cos": _row_dominance(core.C_Wp, m),
        "VA_diagonal": VA["diagonal"],
        "VA_row_specificity": VA["row_specificity"],
        "WV_diagonal": WV["diagonal"],
        "WV_row_specificity": WV["row_specificity"],
        "WU_diagonal": WU["diagonal"],
        "WU_row_specificity": WU["row_specificity"],
        "V_offdiag_mean_abs": V_mean_abs,
        "V_offdiag_rms": V_rms,
        "V_offdiag_max_abs": V_max_abs,
        "U_offdiag_mean_abs": U_mean_abs,
        "U_offdiag_rms": U_rms,
        "U_offdiag_max_abs": U_max_abs,
        "U_relative_step_fro": U_relative_step_fro,
        "Wm_offdiag_mean_abs": Wm_mean_abs,
        "Wm_offdiag_rms": Wm_rms,
        "Wm_offdiag_max_abs": Wm_max_abs,
        "Wp_offdiag_mean_abs": Wp_mean_abs,
        "Wp_offdiag_rms": Wp_rms,
        "Wp_offdiag_max_abs": Wp_max_abs,
        "update_to_momentum_isotropy_ratio": update_to_momentum_isotropy_ratio,
        "gram_geometry_alignment": gram_geometry_alignment,
        "global_direction_alignment": global_direction_alignment,
        "VA_row_top1_identity": VA["row_top1_identity"],
        "VA_column_top1_identity": VA["column_top1_identity"],
        "VA_diagonal_energy_fraction": VA["diagonal_energy_fraction"],
        "WV_row_top1_identity": WV["row_top1_identity"],
        "WV_column_top1_identity": WV["column_top1_identity"],
        "WV_diagonal_energy_fraction": WV["diagonal_energy_fraction"],
        "WU_row_top1_identity": WU["row_top1_identity"],
        "WU_column_top1_identity": WU["column_top1_identity"],
        "WU_diagonal_energy_fraction": WU["diagonal_energy_fraction"],
        "gram_change_relative": gram_change_relative,
        "correlation_change_per_row": correlation_change_per_row,
        "relational_change_fraction": relational_change_fraction,
        # Tautologically ~0 (floating-point precision only): U_actual is
        # always derived as W_after - W_before internally, never
        # independently measured, so this identity holds by construction
        # regardless of how faithful W_after (pseudo_w) is to a true
        # post-update read. Kept as a cheap dtype/precision sanity monitor,
        # not a training-dynamics signal -- don't expect it to correlate
        # with anything interesting.
        "gram_identity_residual": gram_identity_residual,
    }


@dataclass
class _Level2Extras:
    # Ascending order, straight from eigh -- level 3 reuses these directly
    # rather than re-flipping to descending and back: U @ diag(f(eigvals)) @
    # U.T is invariant to eigenpair ordering, so this is a pure
    # simplification, not a behavior change.
    gwm_asc: torch.Tensor
    UWm_asc: torch.Tensor
    gv_asc: torch.Tensor
    UV_asc: torch.Tensor
    gu_asc: torch.Tensor
    UU_asc: torch.Tensor


def _level2_metrics(
    core: _GramCore, topk: int
) -> tuple[dict[str, torch.Tensor], _Level2Extras]:
    m = core.m

    gwm_asc, UWm_asc = torch.linalg.eigh(core.G_Wm)
    gwp_asc, UWp_asc = torch.linalg.eigh(core.G_Wp)
    gv_asc, UV_asc = torch.linalg.eigh(core.G_V)
    gu_asc, UU_asc = torch.linalg.eigh(core.G_U)

    gwm, UWm = gwm_asc.flip(0), UWm_asc.flip(1)
    gwp, UWp = gwp_asc.flip(0), UWp_asc.flip(1)
    gv, UV = gv_asc.flip(0), UV_asc.flip(1)
    gu, UU = gu_asc.flip(0), UU_asc.flip(1)

    cwm = torch.linalg.eigvalsh(core.C_Wm).flip(0)
    cwp = torch.linalg.eigvalsh(core.C_Wp).flip(0)
    cv = torch.linalg.eigvalsh(core.C_V).flip(0)
    cu = torch.linalg.eigvalsh(core.C_U).flip(0)

    k = min(topk, m)

    def overlap(UX: torch.Tensor, UY: torch.Tensor) -> torch.Tensor:
        return (UX[:, :k].T @ UY[:, :k]).square().sum() / k

    q_V = torch.diagonal(UWm.T @ core.G_V @ UWm).clamp_min(0)
    q_U = torch.diagonal(UWm.T @ core.G_U @ UWm).clamp_min(0)
    # Bounded distributions (each entry <= the sum of all non-negative
    # entries) -- a tiny absolute floor suffices, same reasoning as
    # _effective_rank's probabilities.
    q_tiny = torch.finfo(q_V.dtype).tiny
    q_V_dist = q_V / q_V.sum().clamp_min(q_tiny)
    q_U_dist = q_U / q_U.sum().clamp_min(q_tiny)

    scalars: dict[str, torch.Tensor] = {}
    for prefix, eig in (
        ("G_Wm", gwm),
        ("G_Wp", gwp),
        ("G_V", gv),
        ("G_U", gu),
        ("C_Wm", cwm),
        ("C_Wp", cwp),
        ("C_V", cv),
        ("C_U", cu),
    ):
        for name, val in _spectral_summary(eig, topk).items():
            scalars[f"{prefix}_{name}"] = val

    scalars.update(
        {
            "C_Wm_deviation_from_identity": (cwm - 1).square().mean().sqrt(),
            "C_Wp_deviation_from_identity": (cwp - 1).square().mean().sqrt(),
            "C_V_deviation_from_identity": (cv - 1).square().mean().sqrt(),
            "C_U_deviation_from_identity": (cu - 1).square().mean().sqrt(),
            "overlap_Wm_V": overlap(UWm, UV),
            "overlap_Wm_U": overlap(UWm, UU),
            "overlap_V_U": overlap(UV, UU),
            "overlap_Wm_Wp": overlap(UWm, UWp),
            "G_W_effective_rank_delta": (_effective_rank(gwp) - _effective_rank(gwm)),
        }
    )

    # gwp - gwm compares the i-th LARGEST eigenvalue of G_Wp against the
    # i-th largest of G_Wm -- a rank/position-wise comparison of two
    # independently-sorted spectra, NOT a per-eigenvector-tracked change
    # (sorting can reshuffle which actual eigenvector lands at position i
    # between the two matrices) -- named accordingly, not just
    # "eigenvalue_delta". Contrast with G_W_change_eigenvalues below, the
    # eigenvalues of the actual difference matrix delta_GW = G_Wp - G_Wm
    # itself -- a more principled measure of the Gram change's own spectral
    # content, unaffected by any eigenvector reshuffling between G_Wp/G_Wm.
    # Unlike G_W/C_W eigenvalues (always >=0, real Gram/correlation
    # matrices), delta_GW is a difference of two PSD matrices and generally
    # indefinite -- signed, not clamped, same treatment as level 3's
    # J_eigenvalues (delta_GW's whitened counterpart).
    G_W_rankwise_eigenvalue_delta = gwp - gwm
    G_W_change_eigenvalues = torch.linalg.eigvalsh(core.delta_GW).flip(0)

    vectors = {
        "G_Wm_eigenvalues": gwm,
        "G_Wp_eigenvalues": gwp,
        "G_V_eigenvalues": gv,
        "G_U_eigenvalues": gu,
        "C_Wm_eigenvalues": cwm,
        "C_Wp_eigenvalues": cwp,
        "C_V_eigenvalues": cv,
        "C_U_eigenvalues": cu,
        "energy_V_in_Wm_basis": q_V,
        "energy_U_in_Wm_basis": q_U,
        "energy_V_in_Wm_basis_distribution": q_V_dist,
        "energy_U_in_Wm_basis_distribution": q_U_dist,
        "G_W_rankwise_eigenvalue_delta": G_W_rankwise_eigenvalue_delta,
        "G_W_change_eigenvalues": G_W_change_eigenvalues,
    }

    extras = _Level2Extras(
        gwm_asc=gwm_asc,
        UWm_asc=UWm_asc,
        gv_asc=gv_asc,
        UV_asc=UV_asc,
        gu_asc=gu_asc,
        UU_asc=UU_asc,
    )
    return {**scalars, **vectors}, extras


def _log_rate_spread(rates: torch.Tensor) -> torch.Tensor:
    # log(rates + eps) has the same absolute-eps swamping problem as
    # everywhere else in this file: a genuine eigendirection the update
    # doesn't touch at all gives rate == 0 exactly (not just floating-point
    # noise), and if the OTHER rates are uniformly small too (a small
    # update relative to the weight, a real training regime), an absolute
    # eps makes every log(rate + eps) collapse toward the same log(eps)
    # constant, corrupting the spread. Floor each rate relative to the
    # largest rate instead -- unlike _row_normalise (where referencing
    # other rows was wrong, since rows are logically independent), the
    # rates being compared against each other via max_rate is exactly what
    # "spread" means here, so this is the right reference.
    eps_rel = torch.finfo(rates.dtype).eps
    max_rate = rates.max()
    floor = _relative_floor(max_rate, eps_rel)
    log_rates = torch.log(torch.maximum(rates, floor))
    spread = log_rates.std()
    return torch.where(max_rate > 0, spread, torch.zeros_like(spread))


def _level3_metrics(
    core: _GramCore, eps: float, extras: _Level2Extras
) -> dict[str, torch.Tensor]:
    W_scale = torch.diagonal(core.G_Wm).mean()
    V_scale = torch.diagonal(core.G_V).mean()
    U_scale = torch.diagonal(core.G_U).mean()

    GW_inv_sqrt = _inverse_sqrt_from_eigh(extras.gwm_asc, extras.UWm_asc, W_scale, eps)
    GV_inv_sqrt = _inverse_sqrt_from_eigh(extras.gv_asc, extras.UV_asc, V_scale, eps)
    GU_inv_sqrt = _inverse_sqrt_from_eigh(extras.gu_asc, extras.UU_asc, U_scale, eps)

    K_V = GW_inv_sqrt @ core.G_V @ GW_inv_sqrt
    K_U = GW_inv_sqrt @ core.G_U @ GW_inv_sqrt
    J = GW_inv_sqrt @ core.delta_GW @ GW_inv_sqrt
    K_V = 0.5 * (K_V + K_V.T)
    K_U = 0.5 * (K_U + K_U.T)
    J = 0.5 * (J + J.T)

    eig_KV = torch.linalg.eigvalsh(K_V).clamp_min(0).flip(0)
    eig_KU = torch.linalg.eigvalsh(K_U).clamp_min(0).flip(0)
    eig_J = torch.linalg.eigvalsh(J).flip(0)  # signed -- no clamp

    K_V_rates = eig_KV.sqrt()
    K_U_rates = eig_KU.sqrt()
    K_U_log_rate_spread = _log_rate_spread(K_U_rates)

    # Bounded distributions/fractions (each entry, or each signed part, is
    # <= the sum of all non-negative magnitudes) -- a tiny absolute floor
    # suffices, same reasoning as _effective_rank's probabilities.
    eig_tiny = torch.finfo(eig_KV.dtype).tiny
    norm_KV = eig_KV / eig_KV.sum().clamp_min(eig_tiny)
    norm_KU = eig_KU / eig_KU.sum().clamp_min(eig_tiny)
    relative_spectrum_l1_distance = (norm_KV - norm_KU).abs().sum()

    J_abs_sum = eig_J.abs().sum().clamp_min(torch.finfo(eig_J.dtype).tiny)
    J_positive_fraction = eig_J.clamp_min(0).sum() / J_abs_sum
    J_negative_fraction = (-eig_J.clamp_max(0)).sum() / J_abs_sum

    Q_WV = GW_inv_sqrt @ (core.W_before @ core.V_raw.T) @ GV_inv_sqrt
    Q_WU = GW_inv_sqrt @ (core.W_before @ core.U_actual.T) @ GU_inv_sqrt
    Q_VA = GV_inv_sqrt @ (core.V_raw @ core.A_actual.T) @ GU_inv_sqrt

    return {
        "K_V_eigenvalues": eig_KV,
        "K_V_rates": K_V_rates,
        "K_U_eigenvalues": eig_KU,
        "K_U_rates": K_U_rates,
        "J_eigenvalues": eig_J,
        "C_WV_singular_values": torch.linalg.svdvals(core.C_WV),
        "C_WU_singular_values": torch.linalg.svdvals(core.C_WU),
        "C_VA_singular_values": torch.linalg.svdvals(core.C_VA),
        "Q_WV_canonical_correlations": torch.linalg.svdvals(Q_WV),
        "Q_WU_canonical_correlations": torch.linalg.svdvals(Q_WU),
        "Q_VA_canonical_correlations": torch.linalg.svdvals(Q_VA),
        "K_U_log_rate_spread": K_U_log_rate_spread,
        "relative_spectrum_l1_distance": relative_spectrum_l1_distance,
        "J_positive_fraction": J_positive_fraction,
        "J_negative_fraction": J_negative_fraction,
    }


GRAM_SCALAR_NAMES_BY_LEVEL: dict[int, list[str]] = {
    1: [
        "V_offdiag_mean_abs",
        "V_offdiag_rms",
        "V_offdiag_max_abs",
        "U_offdiag_mean_abs",
        "U_offdiag_rms",
        "U_offdiag_max_abs",
        "U_relative_step_fro",
        "Wm_offdiag_mean_abs",
        "Wm_offdiag_rms",
        "Wm_offdiag_max_abs",
        "Wp_offdiag_mean_abs",
        "Wp_offdiag_rms",
        "Wp_offdiag_max_abs",
        "update_to_momentum_isotropy_ratio",
        "gram_geometry_alignment",
        "global_direction_alignment",
        "VA_row_top1_identity",
        "VA_column_top1_identity",
        "VA_diagonal_energy_fraction",
        "WV_row_top1_identity",
        "WV_column_top1_identity",
        "WV_diagonal_energy_fraction",
        "WU_row_top1_identity",
        "WU_column_top1_identity",
        "WU_diagonal_energy_fraction",
        "gram_change_relative",
        "correlation_change_per_row",
        "relational_change_fraction",
        "gram_identity_residual",
    ],
}
GRAM_SCALAR_NAMES_BY_LEVEL[2] = (
    GRAM_SCALAR_NAMES_BY_LEVEL[1]
    + [
        f"{prefix}_{field}"
        for prefix in ("G_Wm", "G_Wp", "G_V", "G_U", "C_Wm", "C_Wp", "C_V", "C_U")
        for field in (
            "effective_rank",
            "largest",
            "smallest",
            "condition_regularized",
            "topk_energy_fraction",
        )
    ]
    + [
        "C_Wm_deviation_from_identity",
        "C_Wp_deviation_from_identity",
        "C_V_deviation_from_identity",
        "C_U_deviation_from_identity",
        "overlap_Wm_V",
        "overlap_Wm_U",
        "overlap_V_U",
        "overlap_Wm_Wp",
        "G_W_effective_rank_delta",
    ]
)
GRAM_SCALAR_NAMES_BY_LEVEL[3] = GRAM_SCALAR_NAMES_BY_LEVEL[2] + [
    "K_U_log_rate_spread",
    "relative_spectrum_l1_distance",
    "J_positive_fraction",
    "J_negative_fraction",
]

GRAM_VECTOR_NAMES_BY_LEVEL: dict[int, list[str]] = {
    1: [
        "V_R_raw",
        "V_R_cos",
        "U_R_raw",
        "U_R_cos",
        "Wm_R_raw",
        "Wm_R_cos",
        "Wp_R_raw",
        "Wp_R_cos",
        "VA_diagonal",
        "VA_row_specificity",
        "WV_diagonal",
        "WV_row_specificity",
        "WU_diagonal",
        "WU_row_specificity",
    ],
}
GRAM_VECTOR_NAMES_BY_LEVEL[2] = GRAM_VECTOR_NAMES_BY_LEVEL[1] + [
    "G_Wm_eigenvalues",
    "G_Wp_eigenvalues",
    "G_V_eigenvalues",
    "G_U_eigenvalues",
    "C_Wm_eigenvalues",
    "C_Wp_eigenvalues",
    "C_V_eigenvalues",
    "C_U_eigenvalues",
    "energy_V_in_Wm_basis",
    "energy_U_in_Wm_basis",
    "energy_V_in_Wm_basis_distribution",
    "energy_U_in_Wm_basis_distribution",
    "G_W_rankwise_eigenvalue_delta",
    "G_W_change_eigenvalues",
]
GRAM_VECTOR_NAMES_BY_LEVEL[3] = GRAM_VECTOR_NAMES_BY_LEVEL[2] + [
    "K_V_eigenvalues",
    "K_V_rates",
    "K_U_eigenvalues",
    "K_U_rates",
    "J_eigenvalues",
    "C_WV_singular_values",
    "C_WU_singular_values",
    "C_VA_singular_values",
    "Q_WV_canonical_correlations",
    "Q_WU_canonical_correlations",
    "Q_VA_canonical_correlations",
]


def gram_scalar_names(level: int) -> list[str]:
    return GRAM_SCALAR_NAMES_BY_LEVEL.get(level, [])


def gram_vector_names(level: int) -> list[str]:
    return GRAM_VECTOR_NAMES_BY_LEVEL.get(level, [])


@torch.no_grad()
def calculate_gram_metrics(
    W_before: torch.Tensor,
    V_raw: torch.Tensor,
    W_after: torch.Tensor,
    level: int = 0,
    eps: float = _DEFAULT_GRAM_EPS,
    topk: int = _DEFAULT_GRAM_TOPK,
    transpose: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Cheap no-op ({}) for level <= 0 -- the single early-return point; every
    disco.py call site stays unconditional (see module docstring). `V_raw`
    should be the raw effective grad/momentum (whatever's fed into
    AbstractDiSCO.lmo()); `W_after` should be the post-update weight
    (disco.py passes `pseudo_w`) -- see readme.md.

    Mirrors norm_helper.calculate_norm's unwrap contract for all three
    tensors (Parameter/DTensor -> local tensor, 1-D -> diag_embed, optional
    transpose), then upcasts each to float32 if in fp16/bf16 (Gram/eigh/svd
    are unreliable in half precision).

    Returns a dict whose key set is a deterministic function of `level`
    alone (gram_scalar_names(level) + gram_vector_names(level)),
    independent of parameter shape -- disco.py's DDP/FSDP/experts packing
    code relies on this fixed arity. Degenerate shapes (m < 2) return {}
    rather than ill-defined values, same as level <= 0.
    """
    if level <= 0:
        return {}
    W_before = _prep(W_before, transpose)
    V_raw = _prep(V_raw, transpose)
    W_after = _prep(W_after, transpose)
    if (
        W_before.ndim < 2
        or V_raw.ndim < 2
        or W_after.ndim < 2
        or W_before.shape != V_raw.shape
        or W_before.shape != W_after.shape
        or W_before.shape[0] < 2
    ):
        return {}
    if W_before.dtype in (torch.float16, torch.bfloat16):
        W_before = W_before.float()
    if V_raw.dtype in (torch.float16, torch.bfloat16):
        V_raw = V_raw.float()
    if W_after.dtype in (torch.float16, torch.bfloat16):
        W_after = W_after.float()

    core = _build_gram_core(W_before, V_raw, W_after)
    out: dict[str, torch.Tensor] = dict(_level1_metrics(core))
    extras = None
    if level >= 2:
        lvl2, extras = _level2_metrics(core, topk)
        out.update(lvl2)
    if level >= 3:
        out.update(_level3_metrics(core, eps, extras))
    return out
