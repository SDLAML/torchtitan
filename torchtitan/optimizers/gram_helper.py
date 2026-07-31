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


def _row_normalise(X: torch.Tensor, eps: float) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp_min(eps)


def _corr(X: torch.Tensor, eps: float) -> torch.Tensor:
    X_hat = _row_normalise(X, eps)
    return X_hat @ X_hat.T


def _cross_corr(X: torch.Tensor, Y: torch.Tensor, eps: float) -> torch.Tensor:
    return _row_normalise(X, eps) @ _row_normalise(Y, eps).T


def _offdiag(A: torch.Tensor) -> torch.Tensor:
    return A - torch.diag_embed(torch.diagonal(A))


def _row_dominance(A: torch.Tensor, m: int, eps: float) -> torch.Tensor:
    diagonal = torch.diagonal(A)
    off_mean = (A.abs().sum(dim=1) - diagonal.abs()) / max(m - 1, 1)
    return diagonal.abs() / (off_mean + eps)


def _offdiag_stats(
    C: torch.Tensor, m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    off_mask = ~torch.eye(m, dtype=torch.bool, device=C.device)
    x = C[off_mask]
    ax = x.abs()
    return ax.mean(), x.square().mean().sqrt(), ax.max()


def _cross_summary(C_XY: torch.Tensor, m: int, eps: float) -> dict[str, torch.Tensor]:
    diagonal = torch.diagonal(C_XY)
    off_mean = (C_XY.abs().sum(dim=1) - diagonal.abs()) / max(m - 1, 1)
    specificity = diagonal.abs() / (off_mean + eps)
    indices = torch.arange(m, device=C_XY.device)
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
            diagonal.square().sum() / (C_XY.square().sum() + eps)
        ),
    }


def _matrix_cosine(A: torch.Tensor, B: torch.Tensor, eps: float) -> torch.Tensor:
    return (A * B).sum() / (A.norm() * B.norm() + eps)


def _effective_rank(eigenvalues: torch.Tensor, eps: float) -> torch.Tensor:
    values = eigenvalues.clamp_min(0)
    probabilities = values / (values.sum() + eps)
    entropy = -(probabilities * torch.log(probabilities + eps)).sum()
    return torch.exp(entropy)


def _spectral_summary(
    eigenvalues: torch.Tensor, eps: float, topk: int
) -> dict[str, torch.Tensor]:
    # Expects descending-sorted, non-negative-clamped eigenvalues.
    eigenvalues = eigenvalues.clamp_min(0)
    total = eigenvalues.sum()
    k = min(topk, eigenvalues.numel())
    return {
        "effective_rank": _effective_rank(eigenvalues, eps),
        "largest": eigenvalues[0],
        "smallest": eigenvalues[-1],
        "condition_regularized": eigenvalues[0] / (eigenvalues[-1] + eps),
        "topk_energy_fraction": eigenvalues[:k].sum() / (total + eps),
    }


def _inverse_sqrt_from_eigh(
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    matrix_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    # Scale-relative floor (NOT a bare eps clamp) -- keeps this
    # scale-invariant across params of very different magnitude.
    floor = eps * matrix_scale.clamp_min(eps)
    inv_sqrt_values = eigenvalues.clamp_min(floor).rsqrt()
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
    W_before: torch.Tensor, V_raw: torch.Tensor, W_after: torch.Tensor, eps: float
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
        _corr(W_before, eps),
        _corr(W_after, eps),
        _corr(V_raw, eps),
        _corr(U_actual, eps),
    )
    C_WV = _cross_corr(W_before, V_raw, eps)
    C_WU = _cross_corr(W_before, U_actual, eps)
    C_VA = _cross_corr(V_raw, A_actual, eps)

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


def _level1_metrics(core: _GramCore, eps: float) -> dict[str, torch.Tensor]:
    m = core.m

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
    VA = _cross_summary(core.C_VA, m, eps)
    WV = _cross_summary(core.C_WV, m, eps)
    WU = _cross_summary(core.C_WU, m, eps)

    U_relative_step_fro = core.U_actual.norm() / (core.W_before.norm() + eps)
    update_to_momentum_isotropy_ratio = U_rms / (V_rms + eps)
    gram_geometry_alignment = _matrix_cosine(
        _offdiag(core.C_V), _offdiag(core.C_U), eps
    )
    global_direction_alignment = _matrix_cosine(core.V_raw, core.A_actual, eps)

    gram_change_relative = core.delta_GW.norm() / (core.G_Wm.norm() + eps)
    correlation_change_per_row = core.delta_CW.norm() / (m**0.5)
    relational_change_fraction = _offdiag(core.delta_GW).square().sum() / (
        core.delta_GW.square().sum() + eps
    )
    gram_identity_residual = (core.delta_GW - core.reconstructed_delta_GW).norm() / (
        core.delta_GW.norm() + eps
    )

    return {
        "V_R_raw": _row_dominance(core.G_V, m, eps),
        "V_R_cos": _row_dominance(core.C_V, m, eps),
        "U_R_raw": _row_dominance(core.G_U, m, eps),
        "U_R_cos": _row_dominance(core.C_U, m, eps),
        "Wm_R_raw": _row_dominance(core.G_Wm, m, eps),
        "Wm_R_cos": _row_dominance(core.C_Wm, m, eps),
        "Wp_R_raw": _row_dominance(core.G_Wp, m, eps),
        "Wp_R_cos": _row_dominance(core.C_Wp, m, eps),
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
    core: _GramCore, eps: float, topk: int
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
    q_V_dist = q_V / (q_V.sum() + eps)
    q_U_dist = q_U / (q_U.sum() + eps)

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
        for name, val in _spectral_summary(eig, eps, topk).items():
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
            "G_W_effective_rank_delta": (
                _effective_rank(gwp, eps) - _effective_rank(gwm, eps)
            ),
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
    K_U_log_rate_spread = torch.log(K_U_rates + eps).std()

    norm_KV = eig_KV / (eig_KV.sum() + eps)
    norm_KU = eig_KU / (eig_KU.sum() + eps)
    relative_spectrum_l1_distance = (norm_KV - norm_KU).abs().sum()

    J_abs_sum = eig_J.abs().sum() + eps
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

    core = _build_gram_core(W_before, V_raw, W_after, eps)
    out: dict[str, torch.Tensor] = dict(_level1_metrics(core, eps))
    extras = None
    if level >= 2:
        lvl2, extras = _level2_metrics(core, eps, topk)
        out.update(lvl2)
    if level >= 3:
        out.update(_level3_metrics(core, eps, extras))
    return out
