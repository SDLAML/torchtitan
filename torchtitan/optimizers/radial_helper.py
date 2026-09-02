# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Whole-tensor "radial dynamics" metrics -- how a weight's norm and direction
evolve under training.

The original metrics in this file use Frobenius geometry and only need the
weight before/after this step's update (`W_before`, `W_after` -- disco.py
passes the same `pseudo_w` already used as gram's `W_after`). They are
whole-tensor scalar metrics and are independent of `gram_level` and
`norms_to_log`.

Notation for the Frobenius metrics:
  W_t    = W_before, the weight before this step's update.
  W_t+1  = W_after, the weight after (disco.py's `pseudo_w` approximation).
  dW_t   = W_after - W_before, the realised displacement.
  r_t    = ||W_t||_F  (Frobenius "radius").
  a_t    = ||dW_t||_F (Frobenius raw step).
  q_t    = W_t / r_t,  v_t = dW_t / a_t.
  c_t    = <q_t, v_t> ("radial_cosine").

Four running accumulators (`raw_A2`, `angular_A1`, `angular_A2`, `R1`)
persist across steps in the caller-supplied `state` dict, mutated in place.
`R2(t)` from the radial-error formula is the same running sum as `raw_A2`,
so no separate `R2` accumulator is stored.

In addition, this module can compute update radiality in several induced
operator-norm geometries for 2-D matrix weights.  For a primal matrix norm
N and realised displacement U = W_after - W_before, the quantity is

    <D_{N*}(W_before), U / N(U)>,

where D_{N*}(W_before) is a norming covector of W_before.  It lies in
[-1, 1] away from degenerate cases and measures the signed first-order
alignment of the update with a selected outward normal of the N-unit ball.

For the same geometries, directional separation is

    N(W_after / N(W_after) - W_before / N(W_before)).

It compares the directions of two nonzero consecutive iterates without
depending on their radii.

Supported geometries:
  * "rms_to_rms": RMS -> RMS induced operator norm.
  * "rms_to_inf": RMS -> l_infinity induced operator norm.
  * "l1_to_rms":  l1  -> RMS induced operator norm.

All three are enabled by default, but callers can pass a subset through
`update_radiality_geometries`.  Disabled, non-applicable (non-2-D), or
degenerate geometry metrics are returned as NaN rather than 0.  Update
radiality is degenerate when N(W_before) = 0 or when
N(U) <= `_REL_DEGENERACY_EPS` * N(W_before); directional separation is
degenerate when either iterate has zero norm.  Zero remains a meaningful
valid result: tangent/aligned orthogonally for update radiality and unchanged
direction for directional separation.

Important cost note: the exact RMS -> RMS metrics require spectral-norm
computations for the current weight, update, next weight, and normalized
direction difference, so unlike the original
Frobenius scalar metrics it is not a cheap O(1)-style reduction.  The
RMS -> inf and l1 -> RMS metrics only require row/column L2 reductions.

At non-smooth points (e.g. repeated top singular values or ties between
maximal row/column norms), the norming covector is not unique.  The
implementation returns one valid selection: PyTorch's SVD selection for
RMS -> RMS, and the first argmax row/column for the max-row/max-column
geometries.
"""

from collections.abc import Iterable

import torch

from .norm_helper import (
    l1_to_rms_norm,
    rms_to_inf_norm,
    rms_to_rms_norm,
)


UPDATE_RADIALITY_GEOMETRIES: tuple[str, ...] = (
    "rms_to_rms",
    "rms_to_inf",
    "l1_to_rms",
)

_UPDATE_RADIALITY_METRIC_BY_GEOMETRY: dict[str, str] = {
    "rms_to_rms": "update_radiality_rms_to_rms",
    "rms_to_inf": "update_radiality_rms_to_inf",
    "l1_to_rms": "update_radiality_l1_to_rms",
}

_DIRECTIONAL_SEPARATION_METRIC_BY_GEOMETRY: dict[str, str] = {
    "rms_to_rms": "directional_separation_rms_to_rms",
    "rms_to_inf": "directional_separation_rms_to_inf",
    "l1_to_rms": "directional_separation_l1_to_rms",
}

RADIAL_METRIC_NAMES: list[str] = [
    "radius",
    "raw_step",
    "relative_step",
    "radial_cosine",
    "tangent_fraction",
    "angle",
    "angle_from_cos",
    "radial_first_order",
    "radial_second_order",
    "radial_ratio",
    "raw_A2",
    "angular_A1",
    "angular_A2",
    "R1",
    "E_radial",
    "update_radiality_rms_to_rms",
    "update_radiality_rms_to_inf",
    "update_radiality_l1_to_rms",
    "directional_separation_rms_to_rms",
    "directional_separation_rms_to_inf",
    "directional_separation_l1_to_rms",
]

_ACCUMULATOR_NAMES: tuple[str, ...] = ("raw_A2", "angular_A1", "angular_A2", "R1")

# An update whose norm is below this fraction of the current weight norm is
# treated as degenerate.  The comparison is performed in the corresponding
# geometry; fixed dimension-normalisation constants cancel on both sides.
_REL_DEGENERACY_EPS = 1e-6


def new_radial_state(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    shape: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    """Fresh, zero-initialized accumulator state.

    `shape=()` (the default) is for a single tracked tensor;
    `shape=(num_local_experts,)` is for expert params, where each expert
    index needs its own independent accumulators.  Index into the result
    per expert (for example `state["raw_A2"][ep_idx]`) when calling
    `calculate_radial_metrics`; the passed 0-d views are mutated in place.
    """
    return {
        name: torch.zeros(shape, device=device, dtype=dtype)
        for name in _ACCUMULATOR_NAMES
    }


def _validate_update_radiality_geometries(
    geometries: Iterable[str] | str | None,
) -> tuple[str, ...]:
    if geometries is None:
        return UPDATE_RADIALITY_GEOMETRIES
    if isinstance(geometries, str):
        geometries = (geometries,)

    selected = tuple(dict.fromkeys(geometries))
    for geometry in selected:
        if geometry not in UPDATE_RADIALITY_GEOMETRIES:
            supported = ", ".join(UPDATE_RADIALITY_GEOMETRIES)
            raise ValueError(
                f"Unknown update-radiality geometry {geometry!r}. "
                f"Supported geometries: {supported}."
            )
    return selected


def _geometry_norm(W: torch.Tensor, geometry: str) -> torch.Tensor:
    """Apply the canonical norm-helper function for a geometry."""
    if geometry == "rms_to_rms":
        return rms_to_rms_norm(W)
    if geometry == "rms_to_inf":
        return rms_to_inf_norm(W)
    if geometry == "l1_to_rms":
        return l1_to_rms_norm(W)
    raise AssertionError(f"Unhandled update-radiality geometry: {geometry}")


def _directional_separation(
    W_before: torch.Tensor,
    W_after: torch.Tensor,
    geometry: str,
) -> torch.Tensor:
    """Measure separation between normalized, nonzero iterates."""
    norm_before = _geometry_norm(W_before, geometry)
    norm_after = _geometry_norm(W_after, geometry)
    tiny = torch.finfo(norm_before.dtype).tiny

    direction_before = W_before / norm_before.clamp_min(tiny)
    direction_after = W_after / norm_after.clamp_min(tiny)
    separation = _geometry_norm(direction_after - direction_before, geometry)
    valid = (norm_before > 0) & (norm_after > 0)

    return torch.where(
        valid, separation, torch.full_like(separation, float("nan"))
    )


def _update_radiality_rms_to_rms(
    W: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """RMS -> RMS radiality.

    For W in R^{d_out x d_in},

        N(W) = sqrt(d_in / d_out) * ||W||_op.

    A norming covector is

        D_{N*}(W) = sqrt(d_in / d_out) * u1 v1^T,

    for a leading singular-vector pair (u1, v1).  Since the same dimension
    factor appears in N(U), it cancels in

        <D_{N*}(W), U / N(U)>
          = u1^T U v1 / ||U||_op.

    Returns NaN when ||W||_op = 0 or when ||U||_op is at most
    `_REL_DEGENERACY_EPS` times ||W||_op.
    """
    tiny = torch.finfo(W.dtype).tiny

    Uw, Sw, Vhw = torch.linalg.svd(W, full_matrices=False, driver="gesvd")
    sigma_w = Sw[0]
    u1 = Uw[:, 0]
    v1 = Vhw[0, :]

    sigma_u = torch.linalg.matrix_norm(U, ord=2)
    radial_component = u1 @ (U @ v1)
    radiality = radial_component / sigma_u.clamp_min(tiny)
    valid = (sigma_w > 0) & (sigma_u > _REL_DEGENERACY_EPS * sigma_w)

    return torch.where(valid, radiality, torch.full_like(radiality, float("nan")))


def _update_radiality_rms_to_inf(
    W: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """RMS -> l_infinity radiality.

    The induced norm is

        N(W) = sqrt(d_in) * max_i ||row_i(W)||_2.

    If i* is an index of a largest-L2 row of W, one valid norming covector
    has only row i* non-zero and that row points along row_i*(W).  The
    sqrt(d_in) scale cancels against N(U), giving

        radiality =
            <row_i*(W), row_i*(U)>
            / (||row_i*(W)||_2 * max_j ||row_j(U)||_2).

    At ties, torch.argmax selects the first maximal row; this is one valid
    subgradient selection at that non-smooth point.

    Returns NaN when the largest row norm of W is zero or when the largest
    row norm of U is at most `_REL_DEGENERACY_EPS` times that of W.
    """
    tiny = torch.finfo(W.dtype).tiny

    row_norms_w = torch.linalg.vector_norm(W, ord=2, dim=1)
    row_norms_u = torch.linalg.vector_norm(U, ord=2, dim=1)

    max_row_w, row_idx = torch.max(row_norms_w, dim=0)
    max_row_u = torch.max(row_norms_u)

    row_w = W[row_idx, :]
    row_u = U[row_idx, :]
    radial_component = torch.dot(row_w, row_u)
    denominator = (max_row_w * max_row_u).clamp_min(tiny)
    radiality = radial_component / denominator
    valid = (max_row_w > 0) & (
        max_row_u > _REL_DEGENERACY_EPS * max_row_w
    )

    return torch.where(valid, radiality, torch.full_like(radiality, float("nan")))


def _update_radiality_l1_to_rms(
    W: torch.Tensor,
    U: torch.Tensor,
) -> torch.Tensor:
    """l1 -> RMS radiality.

    l1 -> RMS induced operator norm is

        N(W) = max_j ||col_j(W)||_RMS
             = (1 / sqrt(d_out)) * max_j ||col_j(W)||_2.

    If j* is an index of a largest-L2 column of W, a norming covector for
    the dual geometry concentrates on column j*.  The 1/sqrt(d_out) scale
    cancels against N(U), giving

        radiality =
            <col_j*(W), col_j*(U)>
            / (||col_j*(W)||_2 * max_k ||col_k(U)||_2).

    Note that this norming covector is intentionally different from the
    primal dualization map for a gradient, which independently
    normalizes every gradient column.  Here W is used to select a norming
    covector of the *dual* norm because the target quantity is radial
    alignment of a primal update.

    At ties, torch.argmax selects the first maximal column.

    Returns NaN when the largest column norm of W is zero or when the largest
    column norm of U is at most `_REL_DEGENERACY_EPS` times that of W.
    """
    tiny = torch.finfo(W.dtype).tiny

    col_norms_w = torch.linalg.vector_norm(W, ord=2, dim=0)
    col_norms_u = torch.linalg.vector_norm(U, ord=2, dim=0)

    max_col_w, col_idx = torch.max(col_norms_w, dim=0)
    max_col_u = torch.max(col_norms_u)

    col_w = W[:, col_idx]
    col_u = U[:, col_idx]
    radial_component = torch.dot(col_w, col_u)
    denominator = (max_col_w * max_col_u).clamp_min(tiny)
    radiality = radial_component / denominator
    valid = (max_col_w > 0) & (
        max_col_u > _REL_DEGENERACY_EPS * max_col_w
    )

    return torch.where(valid, radiality, torch.full_like(radiality, float("nan")))


def _calculate_update_radialities(
    W: torch.Tensor,
    U: torch.Tensor,
    geometries: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute radialities from an already-prepared weight and update."""
    result: dict[str, torch.Tensor] = {}
    for geometry in geometries:
        if geometry == "rms_to_rms":
            value = _update_radiality_rms_to_rms(W, U)
        elif geometry == "rms_to_inf":
            value = _update_radiality_rms_to_inf(W, U)
        elif geometry == "l1_to_rms":
            value = _update_radiality_l1_to_rms(W, U)
        else:
            raise AssertionError(f"Unhandled update-radiality geometry: {geometry}")

        result[_UPDATE_RADIALITY_METRIC_BY_GEOMETRY[geometry]] = value

    return result


def _calculate_directional_separations(
    W_before: torch.Tensor,
    W_after: torch.Tensor,
    geometries: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute directional separations for the selected geometries."""
    return {
        _DIRECTIONAL_SEPARATION_METRIC_BY_GEOMETRY[
            geometry
        ]: _directional_separation(W_before, W_after, geometry)
        for geometry in geometries
    }


@torch.no_grad()
def calculate_radial_metrics(
    W_before: torch.Tensor,
    W_after: torch.Tensor,
    state: dict[str, torch.Tensor],
    update_radiality_geometries: Iterable[str] | str | None = None,
) -> dict[str, torch.Tensor]:
    """Return all of `RADIAL_METRIC_NAMES` as a flat dict of 0-d tensors.

    `state` holds the four running Frobenius accumulators (`raw_A2`,
    `angular_A1`, `angular_A2`, `R1`), mutated in place.  This call's
    accumulator outputs reflect Sum_{i<t} (they do NOT include the current
    step), then `state` is updated so the next call sees this step.

    `update_radiality_geometries` controls which 2-D geometry metrics are
    actually computed. This includes both update radiality and directional
    separation. `None` means all supported geometries; pass an
    empty tuple to disable them all. Geometry names must exactly match
    `UPDATE_RADIALITY_GEOMETRIES`. The output schema stays fixed:
    disabled geometries and non-2-D tensors receive NaN sentinels. Update
    radiality is also NaN for zero weights and updates satisfying
    N(U) <= `_REL_DEGENERACY_EPS` * N(W_before); directional separation is
    NaN when either iterate has zero norm.
    """
    if isinstance(W_before, torch.nn.Parameter):
        W_before = W_before.data
    if isinstance(W_after, torch.nn.Parameter):
        W_after = W_after.data
    if W_before.dtype in (torch.float16, torch.bfloat16):
        W_before = W_before.float()
    if W_after.dtype in (torch.float16, torch.bfloat16):
        W_after = W_after.float()

    if W_before.shape != W_after.shape:
        raise ValueError("W_before and W_after must have identical shapes.")

    if W_before.dtype != W_after.dtype:
        common_dtype = torch.promote_types(W_before.dtype, W_after.dtype)
        W_before = W_before.to(dtype=common_dtype)
        W_after = W_after.to(dtype=common_dtype)

    dtype = W_before.dtype
    tiny = torch.finfo(dtype).tiny

    U = W_after - W_before
    r_t = W_before.norm()
    r_next = W_after.norm()
    a_t = U.norm()

    # Relative (not absolute/exact-zero) floor on a_t: an update whose
    # magnitude is numerically negligible compared to the weight's own
    # scale should report the same deterministic degenerate sentinel across
    # parallelism/reduction-order variants.
    valid_wu = (r_t > 0) & (a_t > _REL_DEGENERACY_EPS * r_t)
    valid_ww = (r_t > 0) & (r_next > 0)

    relative_step = a_t / r_t.clamp_min(tiny)

    dot_wu = (W_before * U).sum()
    radial_cosine_raw = (dot_wu / (r_t * a_t).clamp_min(tiny)).clamp(-1.0, 1.0)
    radial_cosine = torch.where(
        valid_wu, radial_cosine_raw, torch.zeros_like(radial_cosine_raw)
    )
    tangent_fraction = (1.0 - radial_cosine * radial_cosine).clamp_min(0.0).sqrt()

    # Canonical angle via atan2, not acos.  This is numerically better for
    # the small angles common in training.
    angle_raw = torch.atan2(a_t * tangent_fraction, r_t + a_t * radial_cosine)
    angle = torch.where(valid_ww, angle_raw, torch.zeros_like(angle_raw))

    # Independent sanity check against `angle` via the direct acos formula.
    dot_ww = (W_before * W_after).sum()
    cos_angle = (dot_ww / (r_t * r_next).clamp_min(tiny)).clamp(-1.0, 1.0)
    angle_from_cos = torch.where(
        valid_ww, torch.arccos(cos_angle), torch.zeros_like(cos_angle)
    )

    # ||W + U||_F^2 - ||W||_F^2 = 2 <W, U>_F + ||U||_F^2.
    radial_first_order = 2.0 * dot_wu
    radial_second_order = a_t * a_t
    radial_ratio_raw = radial_first_order.abs() / radial_second_order.clamp_min(tiny)
    radial_ratio = torch.where(
        valid_wu, radial_ratio_raw, torch.zeros_like(radial_ratio_raw)
    )

    # Keep a fixed output schema.  NaN means "not computed / not applicable"
    # and is deliberately distinct from the valid score 0.
    nan_scalar = torch.full((), float("nan"), device=W_before.device, dtype=dtype)
    update_radiality_metrics: dict[str, torch.Tensor] = {
        metric_name: nan_scalar.clone()
        for metric_name in (
            *_UPDATE_RADIALITY_METRIC_BY_GEOMETRY.values(),
            *_DIRECTIONAL_SEPARATION_METRIC_BY_GEOMETRY.values(),
        )
    }

    selected_geometries = _validate_update_radiality_geometries(
        update_radiality_geometries
    )
    if W_before.ndim == 2 and W_before.numel() > 0 and selected_geometries:
        update_radiality_metrics.update(
            _calculate_update_radialities(
                W_before,
                U,
                selected_geometries,
            )
        )
        update_radiality_metrics.update(
            _calculate_directional_separations(
                W_before,
                W_after,
                selected_geometries,
            )
        )

    raw_A2 = state["raw_A2"].clone()
    angular_A1 = state["angular_A1"].clone()
    angular_A2 = state["angular_A2"].clone()
    R1 = state["R1"].clone()
    E_radial = R1 / (raw_A2 + tiny)

    state["raw_A2"].add_(radial_second_order)
    state["angular_A1"].add_(angle)
    state["angular_A2"].add_(angle * angle)
    state["R1"].add_(radial_first_order)

    return {
        "radius": r_t,
        "raw_step": a_t,
        "relative_step": relative_step,
        "radial_cosine": radial_cosine,
        "tangent_fraction": tangent_fraction,
        "angle": angle,
        "angle_from_cos": angle_from_cos,
        "radial_first_order": radial_first_order,
        "radial_second_order": radial_second_order,
        "radial_ratio": radial_ratio,
        "raw_A2": raw_A2,
        "angular_A1": angular_A1,
        "angular_A2": angular_A2,
        "R1": R1,
        "E_radial": E_radial,
        **update_radiality_metrics,
    }
