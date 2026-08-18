# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Whole-tensor "radial dynamics" metrics -- how a weight's norm and direction
evolve under training. Unlike gram_helper.py's row-wise Gram-matrix
framework (which needs the raw momentum/gradient `V_raw` and is gated
behind `gram_level`), these only need the weight before/after this step's
update (`W_before`, `W_after` -- disco.py passes the same `pseudo_w`
already used as gram's `W_after`) and are all whole-tensor Frobenius-norm
-scale scalars, never a row-wise vector -- cheap enough to always compute
whenever any per-param logging fires at all, independent of `gram_level`
and `norms_to_log`.

Notation:
  W_t    = W_before, the weight before this step's update.
  W_t+1  = W_after, the weight after (disco.py's `pseudo_w` approximation).
  dW_t   = W_after - W_before, the realised displacement.
  r_t    = ||W_t||  (Frobenius norm -- "radius").
  a_t    = ||dW_t|| (Frobenius norm -- "raw step").
  q_t    = W_t / r_t,  v_t = dW_t / a_t  (unit directions).
  c_t    = <q_t, v_t>  ("radial_cosine" -- is the step outward-radial or
           tangential relative to the weight's own direction).

Four running accumulators (`raw_A2`, `angular_A1`, `angular_A2`, `R1`)
persist across steps in the caller-supplied `state` dict, mutated in
place -- disco.py stores the canonical values in
`self.state[p]["radial_state"]` (the same place `momentum_buffer` lives),
so they survive checkpoint save/restore via the optimizer's default
`state_dict()`/`load_state_dict()` with no extra plumbing. `R2(t)` from
the "radial error" formula is exactly the same running sum as `raw_A2` --
one accumulator serves both, so `R2` is not separately stored.
`relative_step` here is the same formula as gram_helper.py's
`U_relative_step_fro` -- expected, not an accidental duplicate: this one
is unconditional, that one is gated behind `gram_level`.

`alpha_fit`/`tau_fit` (fitting the angle-decay power law
`theta_t = C * (t + tau)^-alpha`) is a deliberately deferred follow-up --
it needs bounded/subsampled history storage and periodic (not per-step)
refitting to actually stay cheap at scale, unlike everything here, which
is a genuine O(1)-per-call update.
"""

import torch

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
]

_ACCUMULATOR_NAMES: tuple[str, ...] = ("raw_A2", "angular_A1", "angular_A2", "R1")

# a_t below this fraction of r_t is treated as a degenerate (no real update)
# step -- see the valid_wu comment in calculate_radial_metrics for why a
# relative floor is needed instead of a bare a_t > 0 check.
_REL_DEGENERACY_EPS = 1e-6


def new_radial_state(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    shape: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    """Fresh, zero-initialized accumulator state. `shape=()` (the default)
    for a single tracked tensor; `shape=(num_local_experts,)` for expert
    params, where each expert index needs its own independent accumulators
    (each has its own `W_before`/`W_after` pair) -- index into the result
    per-expert (e.g. `state["raw_A2"][ep_idx]`) when calling
    `calculate_radial_metrics`, which mutates whatever 0-d view it's given
    in place."""
    return {
        name: torch.zeros(shape, device=device, dtype=dtype)
        for name in _ACCUMULATOR_NAMES
    }


@torch.no_grad()
def calculate_radial_metrics(
    W_before: torch.Tensor,
    W_after: torch.Tensor,
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    Returns all of `RADIAL_METRIC_NAMES` as a flat dict of 0-d tensors.

    `state` holds the 4 running accumulators (`raw_A2`, `angular_A1`,
    `angular_A2`, `R1`), mutated in place: this call's `raw_A2`/
    `angular_A1`/`angular_A2`/`R1`/`E_radial` outputs reflect `Sum_{i<t}`
    (i.e. NOT including this step's own contribution -- the correct
    semantics for these "history so far" metrics), then `state` is updated
    afterward so the NEXT call sees this step's contribution included.
    """
    if isinstance(W_before, torch.nn.Parameter):
        W_before = W_before.data
    if isinstance(W_after, torch.nn.Parameter):
        W_after = W_after.data
    if W_before.dtype in (torch.float16, torch.bfloat16):
        W_before = W_before.float()
    if W_after.dtype in (torch.float16, torch.bfloat16):
        W_after = W_after.float()

    dtype = W_before.dtype
    tiny = torch.finfo(dtype).tiny

    U = W_after - W_before
    r_t = W_before.norm()
    r_next = W_after.norm()
    a_t = U.norm()

    # Relative (not absolute/exact-zero) floor on a_t: an update whose
    # magnitude is numerically negligible compared to the weight's own
    # scale (e.g. a near-zero-lr step at the tail of a decay schedule)
    # should report the same deterministic degenerate sentinel regardless
    # of which parallelism strategy computed it. `a_t > 0` alone only
    # catches the literal-zero case -- below this relative threshold, a_t
    # is dominated by ordinary floating-point reduction-order noise (e.g.
    # DDP's all-reduce vs FSDP's all-gather summing gradients in a
    # different order), which radial_cosine/radial_ratio (both divide by
    # a_t or a_t^2) amplify into large, run-to-run-inconsistent swings
    # even though the underlying update is physically negligible.
    valid_wu = (r_t > 0) & (a_t > _REL_DEGENERACY_EPS * r_t)
    valid_ww = (r_t > 0) & (r_next > 0)

    relative_step = a_t / r_t.clamp_min(tiny)

    dot_wu = (W_before * U).sum()
    radial_cosine_raw = (dot_wu / (r_t * a_t).clamp_min(tiny)).clamp(-1.0, 1.0)
    # Explicit degenerate-case sentinel (0, an undefined angle) rather than
    # letting the tiny floor alone produce an arbitrary non-zero value --
    # same convention as _row_normalise/_effective_rank's torch.where
    # guards in gram_helper.py.
    radial_cosine = torch.where(
        valid_wu, radial_cosine_raw, torch.zeros_like(radial_cosine_raw)
    )
    tangent_fraction = (1.0 - radial_cosine * radial_cosine).clamp_min(0.0).sqrt()

    # Canonical angle via atan2, not acos: acos's derivative blows up near
    # cos=1, so small angles (the common case most training steps) lose
    # precision in fp32 -- nearby small angles round to indistinguishable
    # cosine values. atan2 doesn't have this issue. Mathematically the same
    # quantity as arccos(<q_t, q_t+1>) -- verified via the 2D-trigonometry
    # identity: place q_t at (r_t, 0); the step lands W_t+1 at
    # (r_t + a_t*c_t, a_t*sqrt(1-c_t^2)), whose angle from the x-axis is
    # exactly this atan2 expression -- just computed via a more numerically
    # robust path. Gated on `valid_ww` (matching angle_from_cos below), not
    # `valid_wu` -- deliberately NOT the relative-degeneracy floor above:
    # unlike radial_cosine/radial_ratio, atan2's inputs (a_t*tangent_fraction,
    # r_t + a_t*radial_cosine) stay well-conditioned as a_t -> 0 (numerator
    # -> 0, denominator -> r_t > 0), so `angle` doesn't inherit the
    # near-zero-a_t noise-amplification these other metrics have, and
    # doesn't need the same protection.
    angle_raw = torch.atan2(a_t * tangent_fraction, r_t + a_t * radial_cosine)
    angle = torch.where(valid_ww, angle_raw, torch.zeros_like(angle_raw))

    # Independent sanity check against `angle` above (same quantity, via
    # the acos formula instead of atan2) -- not fed into the accumulators,
    # kept purely to catch a geometry/implementation bug if it ever
    # meaningfully diverges from `angle`.
    dot_ww = (W_before * W_after).sum()
    cos_angle = (dot_ww / (r_t * r_next).clamp_min(tiny)).clamp(-1.0, 1.0)
    angle_from_cos = torch.where(
        valid_ww, torch.arccos(cos_angle), torch.zeros_like(cos_angle)
    )

    # Algebraically == 2*a_t*r_t*radial_cosine (c_t = dot_wu/(r_t*a_t)),
    # but computed directly from the dot product already at hand: avoids a
    # divide-then-remultiply round trip, and stays well-defined even when
    # r_t or a_t is exactly 0 (dot_wu is 0 there too, no separate
    # degenerate-case guard needed, unlike radial_cosine).
    radial_first_order = 2.0 * dot_wu
    radial_second_order = a_t * a_t
    # Away from the degenerate regime, radial_ratio is deliberately NOT
    # given a relative-floor treatment the way genuinely-unbounded ratios
    # elsewhere in gram_helper.py are: it's *supposed* to swing large or
    # small depending on which regime dominates (that's its whole
    # diagnostic purpose). But when the step itself is degenerate (a_t
    # negligible vs r_t -- see valid_wu above), both radial_first_order and
    # radial_second_order are individually noise-dominated, and dividing
    # noise by noise-squared is pure amplification, not signal -- gate it
    # to the same deterministic 0 sentinel as radial_cosine so it doesn't
    # report large, run-to-run-inconsistent swings on a step where nothing
    # meaningful happened.
    radial_ratio_raw = radial_first_order.abs() / radial_second_order.clamp_min(tiny)
    radial_ratio = torch.where(
        valid_wu, radial_ratio_raw, torch.zeros_like(radial_ratio_raw)
    )

    raw_A2 = state["raw_A2"].clone()
    angular_A1 = state["angular_A1"].clone()
    angular_A2 = state["angular_A2"].clone()
    R1 = state["R1"].clone()
    # Same "meant to swing large" reasoning as radial_ratio -- additive
    # +tiny (not a relative floor) is intentional.
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
    }
