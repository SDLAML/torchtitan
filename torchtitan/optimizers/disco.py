# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from enum import Enum

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Replicate, Shard

from torch.profiler import record_function  # labels in PyTorch profiler

from torchtitan.tools.logging import logger

from . import gram_helper
from .abstract_disco import AbstractDiSCO
from .gram_helper import calculate_gram_metrics
from .norm_helper import calculate_norm
from .pre_norm_helper import (
    pre_norm_category,
    PRE_NORM_FULL_FUNCTIONS,
    PRE_NORM_PARTIAL_FUNCTIONS,
    PRE_NORM_ROW_FUNCTIONS,
    PRE_NORM_SHARDED_APPLY_FUNCTIONS,
)
from .radial_helper import (
    calculate_aus_correction,
    calculate_radial_metrics,
    new_radial_state,
    RADIAL_METRIC_NAMES,
    resolve_aus_geometry,
)
from .utils import remove_orig_mod_and_weight_for_p_name

__all__ = [
    "DiSCO",
]


# these variables are hardcoded for now, taken from torchtitan
CONST_NAME_OF_EMBEDDING = "tok_embeddings"

# Environment variables and default values
# DISCO_DEBUG_MODE = "0"
# DISCO_ENABLE_PERSISTENT_CACHE = "1"
# DISCO_FSDP_A2A_MODE = "once"
# DISCO_TRACK_EMBED_GRAM = "1"


class ParamType(Enum):
    DDP = 0
    FSDP = 1
    Expert = 2


def get_param_type(p, fsdp_enabled, expert_enabled):
    """Classify parameter bucket type from static structure."""
    if p.numel() == 1:
        # treat scalars separately in _build_param_lists(); return DDP here by default
        return ParamType.DDP
    if p.ndim == 3 and (expert_enabled or fsdp_enabled):
        return ParamType.Expert
    if fsdp_enabled:
        return ParamType.FSDP
    return ParamType.DDP


def tp_axis(placements: tuple, tp_enabled: bool = False) -> int | None:
    """
    Return the index in `placements` that belongs to *tensor-parallel* (TP).

    Heuristics (PyTorch-TP default layouts):
      1. Row-parallel weights ⇒ `_StridedShard`  ⟶ that axis is TP.
      2. Col-parallel weights ⇒ `Shard(dim != 0)` ⟶ that axis is TP
         (FSDP shards dim-0, so a non-zero dim means TP).
    """
    # rule 1 – row-parallel
    for i, p in enumerate(placements):
        if isinstance(p, _StridedShard):
            return i

    # rule 2 – col-parallel
    for i, p in enumerate(placements):
        if isinstance(p, Shard) and p.dim != 0:
            return i

    # this is a special case, We do TP only
    if tp_enabled and len(placements) == 1:
        if isinstance(placements[0], Shard):
            return 0
    return None  # could not infer


def gather_tp_shard(tensor, tp_mesh, tp_world_size, original_placements):
    # TP is used, we need to gather the TP-shard params first
    tp_mesh_dim = tp_axis(original_placements, True)
    assert tp_mesh_dim is not None, "TP mesh dimension not found"
    shard_dim = original_placements[tp_mesh_dim].dim

    # output_tensors = [torch.empty_like(tensor) for _ in range(tp_world_size)]
    # dist.all_gather(output_tensors, tensor, group=tp_group)
    # return torch.cat(output_tensors, dim=shard_dim)
    return funcol.all_gather_tensor(tensor, gather_dim=shard_dim, group=tp_mesh)


def _is_fsdp_row_sharded(p) -> bool:
    """
    True iff `p` is a DTensor genuinely row-sharded (Shard(0)) along the
    "fsdp" mesh dimension -- i.e. a local view of `p` only holds *some* rows,
    not the full matrix. `step_fsdp`'s own params are always this way by
    construction; `step_embedding`'s params are routed by a config check
    (backend=="identity"), independent of physical sharding, so this needs a
    per-param structural check.
    """
    if not isinstance(p, DTensor):
        return False
    mesh_dim_names = p.device_mesh.mesh_dim_names
    if not mesh_dim_names or "fsdp" not in mesh_dim_names:
        return False
    placement = p.placements[mesh_dim_names.index("fsdp")]
    return isinstance(placement, Shard) and placement.dim == 0


def _pseudo_post_update_weight(w, u, lr, wd):
    """Cheap elementwise replica of the real apply formula (see
    _update_embed_params_fast / _update_ddp_params_fast /
    _update_expert_params_fast / update_bucket_params: `w = w*(1-wd*lr) -
    lr*u`), used ONLY so that track_param_* norms keep their historical
    post-update meaning once `w` becomes pre-update in-scope for gram
    metrics. Log-only approximation -- not the real applied tensor (which
    may go through an extra communication-dtype cast), so no vectorized
    optimization is needed; the SVD in calculate_norm dominates regardless.
    """
    pseudo_w = w * (1.0 - wd * lr) if wd != 0.0 else w
    return pseudo_w - lr * u


def _pack_segments(
    segments: list[tuple[str, torch.Tensor | None]],
) -> tuple[torch.Tensor, dict[str, int]]:
    """Concatenate present (non-None) segments into one flat buffer for a
    single collective, instead of one all_gather per segment. Returns the
    buffer and a {name: offset} map for slicing it back apart after
    gathering -- offsets are within *one rank's* contribution; the caller
    adds `rank * per_rank_total` on top once gathered.

    Deriving offsets here (from what was actually concatenated, in the same
    call) rather than hand-computing them at each call site is deliberate:
    a hand-derived offset can silently drift out of sync with pack order if
    a segment is added/removed/reordered later, whereas this makes that
    class of bug structurally impossible.
    """
    offsets: dict[str, int] = {}
    parts: list[torch.Tensor] = []
    running = 0
    for name, tensor in segments:
        if tensor is None:
            continue
        offsets[name] = running
        parts.append(tensor)
        running += tensor.numel()
    return torch.cat(parts), offsets


def calculate_shard_shape(shape, rank, world_size):
    full = shape[0]
    splits = torch.arange(full).chunk(world_size)
    if rank >= len(splits):
        dim0 = 0
    else:
        dim0 = len(splits[rank])

    return (dim0, *shape[1:])


def _gram_log_param_name(cleaned_name: str, shape: tuple[int, ...]) -> str:
    """Mark parameter names whose Gram metrics use the transposed orientation."""
    return (
        f"{cleaned_name}.T"
        if gram_helper.gram_matrix_is_transposed(shape)
        else cleaned_name
    )


def parse_env_var():
    debug_mode = os.environ.get("DISCO_DEBUG_MODE", "0") == "1"
    persistent_cache_enabled = (
        os.environ.get("DISCO_ENABLE_PERSISTENT_CACHE", "1") == "1"
    )

    a2a_mode = os.environ.get("DISCO_FSDP_A2A_MODE", "once").strip().lower()
    if a2a_mode not in {"once", "bucket"}:
        raise ValueError(
            f"Unknown DISCO_FSDP_A2A_MODE={a2a_mode}. Supported: once, bucket"
        )

    # tok_embeddings/output's gram metrics are far more expensive than any
    # other tracked param (their full [vocab_size, hidden_dim] matrices --
    # tens of GB and several seconds of fp32 GEMM at large vocab sizes --
    # dwarf every other layer's gram cost, even though the resulting Gram
    # matrix itself is small; see readme.md) -- a separate switch from
    # `gram_level` so it can be turned off independently (e.g. gram_level>0
    # for cheap layers every step, embed/output gram only at sparse
    # checkpoints) without disabling gram tracking everywhere else.
    track_embed_gram = os.environ.get("DISCO_TRACK_EMBED_GRAM", "1") == "1"

    env_vars = {
        "debug_mode": debug_mode,
        "persistent_cache_enabled": persistent_cache_enabled,
        "a2a_mode": a2a_mode,
        "track_embed_gram": track_embed_gram,
    }
    return env_vars


class DiSCO(AbstractDiSCO):
    def __init__(
        self,
        params,
        is_light,
        weight_decay,
        lr,
        momentum,
        nesterov,
        eps,
        norm_factor,
        backend,
        backend_steps,
        parallel_dims,
        pre_norm="identity",
        communication_dtype=torch.bfloat16,
        extra_reduce_for_HSDP=False,
        experts_weights_layout="G-D_out-D_in",
        aus_enabled=False,
        aus_coefficient=0.5,
        aus_alpha=0.5,
    ):
        env_vars = parse_env_var()
        logger.info(f"[DiSCO] Environment variables: {env_vars}")
        debug_mode = env_vars["debug_mode"]
        self.persistent_cache_enabled = env_vars["persistent_cache_enabled"]
        self.fsdp_a2a_mode = env_vars["a2a_mode"]
        # Plain attribute (like persistent_cache_enabled/fsdp_a2a_mode above),
        # not a dedicated setter -- flip it directly at runtime
        # (`optimizer.track_embed_gram = False`) if you want to turn embed/
        # output gram off for some steps without touching gram_level.
        self.track_embed_gram = env_vars["track_embed_gram"]

        # Initialize base optimizer and common state
        self.log_parameters_types = True
        self.is_light = is_light

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eps=eps,
            norm_factor=norm_factor if not debug_mode else "none",
            pre_norm=pre_norm,
            aus_enabled=aus_enabled,
            aus_coefficient=aus_coefficient,
            aus_alpha=aus_alpha,
            backend=backend if not debug_mode else "identity",
            backend_steps=backend_steps,
            splits_into=None,  # should be explicitly set in the extra_param_group_split_rules
            splits_dim=None,  # should be explicitly set in the extra_param_group_split_rules
        )
        assert is_light is False, "is_light must be False"

        is_unconstrained = weight_decay == 0

        self.parallel_dims = parallel_dims

        self.fsdp_enabled = parallel_dims.fsdp_enabled
        self.expert_enabled = parallel_dims.ep_enabled
        self.dp_replicate_enabled = parallel_dims.dp_replicate_enabled
        self.tp_enabled = parallel_dims.tp_enabled

        # this is used to ensure only the DP or FSDP rank 0 will have norms
        if self.dp_replicate_enabled or self.fsdp_enabled:
            self.is_dp_rank_0 = (
                parallel_dims.get_optional_mesh("loss").get_local_rank() == 0
            )
        else:
            # only PP (and/or) TP enabled
            self.is_dp_rank_0 = dist.get_rank() == 0

        assert experts_weights_layout in [
            "G-D_in-D_out",
            "G-D_out-D_in",
        ], f"Unknown experts weights layout: {experts_weights_layout}"
        self.experts_need_transpose = experts_weights_layout == "G-D_in-D_out"
        self.extra_reduce_for_HSDP = extra_reduce_for_HSDP

        logger.info(
            f"[DiSCO] "
            f"(is_light={self.is_light}, is_unconstrained={is_unconstrained}) "
            f"is enabled with world_mesh={self.parallel_dims.world_mesh} | fsdp_enabled={self.fsdp_enabled} | "
            f"EP={self.expert_enabled} | TP={self.tp_enabled} | DP={self.dp_replicate_enabled}"
        )

        super().__init__(params, defaults, is_light=is_light)

        if debug_mode:
            # Force all groups to identity/none in debug mode. This must mutate
            # param_groups directly because step() refreshes groups_info from them.
            for group in self.param_groups:
                group["norm_factor"] = "none"
                group["backend"] = "identity"

        self.aus_enabled = any(group["aus_enabled"] for group in self.param_groups)
        self._validate_aus_configuration()
        if self.aus_enabled and communication_dtype != torch.float32:
            logger.warning(
                "[DiSCO][AUS] Overriding communication_dtype to float32 for "
                "the DDP research prototype."
            )
            communication_dtype = torch.float32

        self.communication_dtype = communication_dtype
        self.groups_info = {}
        self.groups_pre_norm: dict[int, str] = {}
        self.groups_pre_norm_eps: dict[int, float] = {}
        self.parameters_to_groups = {}
        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"],
                "zeropower_backend": group["backend"],
                "backend_steps": group["backend_steps"],
                "splits_into": group["splits_into"],
                "splits_dim": group["splits_dim"],
            }
            self.groups_info[group_idx] = [lr, nesterov, momentum, wd, param_kwargs]
            self.groups_pre_norm[group_idx] = group.get("pre_norm", "identity")
            self.groups_pre_norm_eps[group_idx] = group["eps"]
            for param in group["params"]:
                self.parameters_to_groups[id(param)] = group_idx

            logger.info(
                f"[DiSCO] group_idx: {group_idx} have {len(group['params'])} "
                f"weights || lr: {lr} || nesterov: {nesterov} || "
                f"momentum: {momentum} || wd: {wd} || {param_kwargs}"
            )

        # public caches
        self.scale_params, self.scale_param_names = [], []
        self.embed_params, self.embed_param_names = [], []
        self.ddp_params, self.ddp_param_names = [], []
        self.fsdp_params, self.fsdp_param_names = [], []
        self.expert_params, self.expert_param_names = [], []
        # build once now
        self._build_param_lists()

    def _validate_aus_configuration(self) -> None:
        if not self.aus_enabled:
            return

        unsupported_parallelism = []
        if self.fsdp_enabled:
            unsupported_parallelism.append("FSDP")
        if self.tp_enabled:
            unsupported_parallelism.append("TP")
        if self.expert_enabled:
            unsupported_parallelism.append("expert/MoE")
        if unsupported_parallelism:
            raise NotImplementedError(
                "The AUS research prototype supports replicated dense DDP only; "
                f"disable {', '.join(unsupported_parallelism)}."
            )

        invalid_groups: list[str] = []
        for group_idx, group in enumerate(self.param_groups):
            if not group["aus_enabled"]:
                continue
            unsupported_ranks = sorted(
                {p.ndim for p in group["params"] if p.ndim > 2}
            )
            if unsupported_ranks:
                invalid_groups.append(
                    f"group {group_idx}: tensor ranks {unsupported_ranks} "
                    "are not supported"
                )
                continue
            if group["splits_into"] is not None or group["splits_dim"] is not None:
                invalid_groups.append(
                    f"group {group_idx}: split logical matrices are not supported"
                )
                continue
            try:
                resolve_aus_geometry(group["norm_factor"])
            except ValueError as exc:
                # Vector/scalar-only groups use the nominal AUS directly and do
                # not require a matrix geometry.
                if any(p.ndim == 2 for p in group["params"]):
                    invalid_groups.append(f"group {group_idx}: {exc}")

        if invalid_groups:
            raise ValueError("Invalid AUS configuration: " + "; ".join(invalid_groups))

    def _build_aus_displacement(
        self,
        p: torch.Tensor,
        u: torch.Tensor | None,
        group_idx: int,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build delta = eta * (U + weight_decay * W) for one parameter."""
        group = self.param_groups[group_idx]
        w = self._get_param_local_view(p)
        nominal_aus = torch.as_tensor(
            group["lr"], dtype=torch.float32, device=w.device
        )
        correction = torch.ones_like(nominal_aus)
        valid = torch.zeros_like(nominal_aus, dtype=torch.bool)
        finite = torch.isfinite(w).all()
        if u is not None:
            finite = finite & torch.isfinite(u).all()

        if u is not None and w.ndim == 2:
            geometry, transpose = resolve_aus_geometry(group["norm_factor"])
            result = calculate_aus_correction(
                w,
                u,
                geometry,
                transpose=transpose,
            )
            correction = result["correction"]
            valid = result["valid"]
            finite = result["finite"]

        eta = nominal_aus * correction
        finite = finite & torch.isfinite(eta)
        if u is None:
            if group["weight_decay"] == 0.0:
                return None, correction, eta, valid, finite
            displacement = w.detach().to(torch.float32) * (
                eta * group["weight_decay"]
            )
        else:
            displacement = u.detach().to(torch.float32) * eta
            if group["weight_decay"] != 0.0:
                displacement.add_(
                    w.detach().to(torch.float32)
                    * (eta * group["weight_decay"])
                )

        finite = finite & torch.isfinite(displacement).all()
        return displacement, correction, eta, valid, finite

    def _raise_if_any_aus_nonfinite(
        self,
        finiteness: list[torch.Tensor | None],
        *,
        device: torch.device,
    ) -> None:
        nonfinite_flags = [
            torch.logical_not(finite).to(torch.int32)
            for finite in finiteness
            if finite is not None
        ]
        any_nonfinite = (
            torch.stack(nonfinite_flags).amax()
            if nonfinite_flags
            else torch.zeros((), dtype=torch.int32, device=device)
        )
        dp_mesh = (
            self.parallel_dims.get_optional_mesh("dp_replicate")
            if self.dp_replicate_enabled
            else None
        )
        if dp_mesh is not None and dp_mesh.size() > 1:
            dist.all_reduce(any_nonfinite, op=dist.ReduceOp.MAX, group=dp_mesh.get_group())
        if bool(any_nonfinite.item()):
            raise FloatingPointError(
                "[DiSCO][AUS] non-finite input or correction calculation detected; "
                "no parameter weights were updated. Momentum buffers may have advanced."
            )

    def _ensure_default_param_state(self):
        """
        Lazy-init the per-param `self.state[p]` keys every trainable param
        needs (`momentum_buffer`, `radial_state`), backfilling whichever one
        is missing. Called both from `_build_param_lists` (fresh/first init)
        and from `load_state_dict` (post-restore) -- the latter matters for
        any load path that doesn't go through DCP's strict key-matching
        (which already fails a checkpoint missing a key outright before this
        would ever run), e.g. a direct/manual `load_state_dict()` call with
        a hand-built or partial state dict.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                # Radial-dynamics accumulator state (raw_A2/angular_A1/
                # angular_A2/R1 -- see radial_helper.py), stored the same
                # way as momentum_buffer so it's automatically checkpoint
                # -persistent via the default state_dict()/load_state_dict().
                # 3-D (expert) params get one accumulator set PER expert
                # index, since each expert has its own W_before/W_after.
                if "radial_state" not in self.state[p]:
                    accum_shape = (p.shape[0],) if p.ndim == 3 else ()
                    self.state[p]["radial_state"] = new_radial_state(
                        p.device, shape=accum_shape
                    )

    def _build_param_lists(self):
        self._ensure_default_param_state()

        # clear
        self.scale_params.clear()
        self.scale_param_names.clear()
        self.embed_params.clear()
        self.embed_param_names.clear()
        self.ddp_params.clear()
        self.ddp_param_names.clear()
        self.fsdp_params.clear()
        self.fsdp_param_names.clear()
        self.expert_params.clear()
        self.expert_param_names.clear()

        # decide embedding per-group exactly like in step()
        # (backend == "identity" and norm_factor startswith embed/unembed)
        def _is_embed_group(g):
            nf, be = g["norm_factor"], g["backend"]
            return (be == "identity") and (
                nf.startswith("embed") or nf.startswith("unembed")
            )

        for group in self.param_groups:
            route_to_embed = _is_embed_group(group)
            for p_name, p in zip(group["param_names"], group["params"]):
                if not p.requires_grad:
                    # ignore the non-trainable parameters
                    continue

                # 1) scalar branch identical to step()
                if p.numel() == 1:
                    assert (
                        group["backend"] == "identity"
                    ), "scale params must use identity backend"
                    assert (
                        group["norm_factor"] == "sign"
                    ), "scale params must use sign norm factor"
                    assert group.get("pre_norm", "identity") == "identity", (
                        "scale params must use identity pre_norm -- row/col/mat "
                        "aren't well-defined for a 1-element tensor"
                    )
                    self.scale_params.append(p)
                    self.scale_param_names.append(p_name)
                    continue
                # 2) embedding fast path identical to step() predicate
                if route_to_embed:
                    self.embed_params.append(p)
                    self.embed_param_names.append(p_name)
                    continue
                # 3) structural type without reading p.grad (init-time)
                ptype = get_param_type(p, self.fsdp_enabled, self.expert_enabled)
                if ptype == ParamType.DDP:
                    self.ddp_params.append(p)
                    self.ddp_param_names.append(p_name)
                elif ptype == ParamType.FSDP:
                    self.fsdp_params.append(p)
                    self.fsdp_param_names.append(p_name)
                elif ptype == ParamType.Expert:
                    self.expert_params.append(p)
                    self.expert_param_names.append(p_name)
                else:
                    # static classifier should not return Unknown
                    pass
        if self.ddp_params:
            pairs = list(zip(self.ddp_params, self.ddp_param_names))
            # sort big → small to reduce padding and make buckets well-conditioned
            pairs.sort(key=lambda x: x[0].numel(), reverse=True)

            # snake interleave across buckets to balance per-rank Phase-A compute
            dp_replicate_mesh = (
                self.parallel_dims.get_optional_mesh("dp_replicate")
                if self.dp_replicate_enabled
                else None
            )
            w = dp_replicate_mesh.size() if dp_replicate_mesh is not None else 1
            if w > 1:
                blocks = [pairs[i : i + w] for i in range(0, len(pairs), w)]
                for b, blk in enumerate(blocks):
                    if b % 2 == 1:
                        blk.reverse()
                pairs = [p for blk in blocks for p in blk]

            self.ddp_params, self.ddp_param_names = (
                (list(t) if pairs else [] for t in zip(*pairs)) if pairs else ([], [])
            )

        if self.fsdp_params:
            pairs = list(zip(self.fsdp_params, self.fsdp_param_names))
            pairs.sort(key=lambda x: x[0].numel(), reverse=True)
            self.fsdp_params, self.fsdp_param_names = list(zip(*pairs))
            self.fsdp_params, self.fsdp_param_names = (
                list(self.fsdp_params),
                list(self.fsdp_param_names),
            )

        if self.expert_params:
            pairs = list(zip(self.expert_params, self.expert_param_names))
            pairs.sort(key=lambda x: (x[0].numel(), x[0].shape[1]), reverse=True)
            self.expert_params, self.expert_param_names = list(zip(*pairs))
            self.expert_params, self.expert_param_names = (
                list(self.expert_params),
                list(self.expert_param_names),
            )

        if self.embed_params:
            pairs = list(zip(self.embed_params, self.embed_param_names))
            pairs.sort(key=lambda x: x[0].numel(), reverse=True)
            self.embed_params, self.embed_param_names = map(list, zip(*pairs))

        if self.log_parameters_types:
            logger.info(
                f"[DiSCO] fsdp_params: {len(self.fsdp_params)} | expert_params: {len(self.expert_params)} | "
                f"ddp_params: {len(self.ddp_params)} | embed_params: {len(self.embed_params)} | "
                f"scale_params: {len(self.scale_params)}"
            )
            self.log_parameters_types = False

        # Pre-compute structural metadata so hot paths do zero redundant work per step.
        self._precompute_runtime_caches()
        self._precompute_update_slicing()
        self._precompute_fsdp_metadata()
        self._precompute_experts_metadata()
        self._precompute_ddp_metadata()
        self._precompute_embed_metadata()
        self._precompute_momentum_bufs()
        self._precompute_pre_norm_metadata()
        self._precompute_fsdp_gram_vector_metadata()
        self._precompute_experts_gram_vector_metadata()
        self._precompute_ddp_gram_vector_metadata()
        self._gram_level_at_last_vector_precompute = self.gram_level

    # Param-group keys that come from the run's config (norm_helper/gram_helper
    # setup), not from optimizer state -- see load_state_dict below.
    _CONFIG_ONLY_GROUP_KEYS = (
        "eps",
        "norm_factor",
        "backend",
        "backend_steps",
        "splits_into",
        "splits_dim",
        "pre_norm",
        "aus_enabled",
        "aus_coefficient",
        "aus_alpha",
    )

    def load_state_dict(self, state_dict):
        """
        Two independent fixes over the plain torch.optim.Optimizer.load_state_dict:

        1. Config-vs-checkpoint precedence: torch's base load_state_dict
           overwrites every param_group key (other than "params") with
           whatever was saved in the checkpoint, including
           _CONFIG_ONLY_GROUP_KEYS -- values that come from this run's config
           (norm_factor/backend/eps/etc.), not from training. If the user
           changes one of those between runs (e.g. switching backend or
           norm_factor) and then resumes, the checkpoint would silently
           revert it. Snapshot them before the base call and restore them
           after, warning on any mismatch so a deliberate config change is
           visible rather than silently discarded.

        2. Fast-lookup cache refresh: verified (see radial_helper/disco.py
           session notes) that in TorchTitan's actual dcp.load() resume path,
           OptimizersContainer.state_dict() returns self.state[p]'s tensors
           by reference (torch.optim.Optimizer.state_dict() never clones),
           and DCP fills them in place before load_state_dict ever runs --
           so _momentum_buffer_by_param_id/_radial_state_by_param_id stay
           valid without any rebuild in that path. This refresh is
           defense-in-depth for any load path that bypasses DCP's in-place
           fill (e.g. a direct/manual load_state_dict call with a
           hand-built or detached state dict) -- cheap, and
           _ensure_default_param_state's lazy-init guards make it safe even
           if such a dict is missing a key.

           Note this does NOT help an old checkpoint (saved before
           radial_state existed) resume through the normal dcp.load() path:
           DCP's default LoadPlanner has allow_partial_load=False, so it
           raises "Missing key in checkpoint state_dict: ...radial_state"
           during its own planning phase, before load_state_dict (this
           method included) ever runs. Loading model-only (optimizer state
           excluded entirely, e.g. --checkpoint.initial_load_in_hf /
           initial_load_model_only) sidesteps that, at the cost of ALL
           optimizer state (fresh momentum too, not just radial_state) --
           there is no way to keep momentum while dropping only radial_state
           short of relaxing allow_partial_load checkpoint-wide.
        """
        optional_config_defaults = {
            "pre_norm": "identity",
            # Older checkpoints may omit AUS settings. The current run's
            # values are restored below.
            "aus_enabled": False,
            "aus_coefficient": 0.5,
            "aus_alpha": 0.5,
        }

        def config_group_value(group, key):
            if key in optional_config_defaults:
                return group.get(key, optional_config_defaults[key])
            return group[key]

        pre_load = [
            {
                k: config_group_value(group, k)
                for k in self._CONFIG_ONLY_GROUP_KEYS
            }
            for group in self.param_groups
        ]

        super().load_state_dict(state_dict)

        for group_idx, (group, config_kwargs) in enumerate(
            zip(self.param_groups, pre_load)
        ):
            for key, config_val in config_kwargs.items():
                checkpoint_val = config_group_value(group, key)
                if checkpoint_val != config_val:
                    logger.warning(
                        f"[DiSCO] group_idx {group_idx}: checkpoint's "
                        f"'{key}'={checkpoint_val!r} differs from config's "
                        f"{key}={config_val!r}; keeping config value."
                    )
                group[key] = config_val

        self._ensure_default_param_state()
        self._precompute_runtime_caches()
        self._precompute_momentum_bufs()

    # ------------------------------------------------------------------
    # Pre-compute helpers (called from _build_param_lists, at __init__ and
    # again after every load_state_dict -- see the override above)
    # ------------------------------------------------------------------

    def _precompute_runtime_caches(self):
        """
        Build direct per-parameter runtime caches to avoid repeated dict lookups and
        repeated DTensor -> local view construction in hot paths.
        """
        self._param_local_views: dict[int, torch.Tensor] = {}
        self._momentum_buffer_by_param_id: dict[int, torch.Tensor] = {}
        self._radial_state_by_param_id: dict[int, dict[str, torch.Tensor]] = {}
        self._zero_scalar: torch.Tensor | None = None
        self._padding_norms: dict[str, torch.Tensor] | None = None

        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                if isinstance(p, DTensor):
                    self._param_local_views[pid] = p.to_local()
                else:
                    self._param_local_views[pid] = p
                if p.requires_grad:
                    self._momentum_buffer_by_param_id[pid] = self.state[p][
                        "momentum_buffer"
                    ]
                    self._radial_state_by_param_id[pid] = self.state[p]["radial_state"]

    def _precompute_update_slicing(self):
        """
        Pre-compute TP slicing info for every parameter so update_bucket_params
        avoids isinstance / placements / tp_axis checks per param each step.

        Stores: self._tp_slice_info[id(p)] = None  (no slicing needed)
                                             or (shard_dim, chunk_size, start, slicer_tuple,
                                                 local_shape)
        where local_shape is the expected shape of p.to_local(), used to decide
        whether slicing is needed without calling p.to_local() per step.
        """
        self._tp_slice_info: dict[int, tuple | None] = {}
        if not self.tp_enabled:
            return
        tp_mesh = self.parallel_dims.get_optional_mesh("tp")
        tp_rank = tp_mesh.get_local_rank()
        for group in self.param_groups:
            for p in group["params"]:
                if not isinstance(p, DTensor):
                    self._tp_slice_info[id(p)] = None
                    continue
                placements = p.placements
                tp_dim = tp_axis(placements, tp_enabled=True)
                if tp_dim is None:
                    self._tp_slice_info[id(p)] = None
                else:
                    shard_dim = placements[tp_dim].dim
                    p_local = self._get_param_local_view(p)
                    local_shape = p_local.shape
                    chunk_size = local_shape[shard_dim]
                    start = tp_rank * chunk_size
                    slicer: list = [slice(None)] * p.dim()
                    slicer[shard_dim] = slice(start, start + chunk_size)
                    self._tp_slice_info[id(p)] = (
                        shard_dim,
                        chunk_size,
                        start,
                        tuple(slicer),
                        local_shape,
                    )

    def _get_cached_zero_scalar(self, device):
        if self._zero_scalar is None:
            self._zero_scalar = torch.zeros((), device=device)
        return self._zero_scalar

    def _get_cached_padding_norms(self, device):
        if self._padding_norms is None:
            z = self._get_cached_zero_scalar(device)
            self._padding_norms = {k: z for k in self.norms_to_log}
        return self._padding_norms

    def _get_param_local_view(self, p):
        """
        Return cached local tensor view for `p`. Refresh lazily only if missing.
        """
        pid = id(p)
        p_local = self._param_local_views.get(pid)
        if p_local is None:
            p_local = p.to_local() if isinstance(p, DTensor) else p
            self._param_local_views[pid] = p_local
        return p_local

    def _precompute_embed_metadata(self):
        """
        Pre-compute embed extra-param groups, shape batching metadata, and update plan.
        CPU-only; called once from _build_param_lists at init.
        """
        self._embed_extra_indices: list[int] = []
        self._embed_extra_by_group: list[tuple[int, list[int], list]] = []
        self._embed_extra_shape_groups: dict[tuple, list[int]] = {}
        self._embed_shape_group_gidx: dict[tuple, int] = {}
        self._embed_shape_group_locals: dict[tuple, list[torch.Tensor]] = {}
        self._embed_update_plan: list[
            tuple[
                int,
                torch.device,
                torch.dtype,
                list[int],
                list[torch.Tensor],
                list[torch.Tensor],
            ]
        ] = []
        self._embed_step_workspace_cache: dict | None = None

        if not self.embed_params:
            return

        # Build update apply plan for ALL embed params (grouped by group/device/dtype).
        # Same structure as _expert_update_plan / _ddp_update_plan.
        # Fast-path extras (big_us_by_shape) skip updates[] and are applied separately,
        # so their updates[i] stays None — they are automatically skipped here.
        update_buckets: dict[tuple, dict] = {}
        for param_idx, p in enumerate(self.embed_params):
            group_idx = self.parameters_to_groups[id(p)]
            p_local = self._get_param_local_view(p)
            key = (group_idx, p_local.device, p_local.dtype)
            if key not in update_buckets:
                update_buckets[key] = {"indices": [], "params": [], "locals": []}
            b = update_buckets[key]
            b["indices"].append(param_idx)
            b["params"].append(p)
            b["locals"].append(p_local)
        self._embed_update_plan = [
            (
                group_idx,
                device,
                dtype,
                data["indices"],
                data["params"],
                data["locals"],
            )
            for (group_idx, device, dtype), data in update_buckets.items()
        ]

        if len(self.embed_params) <= 2:
            return

        # Build extra indices (sorted by numel already from embed_params sort above)
        for i in range(2, len(self.embed_params)):
            self._embed_extra_indices.append(i)

        # Group extras by group_idx for batched get_momentum_or_grad_list
        by_group: dict[int, dict] = {}
        for i in self._embed_extra_indices:
            p = self.embed_params[i]
            gidx = self.parameters_to_groups[id(p)]
            if gidx not in by_group:
                by_group[gidx] = {"indices": [], "params": []}
            by_group[gidx]["indices"].append(i)
            by_group[gidx]["params"].append(p)
        self._embed_extra_by_group = [
            (gidx, d["indices"], d["params"]) for gidx, d in by_group.items()
        ]

        # Group by local shape for batched LMO (fill big_g → single lmo() per group)
        for i in self._embed_extra_indices:
            p_local = self._get_param_local_view(self.embed_params[i])
            shape = tuple(p_local.shape)
            self._embed_extra_shape_groups.setdefault(shape, []).append(i)

        # Per shape group: pre-compute group_idx and param local views for fast apply.
        for shape, indices in self._embed_extra_shape_groups.items():
            gidx0 = self.parameters_to_groups[id(self.embed_params[indices[0]])]
            self._embed_shape_group_gidx[shape] = gidx0
            self._embed_shape_group_locals[shape] = [
                self._get_param_local_view(self.embed_params[i]) for i in indices
            ]

    def _precompute_fsdp_metadata(self):
        """
        Pre-compute structural FSDP bucket metadata (shapes/splits/offsets) so
        step_fsdp can switch between:
          - global 2-call all_to_all_single
          - bucketed all_to_all_single fallback
        without recomputing layout every step.
        """
        self._fsdp_total_buckets = 0
        self._fsdp_bucket_ranges: list[tuple[int, int]] = []
        self._fsdp_target_shapes: list[tuple] = []
        self._fsdp_recv_shapes: list[list[tuple]] = []
        self._fsdp_split_rows: list[list[int]] = []
        self._fsdp_param_kwargs_me: list[dict] = []
        self._fsdp_send_shapes: list[list[tuple]] = []
        self._fsdp_tp_gather_info: list[list] = []
        self._fsdp_bucket_params: list[list[torch.Tensor]] = []
        self._fsdp_bucket_group_indices: list[list[int]] = []
        self._fsdp_send_numels: list[list[int]] = []
        self._fsdp_recv_numels: list[list[int]] = []
        self._fsdp_bucket_send_total_elems: list[int] = []
        self._fsdp_bucket_recv_total_elems: list[int] = []
        self._fsdp_bucket_send_chunk_offsets: list[list[int]] = []
        self._fsdp_bucket_recv_chunk_offsets: list[list[int]] = []
        self._fsdp_global_send_offsets: list[list[int]] = []
        self._fsdp_global_recv_offsets: list[list[int]] = []
        self._fsdp_global_input_splits_elems: list[int] = []
        self._fsdp_global_output_splits_elems: list[int] = []
        self._fsdp_global_input_chunk_offsets: list[int] = []
        self._fsdp_global_output_chunk_offsets: list[int] = []
        self._fsdp_global_forward_send_elems = 0
        self._fsdp_global_forward_recv_elems = 0
        self._fsdp_global_reverse_send_elems = 0
        self._fsdp_global_reverse_recv_elems = 0
        self._fsdp_max_bucket_send_elems = 0
        self._fsdp_max_bucket_recv_elems = 0

        self._fsdp_grad_send_slot_bases: list[list[int]] = []
        self._fsdp_upd_send_slot_bases: list[list[int]] = []
        self._fsdp_u_row_offsets: list[list[int]] = []
        self._fsdp_u_flat_offsets: list[list[int]] = []
        self._fsdp_full_g_copy_plan: list[list[tuple[int, int, int]]] = []
        self._fsdp_pack_copy_plan: list[
            tuple[int, int, int, tuple, tuple | None, int]
        ] = []
        self._fsdp_upd_recv_param_plan: list[tuple[int, int, int, tuple]] = []
        self._fsdp_param_local_dtypes: list[torch.dtype] = []
        self._fsdp_uniform_param_dtype: torch.dtype | None = None
        self._fsdp_upd_recv_cast_offset_by_param: list[int] = []
        self._fsdp_upd_recv_cast_numels_by_dtype: dict[torch.dtype, int] = {}
        self._fsdp_upd_recv_cast_plan_by_dtype: dict[
            torch.dtype, list[tuple[int, int, int]]
        ] = {}
        # Persistent workspace: created on first step, reused every subsequent step.
        self._fsdp_once_workspace_cache: dict[str, object] | None = None

        # Static per-parameter singular-value spectrum length/offset tables (see
        # below for how these are populated when fsdp_params is non-empty). Must
        # be initialized here too so referencing them is always safe even when
        # this rank has no FSDP-sharded params.
        self._fsdp_spectrum_len_by_param: list[int] = []
        self._fsdp_rank_owned_param_indices: list[list[int]] = []
        self._fsdp_spectrum_offsets_by_rank: list[list[int]] = []
        self._fsdp_spectrum_total_by_rank: list[int] = []
        self._fsdp_spectrum_max_total: int = 0
        self._fsdp_spectrum_offsets: list[int] = []

        if not self.fsdp_params:
            return

        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        world_size = fsdp_mesh.size()
        rank = fsdp_mesh.get_local_rank()
        self._fsdp_world_size = world_size  # cached for _prepare_fsdp_lmo()
        tp_mesh = (
            self.parallel_dims.get_optional_mesh("tp") if self.tp_enabled else None
        )

        total_buckets = math.ceil(len(self.fsdp_params) / world_size)
        self._fsdp_total_buckets = total_buckets

        # Static per-parameter singular-value spectrum length. `self.fsdp_params[i]`
        # is a DTensor/Parameter, so `.shape` is the *global* logical shape — the
        # a2a reconstruction in step_fsdp gives the owning rank the full (unsharded)
        # tensor for its owned param, so the global shape is what norm/spectrum
        # calculation actually sees. Known at init time from shapes alone, never
        # from data, so every rank can compute this table for every other rank's
        # owned params too (needed to pad every rank's flat spectrum buffer to the
        # same total size, as required by `funcol.all_gather_tensor`).
        def _fsdp_spectrum_len(p) -> int:
            shape = tuple(p.shape)
            if len(shape) == 1:
                return max(int(shape[0]), 1)
            return min(int(shape[-2]), int(shape[-1]))

        self._fsdp_spectrum_len_by_param: list[int] = [
            _fsdp_spectrum_len(p) for p in self.fsdp_params
        ]
        self._fsdp_rank_owned_param_indices: list[list[int]] = [
            [i for i in range(len(self.fsdp_params)) if i % world_size == r]
            for r in range(world_size)
        ]
        self._fsdp_spectrum_offsets_by_rank: list[list[int]] = []
        self._fsdp_spectrum_total_by_rank: list[int] = []
        for r in range(world_size):
            offsets: list[int] = []
            running = 0
            for i in self._fsdp_rank_owned_param_indices[r]:
                offsets.append(running)
                running += self._fsdp_spectrum_len_by_param[i]
            self._fsdp_spectrum_offsets_by_rank.append(offsets)
            self._fsdp_spectrum_total_by_rank.append(running)
        self._fsdp_spectrum_max_total = (
            max(self._fsdp_spectrum_total_by_rank)
            if self._fsdp_spectrum_total_by_rank
            else 0
        )
        self._fsdp_spectrum_offsets = (
            self._fsdp_spectrum_offsets_by_rank[rank]
            if rank < len(self._fsdp_spectrum_offsets_by_rank)
            else []
        )

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * world_size
            end_idx = min(start_idx + world_size, len(self.fsdp_params))
            self._fsdp_bucket_ranges.append((start_idx, end_idx))

            # Determine target shape and kwargs for this rank's owned param.
            my_idx = start_idx + rank
            ref_idx = my_idx if my_idx < end_idx else end_idx - 1
            p_ref = self.fsdp_params[ref_idx]
            target_shape = p_ref.shape  # global (logical) shape
            g_idx = self.parameters_to_groups[id(p_ref)]
            # Structural kwargs (eps/norm_factor/backend/…) are fixed; safe to cache.
            param_kwargs_me = dict(self.groups_info[g_idx][-1])
            self._fsdp_target_shapes.append(target_shape)
            self._fsdp_param_kwargs_me.append(param_kwargs_me)

            # recv_shapes[r]: shape of the shard that rank r holds of my full param.
            recv_shapes = [
                calculate_shard_shape(target_shape, r, world_size)
                for r in range(world_size)
            ]
            self._fsdp_recv_shapes.append(recv_shapes)
            self._fsdp_split_rows.append([s[0] for s in recv_shapes])
            recv_numels = [int(math.prod(s)) for s in recv_shapes]
            recv_chunk_offsets = []
            off = 0
            for n in recv_numels:
                recv_chunk_offsets.append(off)
                off += n
            self._fsdp_bucket_recv_chunk_offsets.append(recv_chunk_offsets)

            send_chunk_offsets = []
            # send_shapes[i] + TP gather info per slot.
            send_shapes: list[tuple] = []
            send_numels: list[int] = []
            tp_infos: list = []
            bucket_params: list[torch.Tensor] = []
            bucket_group_indices: list[int] = []
            for i in range(world_size):
                p_idx = start_idx + i
                if p_idx < end_idx:
                    p = self.fsdp_params[p_idx]
                    g_idx_i = self.parameters_to_groups[id(p)]
                    tp_info = None
                    if isinstance(p, DTensor):
                        placements = p.placements
                        tp_dim = tp_axis(placements)
                        if tp_mesh and tp_dim is not None:
                            shard_dim_tp = placements[tp_dim].dim
                            tp_info = (shard_dim_tp, tp_mesh)
                            send_shape = calculate_shard_shape(
                                p.shape, rank, world_size
                            )
                        else:
                            send_shape = self._get_param_local_view(p).shape
                    else:
                        send_shape = p.shape
                else:
                    # Padding slot – mirror the last real param's local shape.
                    p = self.fsdp_params[end_idx - 1]
                    g_idx_i = self.parameters_to_groups[id(p)]
                    send_shape = (
                        self._get_param_local_view(p).shape
                        if isinstance(p, DTensor)
                        else p.shape
                    )
                    tp_info = None

                send_shapes.append(send_shape)
                send_numels.append(int(math.prod(send_shape)))
                tp_infos.append(tp_info)
                bucket_params.append(p)
                bucket_group_indices.append(g_idx_i)
            off = 0
            for n in send_numels:
                send_chunk_offsets.append(off)
                off += n

            self._fsdp_send_shapes.append(send_shapes)
            self._fsdp_tp_gather_info.append(tp_infos)
            self._fsdp_bucket_params.append(bucket_params)
            self._fsdp_bucket_group_indices.append(bucket_group_indices)
            self._fsdp_send_numels.append(send_numels)
            self._fsdp_recv_numels.append(recv_numels)
            self._fsdp_bucket_send_chunk_offsets.append(send_chunk_offsets)
            self._fsdp_bucket_send_total_elems.append(int(sum(send_numels)))
            self._fsdp_bucket_recv_total_elems.append(int(sum(recv_numels)))

        if total_buckets == 0:
            return

        running_send = [0] * world_size
        running_recv = [0] * world_size
        for bucket_idx in range(total_buckets):
            bucket_send_offsets = []
            bucket_recv_offsets = []
            for i in range(world_size):
                bucket_send_offsets.append(running_send[i])
                bucket_recv_offsets.append(running_recv[i])
                running_send[i] += self._fsdp_send_numels[bucket_idx][i]
                running_recv[i] += self._fsdp_recv_numels[bucket_idx][i]
            self._fsdp_global_send_offsets.append(bucket_send_offsets)
            self._fsdp_global_recv_offsets.append(bucket_recv_offsets)

        self._fsdp_global_input_splits_elems = list(running_send)
        self._fsdp_global_output_splits_elems = list(running_recv)

        input_chunk_offsets = []
        off = 0
        for n in self._fsdp_global_input_splits_elems:
            input_chunk_offsets.append(off)
            off += n
        self._fsdp_global_input_chunk_offsets = input_chunk_offsets

        output_chunk_offsets = []
        off = 0
        for n in self._fsdp_global_output_splits_elems:
            output_chunk_offsets.append(off)
            off += n
        self._fsdp_global_output_chunk_offsets = output_chunk_offsets

        self._fsdp_global_forward_send_elems = int(
            sum(self._fsdp_global_input_splits_elems)
        )
        self._fsdp_global_forward_recv_elems = int(
            sum(self._fsdp_global_output_splits_elems)
        )
        self._fsdp_global_reverse_send_elems = self._fsdp_global_forward_recv_elems
        self._fsdp_global_reverse_recv_elems = self._fsdp_global_forward_send_elems
        self._fsdp_max_bucket_send_elems = max(self._fsdp_bucket_send_total_elems)
        self._fsdp_max_bucket_recv_elems = max(self._fsdp_bucket_recv_total_elems)

        self._fsdp_grad_send_slot_bases = [
            [
                self._fsdp_global_input_chunk_offsets[i]
                + self._fsdp_global_send_offsets[b][i]
                for i in range(world_size)
            ]
            for b in range(total_buckets)
        ]
        self._fsdp_upd_send_slot_bases = [
            [
                self._fsdp_global_output_chunk_offsets[d]
                + self._fsdp_global_recv_offsets[b][d]
                for d in range(world_size)
            ]
            for b in range(total_buckets)
        ]
        self._fsdp_u_row_offsets = []
        self._fsdp_u_flat_offsets = []
        for b in range(total_buckets):
            row_off, offsets = 0, []
            for rows in self._fsdp_split_rows[b]:
                offsets.append(row_off)
                row_off += rows
            self._fsdp_u_row_offsets.append(offsets)
            target_shape = self._fsdp_target_shapes[b]
            cols = int(math.prod(target_shape[1:])) if len(target_shape) > 1 else 1
            self._fsdp_u_flat_offsets.append([int(r * cols) for r in offsets])

        self._fsdp_full_g_copy_plan = []
        for b in range(total_buckets):
            recv_numels = self._fsdp_recv_numels[b]
            recv_offsets = self._fsdp_global_recv_offsets[b]
            bucket_plan: list[tuple[int, int, int]] = []
            dst_offset = 0
            for src in range(world_size):
                numel = recv_numels[src]
                src_base = (
                    self._fsdp_global_output_chunk_offsets[src] + recv_offsets[src]
                )
                bucket_plan.append((src_base, dst_offset, numel))
                dst_offset += numel
            target_numel = int(math.prod(self._fsdp_target_shapes[b]))
            if dst_offset != target_numel:
                raise RuntimeError(
                    "FSDP full_g assembly total mismatch: "
                    f"bucket={b}, assembled={dst_offset}, expected={target_numel}."
                )
            self._fsdp_full_g_copy_plan.append(bucket_plan)

        self._fsdp_pack_copy_plan = []
        for b in range(total_buckets):
            start_idx, end_idx = self._fsdp_bucket_ranges[b]
            send_bases = self._fsdp_grad_send_slot_bases[b]
            send_numels = self._fsdp_send_numels[b]
            send_shapes = self._fsdp_send_shapes[b]
            tp_infos = self._fsdp_tp_gather_info[b]
            group_indices = self._fsdp_bucket_group_indices[b]
            for i in range(world_size):
                # For phantom slots (i >= end_idx - start_idx), clamp to the last
                # real param so the A2A send buffer is filled with actual gradient
                # data rather than zeros. This ensures all ranks run LMO on real
                # tensors and keeps timing symmetric across the fsdp group.
                # _fsdp_upd_recv_param_plan still uses range(end_idx - start_idx)
                # so phantom slot outputs from the reverse A2A are discarded and
                # the last real param is updated exactly once.
                param_idx = min(start_idx + i, end_idx - 1)
                self._fsdp_pack_copy_plan.append(
                    (
                        param_idx,
                        send_bases[i],
                        send_numels[i],
                        send_shapes[i],
                        tp_infos[i],
                        group_indices[i],
                    )
                )

        self._fsdp_upd_recv_param_plan = []
        for b in range(total_buckets):
            start_idx, end_idx = self._fsdp_bucket_ranges[b]
            send_numels = self._fsdp_send_numels[b]
            send_shapes = self._fsdp_send_shapes[b]
            send_offsets = self._fsdp_global_send_offsets[b]
            for i in range(end_idx - start_idx):
                param_idx = start_idx + i
                base = self._fsdp_global_input_chunk_offsets[i] + send_offsets[i]
                self._fsdp_upd_recv_param_plan.append(
                    (param_idx, base, send_numels[i], send_shapes[i])
                )

        self._fsdp_param_local_dtypes = [
            self._get_param_local_view(p).dtype for p in self.fsdp_params
        ]
        self._fsdp_upd_recv_cast_offset_by_param = [0] * len(self.fsdp_params)
        self._fsdp_upd_recv_cast_numels_by_dtype = {}
        self._fsdp_upd_recv_cast_plan_by_dtype = {}
        if self._fsdp_param_local_dtypes and all(
            dt == self._fsdp_param_local_dtypes[0]
            for dt in self._fsdp_param_local_dtypes
        ):
            self._fsdp_uniform_param_dtype = self._fsdp_param_local_dtypes[0]
        else:
            self._fsdp_uniform_param_dtype = None
            offsets_by_dtype: dict[torch.dtype, int] = {}
            for param_idx, base, numel, _ in self._fsdp_upd_recv_param_plan:
                dtype = self._fsdp_param_local_dtypes[param_idx]
                off = offsets_by_dtype.get(dtype, 0)
                self._fsdp_upd_recv_cast_offset_by_param[param_idx] = off
                self._fsdp_upd_recv_cast_plan_by_dtype.setdefault(dtype, []).append(
                    (base, numel, off)
                )
                offsets_by_dtype[dtype] = off + numel
            self._fsdp_upd_recv_cast_numels_by_dtype = offsets_by_dtype

        # Safety guard: all global_upd_send write ranges must be disjoint.
        update_ranges: list[tuple[int, int, int, int]] = []
        total_upd_send_elems = int(self._fsdp_global_reverse_send_elems)
        for b in range(total_buckets):
            for dst in range(world_size):
                start = int(self._fsdp_upd_send_slot_bases[b][dst])
                end = start + int(self._fsdp_recv_numels[b][dst])
                if not (0 <= start <= end <= total_upd_send_elems):
                    raise RuntimeError(
                        "FSDP global_upd_send range out of bounds: "
                        f"bucket={b}, dst={dst}, range=[{start}, {end}), "
                        f"capacity={total_upd_send_elems}."
                    )
                update_ranges.append((start, end, b, dst))

        update_ranges.sort(key=lambda x: (x[0], x[1]))
        prev_end = 0
        prev_meta: tuple[int, int] | None = None
        for start, end, b, dst in update_ranges:
            if prev_meta is not None and start < prev_end:
                pb, pd = prev_meta
                raise RuntimeError(
                    "FSDP global_upd_send write overlap detected: "
                    f"prev=(bucket={pb}, dst={pd}, end={prev_end}) "
                    f"curr=(bucket={b}, dst={dst}, start={start})."
                )
            prev_end = end
            prev_meta = (b, dst)

    def _precompute_fsdp_gram_vector_metadata(self):
        """
        Static per-parameter gram-vector length/offset tables, mirroring the
        spectrum tables in `_precompute_fsdp_metadata` (pad-to-rank-max) but
        using `gram_helper.gram_vector_len` (== shape[-2], NOT
        min(shape[-2], shape[-1])) and scaled by the number of vector-valued
        gram metrics at the current level. Callable independently of the
        rest of `_precompute_fsdp_metadata` (reuses its already-built
        `_fsdp_rank_owned_param_indices`), so it can be cheaply re-run
        whenever `self.gram_level` changes without redoing the full FSDP
        bucket/a2a plan.
        """
        self._fsdp_gram_vec_len_by_param: list[int] = []
        self._fsdp_gram_vec_offsets_by_rank: list[list[int]] = []
        self._fsdp_gram_vec_total_by_rank: list[int] = []
        self._fsdp_gram_vec_max_total: int = 0
        self._fsdp_gram_vec_offsets: list[int] = []
        if not self.fsdp_params:
            return
        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        rank = fsdp_mesh.get_local_rank()
        n_vec = len(self.gram_vector_names)
        self._fsdp_gram_vec_len_by_param = [
            gram_helper.gram_vector_len(tuple(p.shape)) for p in self.fsdp_params
        ]
        for r in range(len(self._fsdp_rank_owned_param_indices)):
            offsets: list[int] = []
            running = 0
            for i in self._fsdp_rank_owned_param_indices[r]:
                offsets.append(running)
                running += self._fsdp_gram_vec_len_by_param[i] * n_vec
            self._fsdp_gram_vec_offsets_by_rank.append(offsets)
            self._fsdp_gram_vec_total_by_rank.append(running)
        self._fsdp_gram_vec_max_total = (
            max(self._fsdp_gram_vec_total_by_rank)
            if self._fsdp_gram_vec_total_by_rank
            else 0
        )
        self._fsdp_gram_vec_offsets = (
            self._fsdp_gram_vec_offsets_by_rank[rank]
            if rank < len(self._fsdp_gram_vec_offsets_by_rank)
            else []
        )

    def _precompute_experts_metadata(self):
        """
        Pre-compute structural expert-batch metadata so step_experts avoids
        per-step recomputation of blocks, ep_per_rank, kwargs, etc.
        """
        self._expert_blocks: list[tuple[int, int]] = []
        self._expert_ep_per_rank: int = 0
        self._expert_kinds_of_norms: int = 0
        self._expert_transpose: bool = False
        # Static per-block singular-value spectrum length (min(A, B) of the local
        # per-expert matrix); uniform within a block since blocks group same-shape
        # experts. Known at init time from shapes alone, never from data.
        self._expert_spectrum_len_per_block: list[int] = []
        # Static offset of each block's spectrum values within the single flat
        # per-rank spectrum buffer (unlike fsdp/ddp, every rank contributes the
        # same shape per block — ep_per_rank is uniform across ranks — so no
        # pad-to-max-per-rank scheme is needed here, just one static layout).
        self._expert_spectrum_block_offset: list[int] = []
        self._expert_spectrum_total_size: int = 0
        self._expert_kwargs_per_block: list[dict | None] = []
        self._expert_block_group_idx: list[int | None] = []
        self._expert_big_g_specs: list[
            tuple[int, int, int, torch.dtype, torch.device] | None
        ] = []
        self._expert_step_workspace_cache: dict[str, object] | None = None
        self._expert_update_plan: list[
            tuple[
                int,
                torch.device,
                torch.dtype,
                list[int],
                list[torch.Tensor],
                list[torch.Tensor],
            ]
        ] = []

        if not self.expert_params:
            return

        total = len(self.expert_params)
        L = total // 3
        assert total == 3 * L, f"Expected 3*L expert params, got {total}"

        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        world_size = fsdp_mesh.size()

        # Keep the original expert split (2 blocks). Stream pool can still be larger.
        def _shape_key(idx: int) -> tuple[int, int]:
            p = self.expert_params[idx]
            return (int(p.shape[1]), int(p.shape[2]))

        s0 = _shape_key(0)
        s1 = _shape_key(L)
        s2 = _shape_key(2 * L)
        if s0 == s1 and s1 != s2:
            self._expert_blocks = [(0, 2 * L), (2 * L, 3 * L)]
        elif s0 != s1 and s1 == s2:
            self._expert_blocks = [(0, L), (L, 3 * L)]
        elif s0 == s1 and s1 == s2:
            self._expert_blocks = [(0, 3 * L)]
        else:
            # Unexpected ordering/layout: fallback to contiguous shape-homogeneous blocks.
            shape_keys: list[tuple[int, int]] = [_shape_key(i) for i in range(total)]
            blocks: list[tuple[int, int]] = []
            start = 0
            for i in range(1, total):
                if shape_keys[i] != shape_keys[i - 1]:
                    blocks.append((start, i))
                    start = i
            blocks.append((start, total))
            self._expert_blocks = blocks

        self._expert_ep_per_rank = math.ceil(
            self.expert_params[0].shape[0] / world_size
        )
        self._expert_kinds_of_norms = len(self.norms_to_log)
        self._expert_transpose = self.experts_need_transpose

        ep_per_rank = self._expert_ep_per_rank
        for block_idx, (start, end) in enumerate(self._expert_blocks):
            block_params = self.expert_params[start:end]
            if block_params:
                g_idx = self.parameters_to_groups[id(block_params[0])]
                kwargs = dict(self.groups_info[g_idx][-1])
                self._expert_block_group_idx.append(g_idx)
                K = end - start
                p0 = block_params[0]
                loc = self._get_param_local_view(p0)  # Tensor [ep_per_rank, A, B]
                self._expert_big_g_specs.append(
                    (
                        K * ep_per_rank,
                        int(loc.shape[1]),
                        int(loc.shape[2]),
                        loc.dtype,
                        loc.device,
                    )
                )
                self._expert_spectrum_len_per_block.append(
                    min(int(loc.shape[1]), int(loc.shape[2]))
                )
            else:
                kwargs = None
                self._expert_block_group_idx.append(None)
                self._expert_big_g_specs.append(None)
                self._expert_spectrum_len_per_block.append(0)
            self._expert_kwargs_per_block.append(kwargs)

        running = 0
        for block_idx, (start, end) in enumerate(self._expert_blocks):
            self._expert_spectrum_block_offset.append(running)
            running += (
                (end - start)
                * ep_per_rank
                * self._expert_spectrum_len_per_block[block_idx]
            )
        self._expert_spectrum_total_size = running

        # Precompute expert update apply plan (grouped by group/device/dtype).
        update_buckets: dict[
            tuple[int, torch.device, torch.dtype], dict[str, list]
        ] = {}
        for param_idx, p in enumerate(self.expert_params):
            group_idx = self.parameters_to_groups[id(p)]
            p_local = self._get_param_local_view(p)
            key = (group_idx, p_local.device, p_local.dtype)
            if key not in update_buckets:
                update_buckets[key] = {"indices": [], "params": [], "locals": []}
            b = update_buckets[key]
            b["indices"].append(param_idx)
            b["params"].append(p)
            b["locals"].append(p_local)
        self._expert_update_plan = [
            (
                group_idx,
                device,
                dtype,
                data["indices"],
                data["params"],
                data["locals"],
            )
            for (group_idx, device, dtype), data in update_buckets.items()
        ]

    def _precompute_experts_gram_vector_metadata(self):
        """
        Static per-block gram-vector length/offset table, mirroring the
        spectrum table in `_precompute_experts_metadata` -- uniform per rank
        (ep_per_rank is uniform across ranks, no pad-to-max needed) -- using
        `gram_helper.gram_vector_len` (== min(loc.shape[1], loc.shape[2]),
        since `calculate_gram_metrics` always orients rows <= cols -- see
        gram_helper.py's orientation-policy docstring) and scaled by the
        number of vector-valued gram metrics at the current level. Callable
        independently of the rest of `_precompute_experts_metadata` (reuses
        its already-built `_expert_blocks`/`_expert_ep_per_rank`), so it can
        be cheaply re-run whenever `self.gram_level` changes without redoing
        the full expert-block plan.
        """
        self._expert_gram_vec_len_per_block: list[int] = []
        self._expert_gram_vec_block_offset: list[int] = []
        self._expert_gram_vec_total_size: int = 0
        if not self.expert_params:
            return
        n_vec = len(self.gram_vector_names)
        ep_per_rank = self._expert_ep_per_rank
        for start, end in self._expert_blocks:
            block_params = self.expert_params[start:end]
            if block_params:
                loc = self._get_param_local_view(block_params[0])
                self._expert_gram_vec_len_per_block.append(
                    gram_helper.gram_vector_len((int(loc.shape[1]), int(loc.shape[2])))
                )
            else:
                self._expert_gram_vec_len_per_block.append(0)
        running = 0
        for block_idx, (start, end) in enumerate(self._expert_blocks):
            self._expert_gram_vec_block_offset.append(running)
            running += (
                (end - start)
                * ep_per_rank
                * self._expert_gram_vec_len_per_block[block_idx]
                * n_vec
            )
        self._expert_gram_vec_total_size = running

    def _precompute_ddp_metadata(self):
        """
        Pre-compute structural DDP metadata so step_ddp avoids per-step
        bucket math, repeated param->group dict lookups, and runtime
        update/logging planning overhead.
        """
        self._ddp_total_buckets: int = 0
        self._ddp_world_size: int = 1
        self._ddp_rank: int = 0
        self._ddp_owned_indices: list[int] = []
        self._ddp_owned_indices_by_group: list[
            tuple[int, list[int], list[torch.Tensor]]
        ] = []
        self._ddp_rank_owned_param_indices: list[list[int]] = []
        self._ddp_rank_owned_numels: list[int] = []
        self._ddp_max_rank_owned_numel: int = 0
        self._ddp_owned_pack_offsets: list[int] = []
        self._ddp_unpack_offsets_by_rank: list[dict[int, int]] = []
        self._ddp_param_group_idx: list[int] = []
        self._ddp_tp_slice_info_by_idx: list[tuple | None] = []
        self._ddp_tp_gather_info_by_idx: list[tuple | None] = []
        self._ddp_local_shape_by_idx: list[tuple[int, ...]] = []
        self._ddp_full_shape_by_idx: list[tuple[int, ...]] = []
        self._ddp_param_local_shapes: list[tuple[int, ...]] = []  # alias for apply path
        self._ddp_cast_indices_by_dtype: dict[torch.dtype, list[int]] = {}
        self._ddp_lmo_non_tp_precast_indices_by_src_dtype: dict[
            torch.dtype, list[int]
        ] = {}
        self._ddp_owner_rank_by_param: list[int] = []
        self._ddp_owner_bucket_by_param: list[int] = []
        self._ddp_clean_param_names: list[str] = []
        # Static per-parameter singular-value spectrum length/offset tables (see
        # below for how these are populated when ddp_params is non-empty). Must
        # be initialized here too so referencing them is always safe even when
        # this rank has no DDP-replicated params.
        self._ddp_spectrum_len_by_param: list[int] = []
        self._ddp_spectrum_offsets_by_rank: list[list[int]] = []
        self._ddp_spectrum_total_by_rank: list[int] = []
        self._ddp_spectrum_max_total: int = 0
        self._ddp_update_plan: list[
            tuple[
                int,
                torch.device,
                torch.dtype,
                list[int],
                list[torch.Tensor],
                list[torch.Tensor],
            ]
        ] = []
        self._ddp_step_workspace_cache: dict[str, object] | None = None

        if not self.ddp_params:
            return

        # TODO(jsc): For now, we force MoE training is with FSDP.
        # SO no MoE training with DDP.
        invalid_ddp_params: list[str] = []
        for p, p_name in zip(self.ddp_params, self.ddp_param_names):
            if p.ndim > 2:
                invalid_ddp_params.append(
                    f"{p_name} shape={tuple(p.shape)} ndim={p.ndim}"
                )
        if invalid_ddp_params:
            raise ValueError(
                "DDP norm path only supports parameters with ndim <= 2. "
                "Move higher-rank tensors to a non-DDP norm path. "
                f"Offending params: {', '.join(invalid_ddp_params)}"
            )

        dp_replicate_mesh = (
            self.parallel_dims.get_optional_mesh("dp_replicate")
            if self.dp_replicate_enabled
            else None
        )
        world_size = dp_replicate_mesh.size() if dp_replicate_mesh else 1
        rank = dp_replicate_mesh.get_local_rank() if dp_replicate_mesh else 0
        self._ddp_world_size = world_size
        self._ddp_rank = rank

        n_params = len(self.ddp_params)
        bucket_size = world_size
        total_buckets = (
            math.ceil(n_params / bucket_size) if world_size > 1 else n_params
        )
        self._ddp_total_buckets = total_buckets

        for bucket_idx in range(total_buckets):
            start_idx = bucket_idx * bucket_size
            end_idx = min(start_idx + bucket_size, n_params)
            my_idx = start_idx + rank
            if my_idx < end_idx:
                self._ddp_owned_indices.append(my_idx)

        self._ddp_rank_owned_param_indices = [
            [i for i in range(n_params) if i % world_size == r]
            for r in range(world_size)
        ]
        self._ddp_rank_owned_numels = [
            sum(int(self.ddp_params[i].numel()) for i in indices)
            for indices in self._ddp_rank_owned_param_indices
        ]
        self._ddp_max_rank_owned_numel = (
            max(self._ddp_rank_owned_numels) if self._ddp_rank_owned_numels else 0
        )

        offset = 0
        for param_idx in self._ddp_owned_indices:
            self._ddp_owned_pack_offsets.append(offset)
            offset += int(self.ddp_params[param_idx].numel())

        for r in range(world_size):
            rank_offsets: dict[int, int] = {}
            offset = 0
            for param_idx in self._ddp_rank_owned_param_indices[r]:
                rank_offsets[param_idx] = offset
                offset += int(self.ddp_params[param_idx].numel())
            self._ddp_unpack_offsets_by_rank.append(rank_offsets)

        update_buckets: dict[
            tuple[int, torch.device, torch.dtype], dict[str, list]
        ] = {}
        owned_by_group: dict[int, dict[str, list]] = {}
        tp_slice_info = self._tp_slice_info if self.tp_enabled else {}
        comm_dtype = self.communication_dtype

        for param_idx, p in enumerate(self.ddp_params):
            group_idx = self.parameters_to_groups[id(p)]
            self._ddp_param_group_idx.append(group_idx)

            p_local = self._get_param_local_view(p)
            local_shape = tuple(p_local.shape)
            full_shape = tuple(p.shape)
            self._ddp_local_shape_by_idx.append(local_shape)
            self._ddp_full_shape_by_idx.append(full_shape)
            self._ddp_param_local_shapes.append(local_shape)
            self._ddp_tp_slice_info_by_idx.append(tp_slice_info.get(id(p)))
            if self.tp_enabled and isinstance(p, DTensor):
                placements = p.placements
                self._ddp_tp_gather_info_by_idx.append(
                    placements
                    if tp_axis(placements, tp_enabled=True) is not None
                    else None
                )
            else:
                self._ddp_tp_gather_info_by_idx.append(None)

            if p_local.dtype != comm_dtype:
                self._ddp_cast_indices_by_dtype.setdefault(p_local.dtype, []).append(
                    param_idx
                )

            key = (group_idx, p_local.device, p_local.dtype)
            if key not in update_buckets:
                update_buckets[key] = {"indices": [], "params": [], "locals": []}
            b = update_buckets[key]
            b["indices"].append(param_idx)
            b["params"].append(p)
            b["locals"].append(p_local)

        for idx in self._ddp_owned_indices:
            group_idx = self._ddp_param_group_idx[idx]
            if group_idx not in owned_by_group:
                owned_by_group[group_idx] = {"indices": [], "params": []}
            owned_by_group[group_idx]["indices"].append(idx)
            owned_by_group[group_idx]["params"].append(self.ddp_params[idx])

            p_local = self._get_param_local_view(self.ddp_params[idx])
            tp_info = self._ddp_tp_gather_info_by_idx[idx]
            if tp_info is None and p_local.dtype != comm_dtype:
                self._ddp_lmo_non_tp_precast_indices_by_src_dtype.setdefault(
                    p_local.dtype, []
                ).append(idx)

        self._ddp_owned_indices_by_group = [
            (group_idx, data["indices"], data["params"])
            for group_idx, data in owned_by_group.items()
        ]

        self._ddp_owner_rank_by_param = [i % world_size for i in range(n_params)]
        self._ddp_owner_bucket_by_param = [i // world_size for i in range(n_params)]
        self._ddp_clean_param_names = [
            remove_orig_mod_and_weight_for_p_name(pn) for pn in self.ddp_param_names
        ]

        # Static per-parameter singular-value spectrum length (min(fan_in, fan_out), or
        # the diag-embedded length for 1-D params). Known at init time from shapes
        # alone, never from data, so every rank can compute this table for every
        # other rank's owned params too (needed to pad every rank's flat spectrum
        # buffer to the same total size, as required by `funcol.all_gather_tensor`).
        def _ddp_spectrum_len(p) -> int:
            shape = tuple(p.shape)
            if len(shape) == 1:
                return max(int(shape[0]), 1)
            return min(int(shape[0]), int(shape[1]))

        self._ddp_spectrum_len_by_param: list[int] = [
            _ddp_spectrum_len(p) for p in self.ddp_params
        ]
        # Per rank, the list of owned param indices is already in increasing
        # owner-bucket order (`_ddp_rank_owned_param_indices[r][pos]` has
        # owner_bucket == pos), so a running cumulative sum gives each owned
        # param's start offset within that rank's flat spectrum buffer.
        self._ddp_spectrum_offsets_by_rank: list[list[int]] = []
        self._ddp_spectrum_total_by_rank: list[int] = []
        for r in range(world_size):
            offsets: list[int] = []
            running = 0
            for i in self._ddp_rank_owned_param_indices[r]:
                offsets.append(running)
                running += self._ddp_spectrum_len_by_param[i]
            self._ddp_spectrum_offsets_by_rank.append(offsets)
            self._ddp_spectrum_total_by_rank.append(running)
        self._ddp_spectrum_max_total = (
            max(self._ddp_spectrum_total_by_rank)
            if self._ddp_spectrum_total_by_rank
            else 0
        )
        self._ddp_update_plan = [
            (
                group_idx,
                device,
                dtype,
                data["indices"],
                data["params"],
                data["locals"],
            )
            for (group_idx, device, dtype), data in update_buckets.items()
        ]

    def _precompute_ddp_gram_vector_metadata(self):
        """
        Static per-parameter gram-vector length/offset tables, mirroring the
        spectrum tables above (pad-to-rank-max, since different ranks own
        different-shaped params) -- but using `gram_helper.gram_vector_len`
        (== shape[-2], NOT min(shape[0], shape[1])) and scaled by the number
        of vector-valued gram metrics at the current level, since every
        param packs `self.gram_vector_names` vectors back-to-back. Callable
        independently of the rest of `_precompute_ddp_metadata` (reuses its
        already-built `_ddp_world_size`/`_ddp_rank_owned_param_indices`), so
        it can be cheaply re-run whenever `self.gram_level` changes without
        redoing the full DDP bucket/ownership plan.
        """
        self._ddp_gram_vec_len_by_param: list[int] = []
        self._ddp_gram_vec_offsets_by_rank: list[list[int]] = []
        self._ddp_gram_vec_total_by_rank: list[int] = []
        self._ddp_gram_vec_max_total: int = 0
        if not self.ddp_params:
            return
        n_vec = len(self.gram_vector_names)
        self._ddp_gram_vec_len_by_param = [
            gram_helper.gram_vector_len(tuple(p.shape)) for p in self.ddp_params
        ]
        for r in range(self._ddp_world_size):
            offsets: list[int] = []
            running = 0
            for i in self._ddp_rank_owned_param_indices[r]:
                offsets.append(running)
                running += self._ddp_gram_vec_len_by_param[i] * n_vec
            self._ddp_gram_vec_offsets_by_rank.append(offsets)
            self._ddp_gram_vec_total_by_rank.append(running)
        self._ddp_gram_vec_max_total = (
            max(self._ddp_gram_vec_total_by_rank)
            if self._ddp_gram_vec_total_by_rank
            else 0
        )

    def _create_embed_step_workspace(self, device: torch.device) -> dict:
        """
        Lazy CUDA alloc for embed step buffers. Step-wise by default;
        persistent when persistent_cache_enabled is True.
        """
        if (
            self.persistent_cache_enabled
            and self._embed_step_workspace_cache is not None
            and self._embed_step_workspace_cache["device"] == device
        ):
            return self._embed_step_workspace_cache

        n_params = len(self.embed_params)

        # Pre-alloc fill buffers for shape groups with ≥2 params (like expert big_g).
        # Views pre-created with unbind (ONE aten::unbind at init, not per step).
        batched_g_bufs: dict[tuple, torch.Tensor] = {}
        buf_views_by_shape: dict[tuple, list[torch.Tensor]] = {}
        for shape, indices in self._embed_extra_shape_groups.items():
            if len(indices) < 2:
                continue
            N = len(indices)
            p0 = self.embed_params[indices[0]]
            dtype = self._get_param_local_view(p0).dtype
            buf = torch.empty(N, *shape, dtype=dtype, device=device)
            batched_g_bufs[shape] = buf
            buf_views_by_shape[shape] = list(buf.unbind(0))  # ONE aten::unbind at init

        # Per-extra-param float32 scratch for Phase 2 norm (avoids -lr*u temp alloc)
        norm_scratch: list[torch.Tensor | None] = [None] * n_params
        for i in self._embed_extra_indices:
            p_local = self._get_param_local_view(self.embed_params[i])
            norm_scratch[i] = torch.empty(
                p_local.shape, dtype=torch.float32, device=device
            )

        ws = {
            "device": device,
            "n_params": n_params,
            "batched_g_bufs": batched_g_bufs,
            "buf_views_by_shape": buf_views_by_shape,
            "norm_scratch": norm_scratch,
            "updates": [None] * n_params,
            "effective_grads": [None] * n_params,
        }
        if self.persistent_cache_enabled:
            self._embed_step_workspace_cache = ws
        return ws

    def _create_ddp_step_workspace(
        self, cast_dtype: torch.dtype, device: torch.device
    ) -> dict[str, object]:
        n_params = len(self.ddp_params)
        norm_k = len(self.norms_to_log)
        if (
            self._ddp_step_workspace_cache is not None
            and self.persistent_cache_enabled
            and self._ddp_step_workspace_cache.get("device") == device
            and self._ddp_step_workspace_cache.get("cast_dtype") == cast_dtype
            and self._ddp_step_workspace_cache.get("n_params") == n_params
            and self._ddp_step_workspace_cache.get("norm_k") == norm_k
            and self._ddp_step_workspace_cache.get("world_size") == self._ddp_world_size
            and self._ddp_step_workspace_cache.get("rank") == self._ddp_rank
        ):
            return self._ddp_step_workspace_cache

        global_update_bufs: list[torch.Tensor] = [
            torch.empty(p.shape, dtype=cast_dtype, device=device)
            for p in self.ddp_params
        ]
        zero_by_shape: dict[tuple[tuple[int, ...], torch.dtype], torch.Tensor] = {}
        norm_scratch: list[torch.Tensor | None] = [None] * n_params

        for p in self.ddp_params:
            shape = tuple(p.shape)
            key = (shape, cast_dtype)
            if key not in zero_by_shape:
                zero_by_shape[key] = torch.zeros(shape, dtype=cast_dtype, device=device)

        for param_idx in self._ddp_owned_indices:
            p = self.ddp_params[param_idx]
            norm_scratch[param_idx] = torch.empty(
                p.shape, dtype=cast_dtype, device=device
            )

        if cast_dtype == self.communication_dtype:
            cast_indices_by_dtype = self._ddp_cast_indices_by_dtype
        else:
            cast_indices_by_dtype: dict[torch.dtype, list[int]] = {}
            for param_idx, p in enumerate(self.ddp_params):
                p_local = self._get_param_local_view(p)
                if p_local.dtype != cast_dtype:
                    cast_indices_by_dtype.setdefault(p_local.dtype, []).append(
                        param_idx
                    )

        cast_dst_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for dtype, indices in cast_indices_by_dtype.items():
            cast_dst_by_dtype[dtype] = [
                torch.empty(
                    self._ddp_param_local_shapes[param_idx], dtype=dtype, device=device
                )
                for param_idx in indices
            ]

        lmo_inputs_by_idx: list[torch.Tensor | None] = [None] * n_params
        tp_local_cast_bufs_by_idx: list[torch.Tensor | None] = [None] * n_params
        lmo_non_tp_cast_bufs_by_idx: list[torch.Tensor | None] = [None] * n_params
        tp_lmo_precast_indices: set[int] = {
            idx
            for indices in self._ddp_cast_indices_by_dtype.values()
            for idx in indices
            if self._ddp_tp_gather_info_by_idx[idx] is not None
        }
        for param_idx in self._ddp_owned_indices:
            if (
                param_idx in tp_lmo_precast_indices
                and self._ddp_tp_gather_info_by_idx[param_idx] is not None
            ):
                tp_local_cast_bufs_by_idx[param_idx] = torch.empty(
                    self._ddp_local_shape_by_idx[param_idx],
                    dtype=cast_dtype,
                    device=device,
                )
        for indices in self._ddp_lmo_non_tp_precast_indices_by_src_dtype.values():
            for param_idx in indices:
                lmo_non_tp_cast_bufs_by_idx[param_idx] = torch.empty(
                    self._ddp_full_shape_by_idx[param_idx],
                    dtype=cast_dtype,
                    device=device,
                )

        max_owned_numel = self._ddp_max_rank_owned_numel
        ddp_flat_send = torch.empty(max_owned_numel, dtype=cast_dtype, device=device)
        ddp_flat_recv_chunks = [
            torch.empty(max_owned_numel, dtype=cast_dtype, device=device)
            for _ in range(self._ddp_world_size)
        ]
        ddp_pack_dst_views: list[torch.Tensor] = []
        for j, param_idx in enumerate(self._ddp_owned_indices):
            offset = self._ddp_owned_pack_offsets[j]
            numel = int(self.ddp_params[param_idx].numel())
            ddp_pack_dst_views.append(
                ddp_flat_send.narrow(0, offset, numel).view(
                    self._ddp_full_shape_by_idx[param_idx]
                )
            )

        ddp_unpack_src_views: list[torch.Tensor] = []
        ddp_unpack_dst_views: list[torch.Tensor] = []
        for r in range(self._ddp_world_size):
            rank_offsets = self._ddp_unpack_offsets_by_rank[r]
            for param_idx in self._ddp_rank_owned_param_indices[r]:
                offset = rank_offsets[param_idx]
                numel = int(self.ddp_params[param_idx].numel())
                ddp_unpack_src_views.append(
                    ddp_flat_recv_chunks[r]
                    .narrow(0, offset, numel)
                    .view(self._ddp_full_shape_by_idx[param_idx])
                )
                ddp_unpack_dst_views.append(global_update_bufs[param_idx])

        total_norm_elems = self._ddp_total_buckets * norm_k
        upd_norm_local_flat = torch.empty(
            total_norm_elems, dtype=torch.float32, device=device
        )

        ws = {
            "device": device,
            "cast_dtype": cast_dtype,
            "n_params": n_params,
            "norm_k": norm_k,
            "world_size": self._ddp_world_size,
            "rank": self._ddp_rank,
            "global_update_bufs": global_update_bufs,
            "zero_by_shape": zero_by_shape,
            "norm_scratch": norm_scratch,
            "global_updates": [None] * n_params,
            "apply_updates": [None] * n_params,
            "cast_indices_by_dtype": cast_indices_by_dtype,
            "cast_dst_by_dtype": cast_dst_by_dtype,
            "lmo_inputs_by_idx": lmo_inputs_by_idx,
            "tp_local_cast_bufs_by_idx": tp_local_cast_bufs_by_idx,
            "lmo_non_tp_cast_bufs_by_idx": lmo_non_tp_cast_bufs_by_idx,
            "ddp_flat_send": ddp_flat_send,
            "ddp_flat_recv_chunks": ddp_flat_recv_chunks,
            "ddp_pack_dst_views": ddp_pack_dst_views,
            "ddp_unpack_src_views": ddp_unpack_src_views,
            "ddp_unpack_dst_views": ddp_unpack_dst_views,
            "upd_norm_local_flat": upd_norm_local_flat,
            "w_norm_local_flat": None,
        }
        if self.persistent_cache_enabled:
            self._ddp_step_workspace_cache = ws
        return ws

    def _create_expert_step_workspace(self) -> dict[str, object]:
        if (
            self.persistent_cache_enabled
            and self._expert_step_workspace_cache is not None
        ):
            return self._expert_step_workspace_cache

        def _allocate_expert_big_g_bufs() -> list[torch.Tensor | None]:
            bufs: list[torch.Tensor | None] = []
            for spec in self._expert_big_g_specs:
                if spec is None:
                    bufs.append(None)
                    continue
                n, a, b, dtype, device = spec
                bufs.append(torch.zeros((n, a, b), dtype=dtype, device=device))
            return bufs

        def _create_expert_big_g_views(
            big_g_bufs: list[torch.Tensor | None] | None,
        ) -> list[list[torch.Tensor]]:
            if not big_g_bufs:
                return []

            ep_per_rank = self._expert_ep_per_rank
            views_by_block: list[list[torch.Tensor]] = []
            for block_idx, (start, end) in enumerate(self._expert_blocks):
                big_g = big_g_bufs[block_idx]
                if big_g is None:
                    views_by_block.append([])
                    continue
                k_count = end - start
                block_views = []
                for k in range(k_count):
                    base = k * ep_per_rank
                    block_views.append(big_g[base : base + ep_per_rank])
                views_by_block.append(block_views)
            return views_by_block

        big_g_bufs = _allocate_expert_big_g_bufs()
        dst_views_by_block = _create_expert_big_g_views(big_g_bufs)
        ws: dict[str, object] = {
            "big_g_bufs": big_g_bufs,
            "dst_views_by_block": dst_views_by_block,
        }
        if self.persistent_cache_enabled:
            self._expert_step_workspace_cache = ws
        return ws

    def _allocate_fsdp_once_workspace(
        self, cast_dtype: torch.dtype, device: torch.device
    ) -> dict[str, object]:
        workspace: dict[str, object] = {
            "global_grad_send": torch.zeros(
                self._fsdp_global_forward_send_elems, dtype=cast_dtype, device=device
            ),
            "global_grad_recv": torch.empty(
                self._fsdp_global_forward_recv_elems, dtype=cast_dtype, device=device
            ),
            "full_g_bufs": [
                torch.empty(shape, dtype=cast_dtype, device=device)
                for shape in self._fsdp_target_shapes
            ],
        }
        """
        used for FSDP once mode, (all_to_all for all parameters)
        """
        workspace["global_upd_send"] = torch.empty(
            self._fsdp_global_reverse_send_elems, dtype=cast_dtype, device=device
        )
        workspace["global_upd_recv"] = torch.empty(
            self._fsdp_global_reverse_recv_elems, dtype=cast_dtype, device=device
        )
        return workspace

    def _allocate_fsdp_bucket_workspace(
        self,
        cast_dtype: torch.dtype,
        device: torch.device,
        need_to_calculate_norm: bool,
        skip_update: bool,
    ) -> dict[str, torch.Tensor | None]:
        """
        used for FSDP bucketed mode, (all_to_all for each bucket)
        """
        workspace: dict[str, torch.Tensor | None] = {
            "grad_send_flat": torch.empty(
                self._fsdp_max_bucket_send_elems, dtype=cast_dtype, device=device
            ),
            "grad_recv_flat": torch.empty(
                self._fsdp_max_bucket_recv_elems, dtype=cast_dtype, device=device
            ),
            "upd_send_flat": None,
            "upd_recv_flat": None,
            "param_send_flat": None,
            "param_recv_flat": None,
        }
        if not skip_update:
            workspace["upd_send_flat"] = torch.empty(
                self._fsdp_max_bucket_recv_elems, dtype=cast_dtype, device=device
            )
            workspace["upd_recv_flat"] = torch.empty(
                self._fsdp_max_bucket_send_elems, dtype=cast_dtype, device=device
            )
        if need_to_calculate_norm:
            workspace["param_send_flat"] = torch.empty(
                self._fsdp_max_bucket_send_elems, dtype=cast_dtype, device=device
            )
            workspace["param_recv_flat"] = torch.empty(
                self._fsdp_max_bucket_recv_elems, dtype=cast_dtype, device=device
            )
        return workspace

    def _create_fsdp_step_workspace(
        self, cast_dtype: torch.dtype, device: torch.device, skip_update: bool
    ) -> dict[str, object]:
        # Persistent workspace: created once, reused every step.
        # All view objects are pre-created once and remain valid because the underlying
        # workspace tensors never move (same Python objects across all steps).

        if self._fsdp_once_workspace_cache and self.persistent_cache_enabled:
            return self._fsdp_once_workspace_cache

        ws = self._allocate_fsdp_once_workspace(cast_dtype, device)

        grad_send = ws["global_grad_send"]
        grad_recv = ws["global_grad_recv"]
        full_g_bufs_ws = ws["full_g_bufs"]
        world_size = self._fsdp_world_size
        total_buckets = self._fsdp_total_buckets

        # Issue 1: dst views into global_grad_send for _prepare_fsdp_lmo.
        ws["pack_dst_views"] = [
            grad_send.narrow(0, base, numel).view(send_shape)
            for (_, base, numel, send_shape, _, _) in self._fsdp_pack_copy_plan
        ]

        # Issue 3a (flat): all full_g assembly in one _foreach_copy_ before LMO loop.
        full_g_prep_src_flat: list[torch.Tensor] = []
        full_g_prep_dst_flat: list[torch.Tensor] = []
        for b in range(total_buckets):
            full_g_flat = full_g_bufs_ws[b].view(-1)
            for src_base, dst_base, numel in self._fsdp_full_g_copy_plan[b]:
                full_g_prep_src_flat.append(grad_recv.narrow(0, src_base, numel))
                full_g_prep_dst_flat.append(full_g_flat.narrow(0, dst_base, numel))
        ws["full_g_prep_src_flat"] = full_g_prep_src_flat
        ws["full_g_prep_dst_flat"] = full_g_prep_dst_flat

        if skip_update:
            ws["global_upd_recv_cast"] = None
            ws["global_upd_recv_cast_by_dtype"] = None
            if self.persistent_cache_enabled:
                self._fsdp_once_workspace_cache = ws
            return ws

        upd_send = ws["global_upd_send"]
        upd_recv = ws["global_upd_recv"]

        # Issue 3b (flat): persistent u buffers + pre-created flat src views and
        # pre-created flat dst views into global_upd_send.
        ws["u_bufs"] = [
            torch.empty(shape, dtype=cast_dtype, device=device)
            for shape in self._fsdp_target_shapes
        ]
        u_src_views_flat: list[torch.Tensor] = []
        upd_pack_dst_views_flat: list[torch.Tensor] = []
        for b in range(total_buckets):
            recv_numels = self._fsdp_recv_numels[b]
            u_flat = ws["u_bufs"][b].view(-1)
            flat_offsets = self._fsdp_u_flat_offsets[b]
            for d in range(world_size):
                u_src_views_flat.append(
                    u_flat.narrow(0, flat_offsets[d], recv_numels[d])
                )
                upd_pack_dst_views_flat.append(
                    upd_send.narrow(
                        0,
                        self._fsdp_upd_send_slot_bases[b][d],
                        recv_numels[d],
                    )
                )
        ws["u_src_views_flat"] = u_src_views_flat
        ws["upd_pack_dst_views_flat"] = upd_pack_dst_views_flat
        if len(u_src_views_flat) != len(upd_pack_dst_views_flat):
            raise RuntimeError(
                "FSDP once-mode pack metadata mismatch: "
                f"src_views={len(u_src_views_flat)}, "
                f"dst_views={len(upd_pack_dst_views_flat)}."
            )

        target_dtype = self._fsdp_uniform_param_dtype
        if target_dtype is not None and target_dtype != cast_dtype:
            ws["global_upd_recv_cast"] = torch.empty(
                self._fsdp_global_reverse_recv_elems, dtype=target_dtype, device=device
            )
        else:
            ws["global_upd_recv_cast"] = None

        if target_dtype is None and self._fsdp_upd_recv_cast_numels_by_dtype:
            cast_by_dtype: dict[torch.dtype, torch.Tensor] = {}
            for dtype, numel in self._fsdp_upd_recv_cast_numels_by_dtype.items():
                if dtype == cast_dtype:
                    continue
                cast_by_dtype[dtype] = torch.empty(numel, dtype=dtype, device=device)
            ws["global_upd_recv_cast_by_dtype"] = (
                cast_by_dtype if cast_by_dtype else None
            )
        else:
            ws["global_upd_recv_cast_by_dtype"] = None

        # Issue 2a: src+dst views for per-dtype cast of global_upd_recv.
        cast_by_dtype_ws = ws.get("global_upd_recv_cast_by_dtype") or {}
        cast_src_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        cast_dst_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for dtype, cast_plan in self._fsdp_upd_recv_cast_plan_by_dtype.items():
            cast_flat = cast_by_dtype_ws.get(dtype)
            if cast_flat is None:
                continue
            cast_src_by_dtype[dtype] = [
                upd_recv.narrow(0, base, numel) for (base, numel, _) in cast_plan
            ]
            cast_dst_by_dtype[dtype] = [
                cast_flat.narrow(0, off, numel) for (_, numel, off) in cast_plan
            ]
        ws["cast_src_by_dtype"] = cast_src_by_dtype
        ws["cast_dst_by_dtype"] = cast_dst_by_dtype

        # Issue 2b: param views into update_source for global_updates extraction.
        cast_recv_uniform = ws.get("global_upd_recv_cast")
        update_source = cast_recv_uniform if cast_recv_uniform is not None else upd_recv
        upd_recv_views: list[torch.Tensor] = []
        for param_idx, base, numel, send_shape in self._fsdp_upd_recv_param_plan:
            if cast_by_dtype_ws:
                dtype = self._fsdp_param_local_dtypes[param_idx]
                cf = cast_by_dtype_ws.get(dtype)
                if cf is not None:
                    off = self._fsdp_upd_recv_cast_offset_by_param[param_idx]
                    upd_recv_views.append(cf.narrow(0, off, numel).view(send_shape))
                    continue
            upd_recv_views.append(update_source.narrow(0, base, numel).view(send_shape))
        ws["upd_recv_views"] = upd_recv_views

        if self.persistent_cache_enabled:
            self._fsdp_once_workspace_cache = ws
        return ws

    def _precompute_momentum_bufs(self):
        """
        Pre-build per-group momentum buffer lists so prepare_gradients_and_momentum
        avoids state dict lookups per param each step.
        """
        # List of (group_idx, buf_list, param_list). We keep all trainable params
        # so dynamic momentum schedules (e.g. 0.0 -> 0.9) are handled correctly.
        self._momentum_state: list[tuple[int, list, list]] = []
        for group_idx, group in enumerate(self.param_groups):
            bufs: list[torch.Tensor] = []
            params: list[torch.Tensor] = []
            for p in group["params"]:
                if p.requires_grad:
                    bufs.append(self._momentum_buffer_by_param_id[id(p)])
                    params.append(p)
            if bufs:
                self._momentum_state.append((group_idx, bufs, params))

        # Cache for per-step execution plan keyed by current momentum schedule.
        self._momentum_plan_key: tuple[float, ...] | None = None
        self._momentum_execution_plan: list[tuple[float, list, list]] = []

    def _precompute_pre_norm_metadata(self):
        """
        Pre-compute the col/mat pre-norm plan: which non-scalar params need
        the fused all-reduce pass (_apply_reduce_pre_norm_pass), grouped by
        (is_fsdp_row_sharded, exact pre_norm string, local_shard_shape, eps)
        so the pass can process each group with one torch.stack + batched op
        instead of a per-param loop -- keying on the exact pre_norm string
        (not just category) and eps keeps every batch's formula/eps uniform,
        since different groups can share a category/shape but use different
        eps or (in the future) different formulas within the same category.
        Row category needs no plan entry -- applied inline at fetch time
        (see get_momentum_or_grad/get_momentum_or_grad_list/
        _get_effective_grad_by_group).
        """
        self._pre_norm_reduce_shape_groups: dict[
            tuple[bool, str, tuple, float], list[tuple[torch.Tensor, int]]
        ] = {}
        self._pre_normed_grad_cache: dict[int, torch.Tensor] = {}
        self._fsdp_group = None
        if self.fsdp_enabled:
            fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
            if fsdp_mesh is not None:
                self._fsdp_group = fsdp_mesh.get_group()

        all_params = (
            list(self.embed_params)
            + list(self.ddp_params)
            + list(self.fsdp_params)
            + list(self.expert_params)
        )
        for p in all_params:
            group_idx = self.parameters_to_groups[id(p)]
            pre_norm = self.groups_pre_norm.get(group_idx, "identity")
            if pre_norm == "identity":
                continue
            category = pre_norm_category(pre_norm)
            if category not in ("row", "col", "mat"):
                raise ValueError(
                    f"Unknown pre_norm '{pre_norm}': category '{category}' "
                    "must be 'row', 'col', or 'mat'."
                )
            local_shape = tuple(self._param_local_views[id(p)].shape)
            if len(local_shape) < 2:
                # row/col/mat all assume a matrix shape [rows, cols] where
                # FSDP shards dim 0 (rows) and dim=-1 (cols) is a separate,
                # unsharded axis. For a genuinely 1-D param (e.g. a bias
                # vector, a real case -- see lmo()'s ndim==1 branch), dim=-1
                # *is* dim 0 *is* the sharded dim, so row's "dim=-1 is never
                # sharded" premise and col/mat's row-vs-column distinction
                # both break down. Fail fast instead of silently computing
                # a wrong reduction (e.g. reducing across stacked *different*
                # params instead of within one).
                raise ValueError(
                    f"pre_norm='{pre_norm}' needs a >=2-D parameter (row vs "
                    f"column/matrix axes aren't well-defined for a 1-D "
                    f"tensor) -- got shape {local_shape}. Use pre_norm="
                    f"'identity' for 1-D params (e.g. biases) for now."
                )
            if category == "row":
                continue  # applied inline, no plan entry needed
            sharded = _is_fsdp_row_sharded(p)
            eps = self.groups_pre_norm_eps[group_idx]
            key = (sharded, pre_norm, local_shape, eps)
            self._pre_norm_reduce_shape_groups.setdefault(key, []).append(
                (p, group_idx)
            )

    @record_function("disco.prepare_ddp_lmo")
    def _prepare_ddp_lmo(
        self, ddp_params, workspace, tp_mesh=None, tp_world_size=1
    ) -> list[torch.Tensor | None]:
        """
        Prepare per-owned LMO inputs for DDP:
          1) grouped get_momentum_or_grad_list by group_idx
          2) cast to communication dtype in batched foreach copies
          3) TP gather on cast dtype for TP-sharded tensors
        """
        cast_dtype: torch.dtype = workspace["cast_dtype"]
        n_params = len(ddp_params)
        lmo_inputs: list[torch.Tensor | None] = workspace["lmo_inputs_by_idx"]
        if len(lmo_inputs) != n_params:
            raise ValueError("DDP LMO input workspace size mismatch.")

        for param_idx in self._ddp_owned_indices:
            lmo_inputs[param_idx] = None

        # 1) grouped effective-grad fetch
        for group_idx, owned_indices, params_bucket in self._ddp_owned_indices_by_group:
            _, nesterov, momentum, _, _ = self.groups_info[group_idx]
            grads = self.get_momentum_or_grad_list(
                params_bucket,
                momentum,
                nesterov,
                group_idx,
                gather_to_local=False,
                to_local=False,
            )
            for j, param_idx in enumerate(owned_indices):
                lmo_inputs[param_idx] = grads[j]

        # 2) dtype cast staging
        non_tp_src_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        non_tp_idx_by_dtype: dict[torch.dtype, list[int]] = {}
        tp_src_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        tp_idx_by_dtype: dict[torch.dtype, list[int]] = {}
        tp_local_after_cast: dict[int, torch.Tensor] = {}

        for param_idx in self._ddp_owned_indices:
            g = lmo_inputs[param_idx]
            if g is None:
                continue

            tp_gather_info = (
                self._ddp_tp_gather_info_by_idx[param_idx]
                if tp_mesh is not None
                else None
            )
            if tp_gather_info is not None:
                g_local = g.to_local() if isinstance(g, DTensor) else g
                if g_local.dtype != cast_dtype:
                    tp_src_by_dtype.setdefault(g_local.dtype, []).append(g_local)
                    tp_idx_by_dtype.setdefault(g_local.dtype, []).append(param_idx)
                else:
                    tp_local_after_cast[param_idx] = g_local
                continue

            g_local = g.to_local() if isinstance(g, DTensor) else g
            if g_local.dtype != cast_dtype:
                non_tp_src_by_dtype.setdefault(g_local.dtype, []).append(g_local)
                non_tp_idx_by_dtype.setdefault(g_local.dtype, []).append(param_idx)
            else:
                lmo_inputs[param_idx] = g_local

        for src_dtype, src_views in non_tp_src_by_dtype.items():
            idxs = non_tp_idx_by_dtype[src_dtype]
            dst_views = [workspace["lmo_non_tp_cast_bufs_by_idx"][i] for i in idxs]
            if any(dst is None for dst in dst_views):
                raise ValueError(
                    "Missing non-TP cast buffers for DDP pre-LMO cast path."
                )
            torch._foreach_copy_(dst_views, src_views)
            for j, param_idx in enumerate(idxs):
                lmo_inputs[param_idx] = dst_views[j]

        if tp_mesh is not None:
            for src_dtype, src_views in tp_src_by_dtype.items():
                idxs = tp_idx_by_dtype[src_dtype]
                dst_views = [workspace["tp_local_cast_bufs_by_idx"][i] for i in idxs]
                if any(dst is None for dst in dst_views):
                    raise ValueError(
                        "Missing TP local cast buffers for DDP pre-LMO cast path."
                    )
                torch._foreach_copy_(dst_views, src_views)
                for j, param_idx in enumerate(idxs):
                    tp_local_after_cast[param_idx] = dst_views[j]

            for param_idx in self._ddp_owned_indices:
                if self._ddp_tp_gather_info_by_idx[param_idx] is None:
                    continue
                g_local = tp_local_after_cast.get(param_idx)
                if g_local is None:
                    lmo_inputs[param_idx] = None
                    continue
                lmo_inputs[param_idx] = gather_tp_shard(
                    g_local,
                    tp_mesh,
                    tp_world_size,
                    self._ddp_tp_gather_info_by_idx[param_idx],
                )

        return lmo_inputs

    @record_function("disco.prepare_experts_lmo_per_block")
    def _prepare_experts_lmo_per_block(
        self,
        expert_params,
        block_idx: int,
        big_g: torch.Tensor | None,
        dst_views_pre: list[torch.Tensor] | None = None,
    ) -> bool:
        start, end = self._expert_blocks[block_idx]
        block_params = expert_params[start:end]
        if not block_params or big_g is None:
            return False
        if dst_views_pre is None:
            raise ValueError(
                "dst_views_pre must be provided for expert fast fill path."
            )

        group_idx = self._expert_block_group_idx[block_idx]
        _, nesterov, momentum, _, _ = self.groups_info[group_idx]

        block_effective_grads = self.get_momentum_or_grad_list(
            block_params,
            momentum,
            nesterov,
            group_idx,
            to_local=True,
        )

        dst_views: list[torch.Tensor] = []
        src_views: list[torch.Tensor] = []
        any_grad = False
        for k, g_local in enumerate(block_effective_grads):
            if g_local is None:
                continue
            assert (
                g_local.ndim == 3
            ), "Batching path assumes MoE expert weights are 3-D."
            if g_local.shape[0] == 0:
                continue
            if k >= len(dst_views_pre):
                raise ValueError(
                    "Expert destination views are shorter than block parameter list."
                )
            dst_views.append(dst_views_pre[k])
            src_views.append(g_local)
            any_grad = True

        if dst_views:
            torch._foreach_copy_(dst_views, src_views)
        return any_grad

    @record_function("disco.prepare_experts_lmo")
    def _prepare_experts_lmo(
        self, expert_params, big_g_bufs, dst_views_by_block=None
    ) -> list[bool]:
        """Fill per-step expert big_g buffers from expert param gradients/momentum."""
        if big_g_bufs is None:
            big_g_bufs = self._allocate_expert_big_g_bufs()
        if dst_views_by_block is None:
            dst_views_by_block = self._create_expert_big_g_views(big_g_bufs)

        self._expert_any_grad: list[bool] = [False] * len(self._expert_blocks)
        for block_idx, (start, end) in enumerate(self._expert_blocks):
            block_params = expert_params[start:end]
            if not block_params:
                continue
            big_g = big_g_bufs[block_idx]
            if big_g is not None:
                big_g.zero_()
            block_dst_views = (
                dst_views_by_block[block_idx]
                if dst_views_by_block is not None
                and block_idx < len(dst_views_by_block)
                else None
            )
            self._expert_any_grad[block_idx] = self._prepare_experts_lmo_per_block(
                expert_params,
                block_idx,
                big_g,
                block_dst_views,
            )
        return self._expert_any_grad

    @record_function("disco.prepare_fsdp_lmo")
    def _prepare_fsdp_lmo(
        self,
        global_grad_send: torch.Tensor,
        pack_dst_views: list | None = None,
    ):
        """Fill step-local global_grad_send for all FSDP buckets."""
        dst_active: list[torch.Tensor] = []
        src_active: list[torch.Tensor] = []
        for i, (
            param_idx,
            base,
            numel,
            send_shape,
            tp_info,
            group_idx,
        ) in enumerate(self._fsdp_pack_copy_plan):
            p = self.fsdp_params[param_idx]
            g = self._get_effective_grad_by_group(p, group_idx, param_idx)
            g_local = self._maybe_unpack_dtensor(g, tp_info)
            dst_active.append(
                pack_dst_views[i]
                if pack_dst_views is not None
                else global_grad_send.narrow(0, base, numel).view(send_shape)
            )
            src_active.append(g_local)
        if dst_active:
            torch._foreach_copy_(dst_active, src_active)

    @record_function("disco.prepare_ddp_apply_updates")
    def _prepare_ddp_apply_updates(
        self, global_updates, workspace, tp_mesh=None
    ) -> list[torch.Tensor | None]:
        """
        Prepare update tensors for DDP apply:
          1) optional TP slice to local shape
          2) dtype cast via batched foreach_copy_ into preallocated typed buffers
        """
        apply_updates: list[torch.Tensor | None] = workspace["apply_updates"]
        n_params = len(self.ddp_params)
        if len(apply_updates) != n_params:
            raise ValueError("DDP apply workspace size mismatch.")

        # Stage 1: gather source views (with optional TP slice) into apply_updates.
        for param_idx in range(n_params):
            u = global_updates[param_idx]
            if u is None:
                apply_updates[param_idx] = None
                continue

            if tp_mesh is not None:
                tp_info = self._ddp_tp_slice_info_by_idx[param_idx]
                if tp_info is not None:
                    _, _, _, slicer, local_shape = tp_info
                    if tuple(u.shape) != tuple(local_shape):
                        u = u[slicer]

            apply_updates[param_idx] = u

        # Stage 2: batched cast by destination dtype (FSDP-style cast phase).
        cast_indices_by_dtype: dict[torch.dtype, list[int]] = workspace[
            "cast_indices_by_dtype"
        ]
        cast_dst_by_dtype: dict[torch.dtype, list[torch.Tensor]] = workspace[
            "cast_dst_by_dtype"
        ]
        for dtype, indices in cast_indices_by_dtype.items():
            if not indices:
                continue
            src_views: list[torch.Tensor] = []
            dst_views_active: list[torch.Tensor] = []
            active_indices: list[int] = []
            dst_views_all = cast_dst_by_dtype[dtype]
            for j, param_idx in enumerate(indices):
                src = apply_updates[param_idx]
                if src is None:
                    continue
                src_views.append(src)
                dst_views_active.append(dst_views_all[j])
                active_indices.append(param_idx)
            if not active_indices:
                continue
            torch._foreach_copy_(dst_views_active, src_views)
            for j, param_idx in enumerate(active_indices):
                apply_updates[param_idx] = dst_views_active[j]

        return apply_updates

    @record_function("disco.update_embed_params_fast")
    def _update_embed_params_fast(
        self,
        updates: list,
        big_us_by_shape: dict,
    ):
        """Apply embed param updates using pre-computed update plan.

        Fast-path extras (all grads present) are applied directly from
        big_us_by_shape: for each shape group, big_u.unbind(0) gives N views
        and _foreach_add_ applies them in one CUDA dispatch.

        Canonicals [0, 1] and partial-fallback extras are applied from
        updates[] via _embed_update_plan (same structure as expert / DDP plan).
        Fast-path extras have updates[i]=None and are skipped automatically.
        """
        # 1. Fast-path extras: big_u per shape group → _foreach_add_ (no updates[] lookup)
        for shape, big_u in big_us_by_shape.items():
            gidx = self._embed_shape_group_gidx[shape]
            lr, _, _, wd, _ = self.groups_info[gidx]
            p_locals = self._embed_shape_group_locals[shape]
            if wd != 0.0:
                torch._foreach_mul_(p_locals, 1.0 - wd * lr)
            torch._foreach_add_(p_locals, list(big_u.unbind(0)), alpha=-lr)

        # 2. Canonicals + partial fallback extras: read from updates[] via plan
        for (
            group_idx,
            _device,
            _dtype,
            param_indices,
            params_bucket,
            locals_bucket,
        ) in self._embed_update_plan:
            lr, _, _, wd, _ = self.groups_info[group_idx]

            aus_enabled = self.param_groups[group_idx]["aus_enabled"]
            active_locals: list[torch.Tensor] = []
            active_updates: list[torch.Tensor] = []
            for j, param_idx in enumerate(param_indices):
                u = updates[param_idx]
                if u is None:
                    continue  # fast-path extra already applied, or no grad

                p = params_bucket[j]
                p_local = locals_bucket[j]
                if p_local.shape != u.shape:
                    if isinstance(p, DTensor):
                        p_local = p.to_local()
                        self._param_local_views[id(p)] = p_local
                        locals_bucket[j] = p_local
                    if p_local.shape != u.shape:
                        raise ValueError(
                            "Shape mismatch between embed parameter shard and update."
                        )

                if u.dtype != p_local.dtype:
                    u = u.to(p_local.dtype)

                active_locals.append(p_local)
                active_updates.append(u)

            if not active_locals:
                continue

            if aus_enabled:
                torch._foreach_add_(active_locals, active_updates, alpha=-1.0)
            else:
                if wd != 0.0:
                    torch._foreach_mul_(active_locals, 1.0 - wd * lr)
                torch._foreach_add_(active_locals, active_updates, alpha=-lr)

    @record_function("disco.update_ddp_params_fast")
    def _update_ddp_params_fast(self, apply_updates):
        if len(apply_updates) != len(self.ddp_params):
            raise ValueError("DDP updates length mismatch with parameter list.")

        for (
            group_idx,
            _device,
            _dtype,
            param_indices,
            params_bucket,
            locals_bucket,
        ) in self._ddp_update_plan:
            lr, _, _, wd, _ = self.groups_info[group_idx]
            aus_enabled = self.param_groups[group_idx]["aus_enabled"]

            active_locals: list[torch.Tensor] = []
            active_updates: list[torch.Tensor] = []
            for j, param_idx in enumerate(param_indices):
                u = apply_updates[param_idx]
                if u is None:
                    continue

                p = params_bucket[j]
                p_local = locals_bucket[j]
                if p_local.shape != u.shape:
                    if isinstance(p, DTensor):
                        p_local = p.to_local()
                        self._param_local_views[id(p)] = p_local
                        locals_bucket[j] = p_local
                        self._ddp_param_local_shapes[param_idx] = tuple(p_local.shape)
                    if p_local.shape != u.shape:
                        raise ValueError(
                            "Shape mismatch between DDP parameter shard and update."
                        )

                if u.dtype != p_local.dtype:
                    raise ValueError(
                        "Unexpected DDP update dtype mismatch after cast-prep: "
                        f"update={u.dtype}, param_local={p_local.dtype}."
                    )

                active_locals.append(p_local)
                active_updates.append(u)

            if not active_locals:
                continue

            if aus_enabled:
                torch._foreach_add_(active_locals, active_updates, alpha=-1.0)
            else:
                if wd != 0.0:
                    torch._foreach_mul_(active_locals, 1.0 - wd * lr)
                torch._foreach_add_(active_locals, active_updates, alpha=-lr)

    @record_function("disco.update_expert_params_fast")
    def _update_expert_params_fast(self, expert_params, all_updates):
        if len(all_updates) != len(expert_params):
            raise ValueError(
                "Expert updates length mismatch with expert parameter list."
            )

        for (
            group_idx,
            _device,
            _dtype,
            param_indices,
            params_bucket,
            locals_bucket,
        ) in self._expert_update_plan:
            lr, _, _, wd, _ = self.groups_info[group_idx]

            active_locals: list[torch.Tensor] = []
            active_updates: list[torch.Tensor] = []
            for j, param_idx in enumerate(param_indices):
                u = all_updates[param_idx]
                if u is None:
                    continue

                p = params_bucket[j]
                p_local = locals_bucket[j]
                if p_local.shape != u.shape:
                    if isinstance(p, DTensor):
                        p_local = p.to_local()
                        self._param_local_views[id(p)] = p_local
                        locals_bucket[j] = p_local
                    if p_local.shape != u.shape:
                        raise ValueError(
                            "Shape mismatch between expert parameter shard and update."
                        )

                if u.dtype != p_local.dtype:
                    u = u.to(p_local.dtype)

                active_locals.append(p_local)
                active_updates.append(u)

            if not active_locals:
                continue

            if wd != 0.0:
                torch._foreach_mul_(active_locals, 1.0 - wd * lr)
            torch._foreach_add_(active_locals, active_updates, alpha=-lr)

    def _refresh_momentum_execution_plan(self):
        """
        Rebuild foreach execution plan only when per-group momentum values change.
        Plan buckets tensors by (device, dtype, momentum) to maximize foreach fusion.
        """
        if not self._momentum_state:
            self._momentum_execution_plan = []
            self._momentum_plan_key = tuple()
            return

        plan_key = tuple(
            float(self.param_groups[group_idx]["momentum"])
            for group_idx, _, _ in self._momentum_state
        )
        if plan_key == self._momentum_plan_key:
            return

        buckets: dict[tuple[torch.device, torch.dtype, float], dict[str, list]] = {}
        for (group_idx, bufs, params), m in zip(self._momentum_state, plan_key):
            if not (0.0 < m < 1.0):
                continue
            for buf, p in zip(bufs, params):
                key = (buf.device, buf.dtype, m)
                if key not in buckets:
                    buckets[key] = {"bufs": [], "params": []}
                buckets[key]["bufs"].append(buf)
                buckets[key]["params"].append(p)

        self._momentum_execution_plan = [
            (m, data["bufs"], data["params"]) for (_, _, m), data in buckets.items()
        ]
        self._momentum_plan_key = plan_key

    def _maybe_unpack_dtensor(self, x, tp_info):
        if isinstance(x, DTensor):
            x_local = x.to_local()
            if tp_info is not None:
                _, tp_world_size, tp_mesh = tp_info
                x_local = gather_tp_shard(x_local, tp_mesh, tp_world_size, x.placements)
            return x_local
        return x

    def _apply_row_pre_norm(self, g, group_idx):
        """
        Applies row-category pre-norm directly to `g` (a DTensor, still
        possibly row-sharded, or a plain Tensor) with zero communication --
        dim=-1 is never the FSDP-sharded dimension, so this dispatches as a
        local per-shard op. No-op for "identity" or col/mat (those go
        through the batched _apply_reduce_pre_norm_pass instead).

        Known limitation: if a param is TP-*column*-sharded (Shard(dim=1),
        splitting dim=-1 across TP ranks -- see tp_axis's "col-parallel"
        case), a local shard only has part of each row, so this would be
        approximate. TP composition is out of scope for this pass (same
        scoping decision as the FSDP+TP case for col/mat).
        """
        if g is None:
            return g
        pre_norm = self.groups_pre_norm.get(group_idx, "identity")
        if pre_norm == "identity" or pre_norm_category(pre_norm) != "row":
            return g
        return PRE_NORM_ROW_FUNCTIONS[pre_norm](g, self.groups_pre_norm_eps[group_idx])

    def _get_effective_grad_by_group(self, p, group_idx, param_idx):
        """
        Fast path for effective grad retrieval in FSDP packing.
        Avoids state dict lookups and the generic helper call overhead.
        """
        # pop, not get: each param is fetched at most once per step via a
        # non-gather_to_local call, so releasing the cache entry immediately
        # (rather than holding it until next step's cache reset) caps how
        # long the extra pre-normed copy stays alive -- see _apply_reduce_
        # pre_norm_pass's docstring for the memory-cost discussion.
        cached = self._pre_normed_grad_cache.pop(id(p), None)
        if cached is not None:
            return cached

        g = p.grad
        if not p.requires_grad:
            p_name = (
                self.fsdp_param_names[param_idx]
                if param_idx < len(self.fsdp_param_names)
                else f"param_idx={param_idx}"
            )
            raise RuntimeError(
                f"[DiSCO] {p_name}. Does not require grad, why do you put it in the param_groups?"
            )

        if g is None:
            p_name = (
                self.fsdp_param_names[param_idx]
                if param_idx < len(self.fsdp_param_names)
                else f"param_idx={param_idx}"
            )
            raise RuntimeError(
                "[DiSCO] FSDP grad pack encountered missing gradient: "
                f"{p_name}. This optimizer expects all FSDP params to have grads."
            )

        _, nesterov, momentum, _, _ = self.groups_info[group_idx]
        use_momentum = (not self.is_light) and (0.0 < momentum < 1.0)
        if not use_momentum:
            return self._apply_row_pre_norm(g, group_idx)

        buf = self._momentum_buffer_by_param_id.get(id(p))
        if buf is None:
            raise ValueError(
                "Momentum buffer missing; ensure pre-pass ran before FSDP packing."
            )
        g = buf if not nesterov else torch.lerp(buf, g, momentum)
        return self._apply_row_pre_norm(g, group_idx)

    @record_function("disco.step")
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # only refresh schedulables each step (unchanged behaviour)
        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            nesterov = group["nesterov"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            param_kwargs = {
                "eps": group["eps"],
                "norm_factor": group["norm_factor"],
                "zeropower_backend": group["backend"],
                "backend_steps": group["backend_steps"],
                "splits_into": group["splits_into"],
                "splits_dim": group["splits_dim"],
            }
            self.groups_info[group_idx] = [lr, nesterov, momentum, wd, param_kwargs]
            self.groups_pre_norm[group_idx] = group.get("pre_norm", "identity")
            self.groups_pre_norm_eps[group_idx] = group["eps"]

        self.prepare_gradients_and_momentum()
        self._apply_reduce_pre_norm_pass()

        # gram_level can change between steps (calculate_norm_at_next_step),
        # unlike the mostly-static structural metadata above -- cheap sentinel
        # check, only re-derives the 3 gram-vector offset tables (not the
        # full FSDP/DDP/experts precompute) when it actually changed.
        if self.gram_level != self._gram_level_at_last_vector_precompute:
            self._precompute_fsdp_gram_vector_metadata()
            self._precompute_experts_gram_vector_metadata()
            self._precompute_ddp_gram_vector_metadata()
            self._gram_level_at_last_vector_precompute = self.gram_level

        fsdp_workspace = None
        if self.fsdp_params and self.fsdp_a2a_mode == "once":
            fsdp_workspace = self._create_fsdp_step_workspace(
                cast_dtype=self.communication_dtype,
                device=self.fsdp_params[0].device,
                skip_update=False,
            )

        expert_workspace = (
            self._create_expert_step_workspace() if self.expert_params else None
        )
        ddp_workspace = None
        if self.ddp_params:
            ddp_workspace = (
                self._create_ddp_step_workspace(
                    cast_dtype=self.communication_dtype,
                    device=self.ddp_params[0].device,
                )
                if self.ddp_params
                else None
            )

        embed_workspace = (
            self._create_embed_step_workspace(device=self.embed_params[0].device)
            if self.embed_params
            else None
        )

        deferred_embed_updates = None
        if self.embed_params:
            deferred_embed_updates = self.step_embedding(
                self.embed_params,
                self.embed_param_names,
                embed_workspace,
                defer_update=self.aus_enabled,
            )

        # Expert LMO launches are coordinated inside step_experts.
        if self.expert_params:
            self.step_experts(
                self.expert_params,
                self.expert_param_names,
                workspace=expert_workspace,
            )

        if self.fsdp_params:
            self.step_fsdp(
                self.fsdp_params,
                self.fsdp_param_names,
                workspace=fsdp_workspace,
            )

        if self.ddp_params:
            self.step_ddp(
                self.ddp_params,
                self.ddp_param_names,
                workspace=ddp_workspace,
            )

        # In AUS mode, both embedding and dense matrix validation must
        # succeed before any weights are applied. step_ddp validates its
        # entire owner batch (across ranks) before applying dense updates.
        if deferred_embed_updates is not None:
            self._update_embed_params_fast(*deferred_embed_updates)

        if self.scale_params:
            self.step_scalar(self.scale_params, self.scale_param_names)

        self.need_to_calculate_norm = False
        return loss

    @record_function("disco.step_scalar")
    @torch.compile()
    def step_scalar(
        self,
        scalar_params,
        scalar_param_names,
        skip_update=False,
    ):
        """
        We hardcode the update for scalar parameters to be the `sign` of the gradient.
        """
        if not scalar_params:
            return

        updates = []
        for p in scalar_params:
            group_idx = self.parameters_to_groups[id(p)]
            _, nesterov, momentum, _, _ = self.groups_info[group_idx]
            g = self.get_momentum_or_grad(p, momentum, nesterov, group_idx)

            if g is None:
                updates.append(None)
                continue

            g_local = g.to_local() if isinstance(g, DTensor) else g
            u = torch.sign(g_local)
            updates.append(u)

        if not skip_update:
            # Scalar parameters are not TP-sharded.
            self.update_bucket_params(
                scalar_params, updates, 0, len(scalar_params), tp_mesh=None
            )

        if not self.need_to_calculate_norm:
            return

        final_norms = {}
        for i, p in enumerate(scalar_params):
            p_local = p.to_local() if isinstance(p, DTensor) else p
            cleaned_p_name = remove_orig_mod_and_weight_for_p_name(
                scalar_param_names[i]
            )
            # The original code only logs the parameter's absolute value, as the
            # update norm is constant (learning_rate * 1.0).
            final_norms[f"scalar_param_supremum/{cleaned_p_name}"] = p_local.abs()

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_embedding")
    def step_embedding(
        self,
        embed_params,
        embed_param_names,
        workspace,
        skip_update=False,
        *,
        defer_update=False,
    ):
        # Reuse pre-allocated Python lists (reset before use)
        effective_grads: list = workspace["effective_grads"]
        updates: list = workspace["updates"]
        for i in range(len(effective_grads)):
            effective_grads[i] = None
        for i in range(len(updates)):
            updates[i] = None

        tp_mesh = (
            self.parallel_dims.get_optional_mesh("tp") if self.tp_enabled else None
        )

        batched_g_bufs = workspace["batched_g_bufs"]
        buf_views = workspace["buf_views_by_shape"]

        # ===== PREPARE: fetch all grads + fill big_g buffers =====

        # Canonical params [0, 1] — per-param fetch (preserves TP edge-case handling)
        for i in range(min(2, len(embed_params))):
            p = embed_params[i]
            group_idx = self.parameters_to_groups[id(p)]
            _, nesterov, momentum, _, _ = self.groups_info[group_idx]
            g = self.get_momentum_or_grad(
                p, momentum, nesterov, group_idx, gather_to_local=False
            )
            if g is not None and not self.fsdp_enabled and self.tp_enabled:
                # Edge case: TP-only (no FSDP) — grad is Replicate, weight is sharded
                original_placements = p.placements
                tp_mesh_dim = tp_axis(original_placements, True)
                tp_sharded_dim = original_placements[tp_mesh_dim].dim
                chunk_size = p.to_local().shape[tp_sharded_dim]
                start_offset = tp_mesh.get_local_rank() * chunk_size
                slicer = [slice(None)] * g.dim()
                slicer[tp_sharded_dim] = slice(start_offset, start_offset + chunk_size)
                g = g[tuple(slicer)]
            effective_grads[i] = g

        # Extra params [2+] — batched fetch grouped by group_idx
        for gidx, indices, params_in_group in self._embed_extra_by_group:
            _, nesterov, momentum, _, _ = self.groups_info[gidx]
            grads = self.get_momentum_or_grad_list(
                params_in_group,
                momentum,
                nesterov,
                gidx,
                gather_to_local=False,
                to_local=True,
            )
            for j, i in enumerate(indices):
                effective_grads[i] = grads[j]

        # Fill big_g buffers for all-valid shape groups — like expert _prepare_experts_lmo.
        # Fill happens BEFORE any LMO so GPU memory writes complete before compute.
        filled_shapes: set[tuple] = set()
        for shape, indices in self._embed_extra_shape_groups.items():
            if shape not in batched_g_bufs:
                continue
            if all(effective_grads[i] is not None for i in indices):
                torch._foreach_copy_(
                    buf_views[shape], [effective_grads[i] for i in indices]
                )
                filled_shapes.add(shape)

        # ===== LMO =====

        # Canonical LMO [0, 1] — per-param (unique shapes, no batching)
        for i in range(min(2, len(embed_params))):
            g = effective_grads[i]
            if g is None:
                continue
            *_, param_kwargs = self.groups_info[
                self.parameters_to_groups[id(embed_params[i])]
            ]
            updates[i] = self.lmo(g, **param_kwargs)

        # Extra LMO by shape group — use pre-filled big_g or fallback to stack
        big_us_by_shape: dict[tuple, torch.Tensor] = {}
        for shape, indices in self._embed_extra_shape_groups.items():
            gidx = self._embed_shape_group_gidx[shape]
            *_, param_kwargs = self.groups_info[gidx]

            if shape in filled_shapes:
                # All-valid fast path: lmo on pre-filled buffer (no allocation).
                # Result collected in big_us_by_shape; applied by _update_embed_params_fast.
                big_us_by_shape[shape] = self.lmo(batched_g_bufs[shape], **param_kwargs)
            else:
                # Partial grads fallback — collect valid, stack, lmo, store in updates[]
                valid_pairs = [
                    (i, effective_grads[i])
                    for i in indices
                    if effective_grads[i] is not None
                ]
                if not valid_pairs:
                    continue
                valid_idx, valid_grads = zip(*valid_pairs)
                if len(valid_grads) >= 2:
                    big_u = self.lmo(torch.stack(list(valid_grads)), **param_kwargs)
                    u_views = big_u.unbind(0)
                    for k, i in enumerate(valid_idx):
                        updates[i] = u_views[k]
                else:
                    updates[valid_idx[0]] = self.lmo(valid_grads[0], **param_kwargs)

        aus_corrections: list[torch.Tensor | None] = [None] * len(embed_params)
        aus_etas: list[torch.Tensor | None] = [None] * len(embed_params)
        aus_validity: list[torch.Tensor | None] = [None] * len(embed_params)
        aus_finiteness: list[torch.Tensor | None] = [None] * len(embed_params)
        if self.aus_enabled:
            # Batched embedding LMOs normally bypass updates[]. Materialize only
            # AUS-enabled shape groups so each matrix gets its own correction.
            for shape in list(big_us_by_shape):
                group_idx = self._embed_shape_group_gidx[shape]
                if not self.param_groups[group_idx]["aus_enabled"]:
                    continue
                indices = self._embed_extra_shape_groups[shape]
                for param_idx, u in zip(
                    indices, big_us_by_shape[shape].unbind(0), strict=True
                ):
                    updates[param_idx] = u
                del big_us_by_shape[shape]

            for i, p in enumerate(embed_params):
                group_idx = self.parameters_to_groups[id(p)]
                if not self.param_groups[group_idx]["aus_enabled"]:
                    continue
                (
                    displacement,
                    aus_corrections[i],
                    aus_etas[i],
                    aus_validity[i],
                    aus_finiteness[i],
                ) = self._build_aus_displacement(p, updates[i], group_idx)
                updates[i] = displacement
            self._raise_if_any_aus_nonfinite(
                aus_finiteness, device=embed_params[0].device
            )

        #  Norm/Gram Calculation (on full tensors for correctness). Runs
        # BEFORE the real update is applied (see below) so `p` here is
        # genuinely pre-update -- needed so calculate_gram_metrics gets W and
        # U simultaneously (see gram_helper.py). `U` is the RAW moment `g`
        # (already in scope, fetched just above, before `self.lmo`), not the
        # LMO-processed `u` -- see readme.md. track_param_* keeps its
        # historical post-update meaning via a cheap local pseudo-weight
        # (_pseudo_post_update_weight) instead of re-reading a real
        # post-update p, which would need a second, redundant full_tensor().
        if self.need_to_calculate_norm:
            final_norms = {}
            norm_scratch: list = workspace["norm_scratch"]

            for i, (p, p_name) in enumerate(zip(embed_params, embed_param_names)):
                # `p` gets reassigned to `p.full_tensor()` below (needed for
                # norm/gram on the full tensor) -- capture the ORIGINAL
                # Parameter's id now, since _radial_state_by_param_id is
                # keyed by that (built once in _precompute_runtime_caches),
                # not by whatever `p` refers to after the reassignment.
                original_pid = id(p)
                group_idx = self.parameters_to_groups[id(p)]
                lr, nesterov, momentum, wd, param_kwargs = self.groups_info[group_idx]
                aus_for_group = self.param_groups[group_idx]["aus_enabled"]

                # Gather full tensor for norm calculation
                g = self.get_momentum_or_grad(
                    p, momentum, nesterov, group_idx, gather_to_local=True
                )
                u = self.lmo(g, **param_kwargs)

                need_T = CONST_NAME_OF_EMBEDDING in p_name

                # Use pre-alloc float32 scratch to avoid -lr*u temp allocation (extras only)
                scratch = norm_scratch[i]
                applied_lr = aus_etas[i] if aus_for_group else lr
                if applied_lr is None:
                    applied_lr = lr
                if scratch is not None and scratch.shape == u.shape:
                    torch.mul(u, -applied_lr, out=scratch)
                    upd_norms = calculate_norm(
                        scratch, self.norms_to_log, transpose=need_T
                    )
                else:
                    upd_norms = calculate_norm(
                        -applied_lr * u, self.norms_to_log, transpose=need_T
                    )
                upd_spectrum = upd_norms.pop("spectrum")

                # Gather the parameter itself to a full tensor. The real
                # update hasn't been applied yet at this point, so this is
                # genuinely the pre-update weight.
                if isinstance(p, DTensor):
                    p = p.full_tensor()

                if aus_for_group:
                    displacement = updates[i]
                    if displacement is None:
                        displacement = torch.zeros_like(p)
                    pseudo_w = p - displacement
                else:
                    pseudo_w = _pseudo_post_update_weight(p, u, lr, wd)
                wnorm = calculate_norm(pseudo_w, self.norms_to_log, transpose=need_T)
                w_spectrum = wnorm.pop("spectrum")

                # embed_params includes the output/lm_head weight, shape
                # [vocab_size, hidden_dim] -- calculate_gram_metrics now
                # always orients rows <= cols internally (see
                # gram_helper.py's orientation-policy docstring), so this no
                # longer OOMs (previously formed a [vocab_size, vocab_size]
                # matrix, since `need_T` only transposed by param name, not
                # by shape). It's still far more expensive than any other
                # tracked param, though -- tens of GB and several seconds of
                # fp32 GEMM at large vocab sizes -- so it's independently
                # gated by `track_embed_gram` (DISCO_TRACK_EMBED_GRAM),
                # separate from gram_level, see readme.md.
                gram_metrics = (
                    calculate_gram_metrics(p, g, pseudo_w, level=self.gram_level)
                    if self.track_embed_gram
                    else {}
                )
                # Whole-tensor radial-dynamics metrics -- always computed,
                # independent of gram_level/norms_to_log (see
                # radial_helper.py); reuses the exact (p, pseudo_w) pair
                # already used as gram's (W_before, W_after).
                radial_w = p.transpose(0, 1) if need_T else p
                radial_pseudo_w = (
                    pseudo_w.transpose(0, 1) if need_T else pseudo_w
                )
                radial_metrics = calculate_radial_metrics(
                    radial_w, radial_pseudo_w, self._radial_state_by_param_id[original_pid]
                )

                cleaned_p_name = remove_orig_mod_and_weight_for_p_name(p_name)
                gram_p_name = _gram_log_param_name(cleaned_p_name, tuple(p.shape))
                for norm_name in self.norms_to_log:
                    final_norms[
                        f"track_update_{norm_name}/{cleaned_p_name}"
                    ] = upd_norms[norm_name]
                    final_norms[f"track_param_{norm_name}/{cleaned_p_name}"] = wnorm[
                        norm_name
                    ]
                if aus_for_group:
                    correction = aus_corrections[i]
                    eta = aus_etas[i]
                    valid = aus_validity[i]
                    if (
                        correction is not None
                        and eta is not None
                        and valid is not None
                    ):
                        final_norms[
                            f"track_aus_correction/{cleaned_p_name}"
                        ] = correction
                        final_norms[f"track_aus_eta/{cleaned_p_name}"] = eta
                        final_norms[f"track_aus_valid/{cleaned_p_name}"] = (
                            valid.to(torch.float32)
                        )
                for gram_name, val in gram_metrics.items():
                    final_norms[f"track_gram_{gram_name}/{gram_p_name}"] = val
                for radial_name, val in radial_metrics.items():
                    final_norms[f"track_radial_{radial_name}/{cleaned_p_name}"] = val
                # This path already operates on fully-materialized local tensors
                # (no FSDP/EP sharding survives to this point), so the spectrum is
                # already complete locally — no extra collective is needed.
                final_norms[f"track_spectrum_update/{cleaned_p_name}"] = upd_spectrum
                final_norms[f"track_spectrum_param/{cleaned_p_name}"] = w_spectrum

            if self.is_dp_rank_0:
                self.norms_at_current_step.update(final_norms)

        # ===== UPDATE =====
        # Fast-path extras applied via big_us_by_shape (_foreach_add_ with unbind).
        # Canonicals [0,1] + partial fallback extras applied via update_plan.
        # Runs AFTER norm/gram calculation (moved from before it) so that the
        # `p`/`p.full_tensor()` reads above see genuinely pre-update weights.
        # Unconditional on need_to_calculate_norm -- the update must happen
        # every step regardless of whether norm-logging ran this step.
        if not skip_update:
            if defer_update:
                return updates, big_us_by_shape
            self._update_embed_params_fast(updates, big_us_by_shape)

    @record_function("disco.step_experts")
    def step_experts(
        self,
        expert_params,
        expert_param_names,
        workspace,
        skip_update=False,
    ):
        (expert_big_g_bufs, expert_dst_views_by_block) = (
            workspace["big_g_bufs"],
            workspace["dst_views_by_block"],
        )

        need_to_calculate_norm = self.need_to_calculate_norm

        norms_of_update, norms_of_weight, norms_of_gram, final_norms = [], [], [], {}
        # Radial-dynamics metrics (radial_helper.py) -- always computed,
        # independent of gram_level.
        norms_of_radial = []

        device = expert_params[0].device
        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        world_size, local_rank = fsdp_mesh.size(), fsdp_mesh.get_local_rank()

        # Use pre-computed structural metadata from init (avoids per-step recomputation).
        ep_per_rank = self._expert_ep_per_rank
        kinds_of_norms = self._expert_kinds_of_norms
        transpose = self._expert_transpose
        blocks = self._expert_blocks

        padding_norms = self._get_cached_zero_scalar(device)

        # Fill all expert big_g buffers once before block LMO loop.

        self._prepare_experts_lmo(
            expert_params,
            expert_big_g_bufs,
            dst_views_by_block=expert_dst_views_by_block,
        )

        all_updates: list[torch.Tensor | None] = [None] * len(expert_params)
        # Raw pre-LMO moment per param, kept alive for gram_metrics' `U`
        # argument (which must be the raw effective grad/momentum, not the
        # LMO-processed update -- see readme.md). Captured straight from
        # `big_g` (the exact tensor `self.lmo` is about to be called on),
        # before the call -- `lmo()` never mutates its input in place, so
        # this is a free alias, not a new allocation.
        all_raw_grads: list[torch.Tensor | None] = [None] * len(expert_params)

        for block_idx, (start, end) in enumerate(blocks):
            block_params = expert_params[start:end]
            if not block_params:
                continue

            # Pre-stored structural lmo kwargs (eps/norm_factor/backend…); lr/wd excluded.
            kwargs0 = self._expert_kwargs_per_block[block_idx]
            if kwargs0 is None:
                continue

            K = end - start

            any_grad = (
                self._expert_any_grad[block_idx]
                if block_idx < len(self._expert_any_grad)
                else False
            )

            big_g = expert_big_g_bufs[block_idx]  # [K*ep_per_rank, A, B]

            if not any_grad or big_g is None:
                all_updates[start:end] = [None] * K
                continue

            all_raw_grads[start:end] = big_g.view(K, ep_per_rank, *big_g.shape[1:])

            lmo_label = f"disco.expert_lmo_block{block_idx}"
            with record_function(lmo_label):
                big_u = self.lmo(big_g, **kwargs0, transpose_experts=transpose)
            all_updates[start:end] = big_u.view(K, ep_per_rank, *big_u.shape[1:])

        # Singular-value spectrum: one flat per-rank buffer covering ALL blocks
        # (static offsets precomputed in _precompute_experts_metadata), pre-zeroed
        # so skipped/padding slots are correct for free. Every rank contributes the
        # same shape here (ep_per_rank is uniform across ranks, unlike fsdp/ddp's
        # diagonal ownership), so this can be gathered with a single collective
        # together with the scalar norms below — no pad-to-max scheme needed.
        update_spectrum_flat = (
            torch.zeros(
                self._expert_spectrum_total_size, dtype=torch.float32, device=device
            )
            if need_to_calculate_norm and self._expert_spectrum_total_size > 0
            else None
        )
        weight_spectrum_flat = (
            torch.zeros(
                self._expert_spectrum_total_size, dtype=torch.float32, device=device
            )
            if update_spectrum_flat is not None
            else None
        )
        # Gram vectors: one flat per-rank buffer covering ALL blocks, same
        # "uniform per rank, no pad-to-max" reasoning as the spectrum buffers
        # above -- static offsets from _precompute_experts_gram_vector_metadata.
        n_vec = len(self.gram_vector_names)
        gram_vec_flat = (
            torch.zeros(
                self._expert_gram_vec_total_size, dtype=torch.float32, device=device
            )
            if need_to_calculate_norm and self._expert_gram_vec_total_size > 0
            else None
        )

        if need_to_calculate_norm:
            for block_idx, (start, end) in enumerate(blocks):
                block_params = expert_params[start:end]
                block_updates = all_updates[start:end]
                block_raw_grads = all_raw_grads[start:end]
                if not block_params or not block_updates:
                    continue
                block_base = self._expert_spectrum_block_offset[block_idx]
                K_block = self._expert_spectrum_len_per_block[block_idx]
                block_base_vec = self._expert_gram_vec_block_offset[block_idx]
                K_block_vec = self._expert_gram_vec_len_per_block[block_idx]
                # lr/wd are per-group (== per-block here, since one block ==
                # one param group), not per-expert -- fetched once per block,
                # used to build each expert's pseudo_w below. Mirrors the
                # other 3 step_* paths, which already do this; step_experts
                # previously used the raw pre-update p_local[ep_idx] directly
                # for weight-norm/gram (a known inconsistency, now fixed).
                lr, _, _, wd, _ = self.groups_info[
                    self._expert_block_group_idx[block_idx]
                ]
                local_pos = 0
                for p, u, g_raw in zip(block_params, block_updates, block_raw_grads):
                    if u is None:
                        continue
                    assert u.ndim == 3
                    p_local = p.to_local() if isinstance(p, DTensor) else p
                    for ep_idx in range(u.shape[0]):
                        update_norms = calculate_norm(
                            u[ep_idx], self.norms_to_log, transpose=transpose
                        )
                        upd_spec = update_norms.pop("spectrum")
                        norms_of_update.extend(update_norms.values())
                        if update_spectrum_flat is not None:
                            off = block_base + local_pos * K_block
                            update_spectrum_flat[off : off + K_block].copy_(upd_spec)

                        # Post-update pseudo-weight (cheap elementwise
                        # approximation of the real apply formula), same as
                        # step_embedding/step_ddp/step_fsdp -- used for both
                        # track_param_* (historical post-update meaning) and
                        # as gram's W_after argument below.
                        pseudo_w = _pseudo_post_update_weight(
                            p_local[ep_idx], u[ep_idx], lr, wd
                        )
                        weight_norms = calculate_norm(
                            pseudo_w,
                            self.norms_to_log,
                            transpose=transpose,
                        )
                        w_spec = weight_norms.pop("spectrum")
                        norms_of_weight.extend(weight_norms.values())
                        if weight_spectrum_flat is not None:
                            off = block_base + local_pos * K_block
                            weight_spectrum_flat[off : off + K_block].copy_(w_spec)

                        # p_local[ep_idx] (pre-update), g_raw[ep_idx] (raw
                        # pre-LMO moment), and pseudo_w (post-update) are all
                        # already in scope here -- the real update is applied
                        # later, at `_update_expert_params_fast` below -- no
                        # reordering needed for this path. Always called --
                        # cheap no-op when gram_level==0 (see gram_helper.py).
                        gram_metrics = calculate_gram_metrics(
                            p_local[ep_idx],
                            g_raw[ep_idx],
                            pseudo_w,
                            level=self.gram_level,
                            # transpose=transpose,
                        )
                        for name in self.gram_scalar_names:
                            norms_of_gram.append(gram_metrics[name])
                        if gram_vec_flat is not None and K_block_vec > 0:
                            base = block_base_vec + local_pos * K_block_vec * n_vec
                            for vi, vname in enumerate(self.gram_vector_names):
                                voff = base + vi * K_block_vec
                                gram_vec_flat[voff : voff + K_block_vec].copy_(
                                    gram_metrics[vname]
                                )

                        # `p` is never reassigned in this loop (only the
                        # derived `p_local` is), so id(p) is safe to use
                        # directly -- unlike step_embedding/step_ddp/
                        # step_fsdp, which reassign their weight variable in
                        # place and need to capture the id beforehand.
                        # Per-expert accumulator state: radial_state's
                        # tensors are shaped (num_local_experts,) for 3-D
                        # expert params (see the lazy-init above), so index
                        # by ep_idx to get this expert's own 0-d views.
                        radial_state_for_p = self._radial_state_by_param_id[id(p)]
                        radial_w = (
                            p_local[ep_idx].transpose(0, 1)
                            if transpose
                            else p_local[ep_idx]
                        )
                        radial_pseudo_w = (
                            pseudo_w.transpose(0, 1) if transpose else pseudo_w
                        )
                        radial_metrics = calculate_radial_metrics(
                            radial_w,
                            radial_pseudo_w,
                            {k: v[ep_idx] for k, v in radial_state_for_p.items()},
                        )
                        for name in RADIAL_METRIC_NAMES:
                            norms_of_radial.append(radial_metrics[name])
                        local_pos += 1

        if not skip_update:
            if any(u is not None for u in all_updates):
                self._update_expert_params_fast(expert_params, all_updates)

        if need_to_calculate_norm:
            expected_total = len(expert_params) * ep_per_rank * kinds_of_norms
            pad_needed = expected_total - len(norms_of_update)
            if pad_needed > 0:
                norms_of_update.extend([padding_norms] * pad_needed)
                norms_of_weight.extend([padding_norms] * pad_needed)

            # G == 0 (gram_level==0) naturally makes expected_total_gram 0
            # and norms_of_gram stays empty -- no separate "is gram active"
            # flag/branch needed anywhere below.
            G = len(self.gram_scalar_names)
            expected_total_gram = len(expert_params) * ep_per_rank * G
            pad_needed_gram = expected_total_gram - len(norms_of_gram)
            if pad_needed_gram > 0:
                norms_of_gram.extend([padding_norms] * pad_needed_gram)

            # Radial metrics are unconditional (R = len(RADIAL_METRIC_NAMES)
            # is a plain constant, never 0) -- but still need the same
            # padding as gram: a param with no update this step (u is None)
            # skips its whole per-expert loop above, contributing zero
            # entries, same reason gram/update/weight norms need padding.
            R = len(RADIAL_METRIC_NAMES)
            expected_total_radial = len(expert_params) * ep_per_rank * R
            pad_needed_radial = expected_total_radial - len(norms_of_radial)
            if pad_needed_radial > 0:
                norms_of_radial.extend([padding_norms] * pad_needed_radial)

            # Single flat per-rank buffer: [scalar update norms, scalar weight
            # norms, gram scalars, radial scalars, gram vectors, spectrum
            # update, spectrum weight] — one collective for everything in
            # this step, instead of a separate all_gather per block/kind
            # (all pieces are fully computed above with no ordering
            # dependency between them).
            local_parts = [
                torch.stack(norms_of_update).float().to(device),
                torch.stack(norms_of_weight).float().to(device),
            ]
            if norms_of_gram:
                local_parts.append(torch.stack(norms_of_gram).float().to(device))
            local_parts.append(torch.stack(norms_of_radial).float().to(device))
            if gram_vec_flat is not None:
                local_parts.append(gram_vec_flat)
            if update_spectrum_flat is not None:
                local_parts.append(update_spectrum_flat)
                if weight_spectrum_flat is not None:
                    local_parts.append(weight_spectrum_flat)

            local_buf = torch.cat(local_parts)
            gathered = funcol.all_gather_tensor(
                local_buf, gather_dim=0, group=fsdp_mesh
            )
            per_rank_total = local_buf.numel()

            if local_rank == 0:
                norm_names = list(self.norms_to_log)

                P = len(expert_params)  # parameters per rank
                E = ep_per_rank  # experts per rank
                K = kinds_of_norms  # norms per expert
                block = P * E * K  # == expected_total

                weight_scalar_offset = expected_total
                gram_scalar_offset = 2 * expected_total
                radial_scalar_offset = 2 * expected_total + expected_total_gram
                gram_vec_offset = radial_scalar_offset + expected_total_radial
                spectrum_offset = gram_vec_offset + (
                    self._expert_gram_vec_total_size if gram_vec_flat is not None else 0
                )

                for idx in range(world_size * block):
                    r, rem = divmod(idx, block)  # producing rank
                    p, rem2 = divmod(rem, E * K)  # parameter index
                    e, k = divmod(rem2, K)  # expert, norm indices

                    actual_ep_idx = e + r * E
                    if actual_ep_idx >= expert_params[0].shape[0]:
                        continue  # skip pure padding slots

                    cleaned_name = remove_orig_mod_and_weight_for_p_name(
                        expert_param_names[p]
                    )
                    norm_name = norm_names[k]
                    rank_base = r * per_rank_total

                    key_update = (
                        f"track_update_{norm_name}/ep_{actual_ep_idx}/{cleaned_name}"
                    )
                    final_norms[key_update] = gathered[rank_base + rem]

                    key_param = (
                        f"track_param_{norm_name}/ep_{actual_ep_idx}/{cleaned_name}"
                    )
                    final_norms[key_param] = gathered[
                        rank_base + weight_scalar_offset + rem
                    ]

                # block_g == 0 when G == 0, so this loop naturally no-ops --
                # no explicit "is gram active" guard needed.
                gram_names = list(self.gram_scalar_names)
                block_g = P * E * G
                for idx in range(world_size * block_g):
                    r, rem = divmod(idx, block_g)  # producing rank
                    p, rem2 = divmod(rem, E * G)  # parameter index
                    e, g = divmod(rem2, G)  # expert, gram-metric indices

                    actual_ep_idx = e + r * E
                    if actual_ep_idx >= expert_params[0].shape[0]:
                        continue  # skip pure padding slots

                    cleaned_name = remove_orig_mod_and_weight_for_p_name(
                        expert_param_names[p]
                    )
                    gram_param_name = _gram_log_param_name(
                        cleaned_name, tuple(expert_params[p].shape)
                    )
                    gram_name = gram_names[g]
                    rank_base = r * per_rank_total

                    key_gram = (
                        f"track_gram_{gram_name}/ep_{actual_ep_idx}/{gram_param_name}"
                    )
                    final_norms[key_gram] = gathered[
                        rank_base + gram_scalar_offset + rem
                    ]

                # Same layout as the gram-scalar loop above, but unconditional
                # (block_radial is never 0, no gram_level gate).
                radial_names = RADIAL_METRIC_NAMES
                block_radial = P * E * R
                for idx in range(world_size * block_radial):
                    r, rem = divmod(idx, block_radial)  # producing rank
                    p, rem2 = divmod(rem, E * R)  # parameter index
                    e, rk = divmod(rem2, R)  # expert, radial-metric indices

                    actual_ep_idx = e + r * E
                    if actual_ep_idx >= expert_params[0].shape[0]:
                        continue  # skip pure padding slots

                    cleaned_name = remove_orig_mod_and_weight_for_p_name(
                        expert_param_names[p]
                    )
                    radial_name = radial_names[rk]
                    rank_base = r * per_rank_total

                    key_radial = (
                        f"track_radial_{radial_name}/ep_{actual_ep_idx}/{cleaned_name}"
                    )
                    final_norms[key_radial] = gathered[
                        rank_base + radial_scalar_offset + rem
                    ]

                if gram_vec_flat is not None:
                    for block_idx, (start, end) in enumerate(blocks):
                        K_block_vec = self._expert_gram_vec_len_per_block[block_idx]
                        if K_block_vec == 0:
                            continue
                        block_base_vec = self._expert_gram_vec_block_offset[block_idx]
                        block_names = expert_param_names[start:end]
                        P_block = end - start
                        expected_block = P_block * ep_per_rank

                        for bidx in range(world_size * expected_block):
                            r, rem = divmod(bidx, expected_block)
                            p_idx, e = divmod(rem, ep_per_rank)
                            actual_ep_idx = e + r * ep_per_rank
                            if actual_ep_idx >= expert_params[0].shape[0]:
                                continue  # skip pure padding slots

                            cleaned_name = remove_orig_mod_and_weight_for_p_name(
                                block_names[p_idx]
                            )
                            gram_param_name = _gram_log_param_name(
                                cleaned_name,
                                tuple(expert_params[start + p_idx].shape),
                            )
                            rank_base = r * per_rank_total
                            local_off = block_base_vec + rem * K_block_vec * n_vec
                            for vi, vname in enumerate(self.gram_vector_names):
                                v_start = (
                                    rank_base
                                    + gram_vec_offset
                                    + local_off
                                    + vi * K_block_vec
                                )
                                gram_key = (
                                    f"track_gram_{vname}/ep_{actual_ep_idx}/"
                                    f"{gram_param_name}"
                                )
                                final_norms[gram_key] = gathered[
                                    v_start : v_start + K_block_vec
                                ]

                if update_spectrum_flat is not None:
                    for block_idx, (start, end) in enumerate(blocks):
                        K_block = self._expert_spectrum_len_per_block[block_idx]
                        if K_block == 0:
                            continue
                        block_base = self._expert_spectrum_block_offset[block_idx]
                        block_names = expert_param_names[start:end]
                        P_block = end - start
                        expected_block = P_block * ep_per_rank

                        for bidx in range(world_size * expected_block):
                            r, rem = divmod(bidx, expected_block)
                            p_idx, e = divmod(rem, ep_per_rank)
                            actual_ep_idx = e + r * ep_per_rank
                            if actual_ep_idx >= expert_params[0].shape[0]:
                                continue  # skip pure padding slots

                            cleaned_name = remove_orig_mod_and_weight_for_p_name(
                                block_names[p_idx]
                            )
                            rank_base = r * per_rank_total
                            local_off = block_base + rem * K_block
                            spec_start = rank_base + spectrum_offset + local_off
                            final_norms[
                                f"track_spectrum_update/ep_{actual_ep_idx}/{cleaned_name}"
                            ] = gathered[spec_start : spec_start + K_block]
                            if weight_spectrum_flat is not None:
                                w_start = spec_start + self._expert_spectrum_total_size
                                final_norms[
                                    f"track_spectrum_param/ep_{actual_ep_idx}/{cleaned_name}"
                                ] = gathered[w_start : w_start + K_block]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_ddp")
    def step_ddp(
        self,
        ddp_params,
        ddp_param_names,
        workspace,
        skip_update: bool = False,
    ):

        need_to_calculate_norm = self.need_to_calculate_norm
        num_gram_types = len(self.gram_scalar_names)
        # Radial-dynamics metrics (radial_helper.py) are always computed
        # whenever any per-param logging fires, independent of gram_level
        # -- a fixed-size list, same for every param regardless of shape,
        # so (unlike gram's vectors) no per-param length table is needed.
        num_radial_types = len(RADIAL_METRIC_NAMES)

        # --- distributed groups ---
        dp_replicate_mesh = (
            self.parallel_dims.get_optional_mesh("dp_replicate")
            if self.dp_replicate_enabled
            else None
        )

        world_size, rank = (
            (dp_replicate_mesh.size(), dp_replicate_mesh.get_local_rank())
            if dp_replicate_mesh
            else (1, 0)
        )
        if world_size != self._ddp_world_size or rank != self._ddp_rank:
            self._precompute_ddp_metadata()

        tp_mesh = (
            self.parallel_dims.get_optional_mesh("tp") if self.tp_enabled else None
        )
        tp_world_size = tp_mesh.size() if tp_mesh is not None else 1

        device = ddp_params[0].device
        cast_dtype = self.communication_dtype  # comm/exchange dtype
        total_buckets = self._ddp_total_buckets
        num_norm_types = len(self.norms_to_log)
        final_norms = {}

        # -------- Phase A: precompute local LMO updates (no comm) --------
        local_updates: list[torch.Tensor | None] = [None] * len(ddp_params)
        local_displacements: list[torch.Tensor | None] = [None] * len(ddp_params)
        local_aus_corrections: list[torch.Tensor | None] = [None] * len(ddp_params)
        local_aus_etas: list[torch.Tensor | None] = [None] * len(ddp_params)
        local_aus_validity: list[torch.Tensor | None] = [None] * len(ddp_params)
        local_aus_finiteness: list[torch.Tensor | None] = [None] * len(ddp_params)
        lmo_inputs = self._prepare_ddp_lmo(
            ddp_params,
            workspace,
            tp_mesh=tp_mesh,
            tp_world_size=tp_world_size,
        )
        for i in self._ddp_owned_indices:
            group_idx = self._ddp_param_group_idx[i]
            g = lmo_inputs[i]
            if g is None:
                local_updates[i] = None
                if self.param_groups[group_idx]["aus_enabled"]:
                    (
                        local_displacements[i],
                        local_aus_corrections[i],
                        local_aus_etas[i],
                        local_aus_validity[i],
                        local_aus_finiteness[i],
                    ) = self._build_aus_displacement(ddp_params[i], None, group_idx)
                continue
            param_kwargs = self.groups_info[group_idx][-1]
            u = self.lmo(g, **param_kwargs)
            local_updates[i] = u
            if self.param_groups[group_idx]["aus_enabled"]:
                (
                    local_displacements[i],
                    local_aus_corrections[i],
                    local_aus_etas[i],
                    local_aus_validity[i],
                    local_aus_finiteness[i],
                ) = self._build_aus_displacement(ddp_params[i], u, group_idx)
            else:
                local_displacements[i] = u
        if self.aus_enabled:
            self._raise_if_any_aus_nonfinite(
                local_aus_finiteness, device=device
            )

        # -------- Phase B: DDP communication (one-shot flat all_gather) and global updates --------
        (global_updates, global_update_bufs, zero_by_shape, norm_scratch) = (
            workspace["global_updates"],
            workspace["global_update_bufs"],
            workspace["zero_by_shape"],
            workspace["norm_scratch"],
        )

        if skip_update:
            global_updates[:] = [None] * len(global_updates)
        if not skip_update:
            if dp_replicate_mesh is not None and world_size > 1:
                (
                    ddp_flat_send,
                    ddp_flat_recv_chunks,
                    ddp_pack_dst_views,
                    ddp_unpack_src_views,
                    ddp_unpack_dst_views,
                ) = (
                    workspace["ddp_flat_send"],
                    workspace["ddp_flat_recv_chunks"],
                    workspace["ddp_pack_dst_views"],
                    workspace["ddp_unpack_src_views"],
                    workspace["ddp_unpack_dst_views"],
                )

                pack_src_views: list[torch.Tensor] = []
                for param_idx in self._ddp_owned_indices:
                    u = local_displacements[param_idx]
                    if u is None:
                        ref = ddp_params[param_idx]
                        u = zero_by_shape[(tuple(ref.shape), cast_dtype)]
                    pack_src_views.append(u)

                if pack_src_views:
                    torch._foreach_copy_(ddp_pack_dst_views, pack_src_views)

                owned_total_numel = self._ddp_rank_owned_numels[rank]
                max_owned_numel = self._ddp_max_rank_owned_numel
                if owned_total_numel < max_owned_numel:
                    ddp_flat_send.narrow(
                        0, owned_total_numel, max_owned_numel - owned_total_numel
                    ).zero_()

                dist.all_gather(
                    ddp_flat_recv_chunks,
                    ddp_flat_send,
                    group=dp_replicate_mesh.get_group(),
                )
                if ddp_unpack_src_views:
                    torch._foreach_copy_(ddp_unpack_dst_views, ddp_unpack_src_views)

                for param_idx, buf in enumerate(global_update_bufs):
                    global_updates[param_idx] = buf
            else:
                for param_idx, p in enumerate(ddp_params):
                    u = local_displacements[param_idx]
                    if u is None:
                        global_updates[param_idx] = zero_by_shape[
                            (tuple(p.shape), cast_dtype)
                        ]
                    else:
                        global_updates[param_idx] = u

        upd_norm_local_flat = None
        w_norm_local_flat = None
        if need_to_calculate_norm:
            required_norm_elems = total_buckets * num_norm_types
            upd_norm_local_flat = workspace.get("upd_norm_local_flat")
            if (
                upd_norm_local_flat is None
                or upd_norm_local_flat.numel() != required_norm_elems
            ):
                upd_norm_local_flat = torch.empty(
                    required_norm_elems, dtype=torch.float32, device=device
                )
                workspace["upd_norm_local_flat"] = upd_norm_local_flat
            upd_norm_local_flat.zero_()

            w_norm_local_flat = workspace.get("w_norm_local_flat")
            if (
                w_norm_local_flat is None
                or w_norm_local_flat.numel() != required_norm_elems
            ):
                w_norm_local_flat = torch.empty(
                    required_norm_elems, dtype=torch.float32, device=device
                )
                workspace["w_norm_local_flat"] = w_norm_local_flat
            w_norm_local_flat.zero_()

        gram_norm_local_flat = None
        if need_to_calculate_norm and num_gram_types > 0:
            required_gram_elems = total_buckets * num_gram_types
            gram_norm_local_flat = workspace.get("gram_norm_local_flat")
            if (
                gram_norm_local_flat is None
                or gram_norm_local_flat.numel() != required_gram_elems
            ):
                gram_norm_local_flat = torch.empty(
                    required_gram_elems, dtype=torch.float32, device=device
                )
                workspace["gram_norm_local_flat"] = gram_norm_local_flat
            gram_norm_local_flat.zero_()

        # Radial-dynamics metrics: same fixed-stride pattern as
        # gram_norm_local_flat above, but unconditional (no gram_level
        # gate -- num_radial_types is a plain constant, never 0).
        radial_local_flat = None
        if need_to_calculate_norm:
            required_radial_elems = total_buckets * num_radial_types
            radial_local_flat = workspace.get("radial_local_flat")
            if (
                radial_local_flat is None
                or radial_local_flat.numel() != required_radial_elems
            ):
                radial_local_flat = torch.empty(
                    required_radial_elems, dtype=torch.float32, device=device
                )
                workspace["radial_local_flat"] = radial_local_flat
            radial_local_flat.zero_()

        # AUS correction, effective eta, and fallback validity are tiny scalar
        # diagnostics. They share the owner-bucket metric gather on logging steps.
        aus_local_flat = None
        if need_to_calculate_norm and self.aus_enabled:
            required_aus_elems = total_buckets * 3
            aus_local_flat = workspace.get("aus_local_flat")
            if (
                aus_local_flat is None
                or aus_local_flat.numel() != required_aus_elems
            ):
                aus_local_flat = torch.zeros(
                    required_aus_elems, dtype=torch.float32, device=device
                )
                workspace["aus_local_flat"] = aus_local_flat
            aus_local_flat.zero_()
            for my_idx in self._ddp_owned_indices:
                group_idx = self._ddp_param_group_idx[my_idx]
                if not self.param_groups[group_idx]["aus_enabled"]:
                    continue
                correction = local_aus_corrections[my_idx]
                eta = local_aus_etas[my_idx]
                valid = local_aus_validity[my_idx]
                if correction is None or eta is None or valid is None:
                    continue
                owner_bucket = self._ddp_owner_bucket_by_param[my_idx]
                aus_base = owner_bucket * 3
                aus_local_flat[aus_base : aus_base + 3].copy_(
                    torch.stack((correction, eta, valid.to(correction.dtype)))
                )

        # Singular-value spectrum: a separate flat buffer padded to the largest
        # per-rank total across all ranks (`_ddp_spectrum_max_total`), since
        # different ranks own params of different shapes and therefore different
        # total spectrum lengths — unlike the fixed-stride scalar norm buffers.
        upd_spectrum_local_flat = None
        w_spectrum_local_flat = None
        rank_spectrum_offsets = (
            self._ddp_spectrum_offsets_by_rank[rank]
            if rank < len(self._ddp_spectrum_offsets_by_rank)
            else []
        )
        if need_to_calculate_norm and self._ddp_spectrum_max_total > 0:
            required_spectrum_elems = self._ddp_spectrum_max_total
            upd_spectrum_local_flat = workspace.get("upd_spectrum_local_flat")
            if (
                upd_spectrum_local_flat is None
                or upd_spectrum_local_flat.numel() != required_spectrum_elems
            ):
                upd_spectrum_local_flat = torch.zeros(
                    required_spectrum_elems, dtype=torch.float32, device=device
                )
                workspace["upd_spectrum_local_flat"] = upd_spectrum_local_flat
            upd_spectrum_local_flat.zero_()

            w_spectrum_local_flat = workspace.get("w_spectrum_local_flat")
            if (
                w_spectrum_local_flat is None
                or w_spectrum_local_flat.numel() != required_spectrum_elems
            ):
                w_spectrum_local_flat = torch.zeros(
                    required_spectrum_elems, dtype=torch.float32, device=device
                )
                workspace["w_spectrum_local_flat"] = w_spectrum_local_flat
            w_spectrum_local_flat.zero_()

        # Gram vectors: same pad-to-rank-max reasoning as the spectrum
        # buffers above (different ranks own different-shaped params, hence
        # different total gram-vector lengths) -- static offsets from
        # _precompute_ddp_gram_vector_metadata.
        gram_vec_local_flat = None
        rank_gram_vec_offsets = (
            self._ddp_gram_vec_offsets_by_rank[rank]
            if rank < len(self._ddp_gram_vec_offsets_by_rank)
            else []
        )
        if need_to_calculate_norm and self._ddp_gram_vec_max_total > 0:
            required_gram_vec_elems = self._ddp_gram_vec_max_total
            gram_vec_local_flat = workspace.get("gram_vec_local_flat")
            if (
                gram_vec_local_flat is None
                or gram_vec_local_flat.numel() != required_gram_vec_elems
            ):
                gram_vec_local_flat = torch.zeros(
                    required_gram_vec_elems, dtype=torch.float32, device=device
                )
                workspace["gram_vec_local_flat"] = gram_vec_local_flat
            gram_vec_local_flat.zero_()

        # ---- local update norms (owner slots only) ----
        if need_to_calculate_norm and upd_norm_local_flat is not None:
            for my_idx in self._ddp_owned_indices:
                u = local_updates[my_idx]
                if u is None:
                    continue
                group_idx = self._ddp_param_group_idx[my_idx]
                lr, *_ = self.groups_info[group_idx]
                scratch = norm_scratch[my_idx]
                if scratch is None:
                    raise ValueError("Missing DDP norm scratch buffer for owned index.")
                if self.param_groups[group_idx]["aus_enabled"]:
                    eta = local_aus_etas[my_idx]
                    if eta is None:
                        continue
                    torch.mul(u, -eta, out=scratch)
                else:
                    torch.mul(u, -lr, out=scratch)
                upd_norms = calculate_norm(scratch, self.norms_to_log)
                upd_spectrum = upd_norms.pop("spectrum")
                owner_bucket = self._ddp_owner_bucket_by_param[my_idx]
                base = owner_bucket * num_norm_types
                upd_norm_local_flat[base : base + num_norm_types].copy_(
                    torch.stack(list(upd_norms.values()))
                )
                if upd_spectrum_local_flat is not None and owner_bucket < len(
                    rank_spectrum_offsets
                ):
                    off = rank_spectrum_offsets[owner_bucket]
                    upd_spectrum_local_flat[off : off + upd_spectrum.numel()].copy_(
                        upd_spectrum
                    )

        # -------- Weight norms + gram metrics (PRE-UPDATE) --------
        # Moved to run BEFORE Phase C's apply below (was "Phase C.5 POST-
        # UPDATE"), so `w` here is genuinely pre-update -- needed so
        # calculate_gram_metrics gets W and U simultaneously (see
        # gram_helper.py). `U` is the RAW moment (`lmo_inputs[my_idx]`, from
        # Phase A, before `self.lmo`), not the LMO-processed `u` -- see
        # readme.md. Same TP-gather/`.to_local()` as before, just earlier --
        # no new communication. track_param_* keeps its historical
        # post-update meaning via a cheap local pseudo-weight
        # (_pseudo_post_update_weight) instead of re-reading the parameter a
        # second time after the real apply.
        if w_norm_local_flat is not None:
            for my_idx in self._ddp_owned_indices:
                w = ddp_params[my_idx]
                group_idx = self._ddp_param_group_idx[my_idx]
                lr, _, _, wd, _ = self.groups_info[group_idx]
                tp_gather_info = (
                    self._ddp_tp_gather_info_by_idx[my_idx]
                    if tp_mesh is not None
                    else None
                )
                if tp_gather_info is not None:
                    w_local = w.to_local() if isinstance(w, DTensor) else w
                    w = gather_tp_shard(w_local, tp_mesh, tp_world_size, tp_gather_info)
                elif isinstance(w, DTensor):
                    w = w.to_local()

                # No gradient for this param this step (local_updates[my_idx]
                # is None) -- the real apply still runs with a zero update
                # (see Phase B's zero_by_shape substitution above, so only
                # weight decay -- if any -- applies), rather than skipping
                # the param outright. Weight-norm/gram must match that: still
                # compute them (unlike the update-norm loop above, which
                # correctly leaves its flat-buffer slot at zero for this
                # param, since the update itself really is zero).
                u = local_updates[my_idx]
                if u is None:
                    u = torch.zeros_like(w)

                if self.param_groups[group_idx]["aus_enabled"]:
                    displacement = local_displacements[my_idx]
                    if displacement is None:
                        displacement = torch.zeros_like(w)
                    pseudo_w = w - displacement
                else:
                    pseudo_w = _pseudo_post_update_weight(w, u, lr, wd)
                w_norms = calculate_norm(pseudo_w, self.norms_to_log)
                w_spectrum = w_norms.pop("spectrum")
                owner_bucket = self._ddp_owner_bucket_by_param[my_idx]
                base = owner_bucket * num_norm_types
                w_norm_local_flat[base : base + num_norm_types].copy_(
                    torch.stack(list(w_norms.values()))
                )
                if w_spectrum_local_flat is not None and owner_bucket < len(
                    rank_spectrum_offsets
                ):
                    off = rank_spectrum_offsets[owner_bucket]
                    w_spectrum_local_flat[off : off + w_spectrum.numel()].copy_(
                        w_spectrum
                    )

                if gram_norm_local_flat is not None or gram_vec_local_flat is not None:
                    g_raw = lmo_inputs[my_idx]
                    if g_raw is None:
                        g_raw = torch.zeros_like(w)
                    gram_metrics = calculate_gram_metrics(
                        w, g_raw, pseudo_w, level=self.gram_level
                    )
                    if gram_norm_local_flat is not None:
                        gram_base = owner_bucket * num_gram_types
                        gram_norm_local_flat[
                            gram_base : gram_base + num_gram_types
                        ].copy_(
                            torch.stack(
                                [gram_metrics[name] for name in self.gram_scalar_names]
                            )
                        )
                    if gram_vec_local_flat is not None and owner_bucket < len(
                        rank_gram_vec_offsets
                    ):
                        off = rank_gram_vec_offsets[owner_bucket]
                        for vname in self.gram_vector_names:
                            vec = gram_metrics[vname]
                            gram_vec_local_flat[off : off + vec.numel()].copy_(vec)
                            off += vec.numel()

                if radial_local_flat is not None:
                    radial_metrics = calculate_radial_metrics(
                        w,
                        pseudo_w,
                        self._radial_state_by_param_id[id(ddp_params[my_idx])],
                    )
                    radial_base = owner_bucket * num_radial_types
                    radial_local_flat[
                        radial_base : radial_base + num_radial_types
                    ].copy_(
                        torch.stack(
                            [radial_metrics[name] for name in RADIAL_METRIC_NAMES]
                        )
                    )

        # -------- Phase C: apply once (pre-cast + grouped foreach apply) --------
        if not skip_update:
            apply_updates = self._prepare_ddp_apply_updates(
                global_updates,
                workspace,
                tp_mesh=tp_mesh,
            )
            self._update_ddp_params_fast(apply_updates)

        # -------- Phase D: final norm gather/log --------
        if not need_to_calculate_norm:
            return

        if upd_norm_local_flat is None:
            return

        # One collective for everything this step (scalar update/weight/gram
        # norms + both spectrum halves), instead of up to 5 separate
        # all_gather_tensor calls -- same "single flat buffer" pattern
        # step_experts already uses. `offsets` is derived from what actually
        # got packed, so it can't drift out of sync with pack order.
        local_buf, offsets = _pack_segments(
            [
                ("upd", upd_norm_local_flat),
                ("w", w_norm_local_flat),
                ("gram", gram_norm_local_flat),
                ("gram_vec", gram_vec_local_flat),
                ("radial", radial_local_flat),
                ("aus", aus_local_flat),
                ("upd_spec", upd_spectrum_local_flat),
                ("w_spec", w_spectrum_local_flat),
            ]
        )
        if dp_replicate_mesh is not None and world_size > 1:
            gathered = funcol.all_gather_tensor(
                local_buf, gather_dim=0, group=dp_replicate_mesh
            )
        else:
            gathered = local_buf
        per_rank_total = local_buf.numel()

        if self.is_dp_rank_0:
            cleaned_names = (
                self._ddp_clean_param_names
                if len(self._ddp_clean_param_names) == len(ddp_param_names)
                else [
                    remove_orig_mod_and_weight_for_p_name(p_name)
                    for p_name in ddp_param_names
                ]
            )
            owner_ranks = (
                self._ddp_owner_rank_by_param
                if len(self._ddp_owner_rank_by_param) == len(ddp_param_names)
                else [i % world_size for i in range(len(ddp_param_names))]
            )
            owner_buckets = (
                self._ddp_owner_bucket_by_param
                if len(self._ddp_owner_bucket_by_param) == len(ddp_param_names)
                else [i // world_size for i in range(len(ddp_param_names))]
            )
            for param_idx, cleaned in enumerate(cleaned_names):
                owner_rank = owner_ranks[param_idx]
                owner_bucket = owner_buckets[param_idx]
                rank_base = owner_rank * per_rank_total
                base = offsets["upd"] + owner_bucket * num_norm_types
                w_base = offsets["w"] + owner_bucket * num_norm_types
                for k, norm_name in enumerate(self.norms_to_log):
                    final_norms[f"track_update_{norm_name}/{cleaned}"] = gathered[
                        rank_base + base + k
                    ]
                    final_norms[f"track_param_{norm_name}/{cleaned}"] = gathered[
                        rank_base + w_base + k
                    ]
                if "gram" in offsets:
                    gram_param_name = _gram_log_param_name(
                        cleaned, tuple(ddp_params[param_idx].shape)
                    )
                    gram_base = offsets["gram"] + owner_bucket * num_gram_types
                    for gk, gram_name in enumerate(self.gram_scalar_names):
                        final_norms[
                            f"track_gram_{gram_name}/{gram_param_name}"
                        ] = gathered[rank_base + gram_base + gk]
                if "radial" in offsets:
                    radial_base = offsets["radial"] + owner_bucket * num_radial_types
                    for rk, radial_name in enumerate(RADIAL_METRIC_NAMES):
                        final_norms[f"track_radial_{radial_name}/{cleaned}"] = gathered[
                            rank_base + radial_base + rk
                        ]
                if (
                    "aus" in offsets
                    and self.param_groups[
                        self._ddp_param_group_idx[param_idx]
                    ]["aus_enabled"]
                ):
                    aus_base = offsets["aus"] + owner_bucket * 3
                    final_norms[f"track_aus_correction/{cleaned}"] = gathered[
                        rank_base + aus_base
                    ]
                    final_norms[f"track_aus_eta/{cleaned}"] = gathered[
                        rank_base + aus_base + 1
                    ]
                    final_norms[f"track_aus_valid/{cleaned}"] = gathered[
                        rank_base + aus_base + 2
                    ]

            if "gram_vec" in offsets:
                gram_vec_lens = (
                    self._ddp_gram_vec_len_by_param
                    if len(self._ddp_gram_vec_len_by_param) == len(ddp_param_names)
                    else None
                )
                gram_vec_offsets_by_rank = self._ddp_gram_vec_offsets_by_rank
                if gram_vec_lens is not None:
                    for param_idx, cleaned in enumerate(cleaned_names):
                        owner_rank = owner_ranks[param_idx]
                        owner_bucket = owner_buckets[param_idx]
                        if owner_rank >= len(gram_vec_offsets_by_rank):
                            continue
                        rank_offsets = gram_vec_offsets_by_rank[owner_rank]
                        if owner_bucket >= len(rank_offsets):
                            continue
                        length = gram_vec_lens[param_idx]
                        rank_base = owner_rank * per_rank_total
                        v_start = (
                            rank_base + offsets["gram_vec"] + rank_offsets[owner_bucket]
                        )
                        gram_param_name = _gram_log_param_name(
                            cleaned, tuple(ddp_params[param_idx].shape)
                        )
                        for vname in self.gram_vector_names:
                            final_norms[
                                f"track_gram_{vname}/{gram_param_name}"
                            ] = gathered[v_start : v_start + length]
                            v_start += length

            if "upd_spec" in offsets:
                spectrum_lens = (
                    self._ddp_spectrum_len_by_param
                    if len(self._ddp_spectrum_len_by_param) == len(ddp_param_names)
                    else None
                )
                spectrum_offsets_by_rank = self._ddp_spectrum_offsets_by_rank
                if spectrum_lens is not None:
                    for param_idx, cleaned in enumerate(cleaned_names):
                        owner_rank = owner_ranks[param_idx]
                        owner_bucket = owner_buckets[param_idx]
                        if owner_rank >= len(spectrum_offsets_by_rank):
                            continue
                        rank_offsets = spectrum_offsets_by_rank[owner_rank]
                        if owner_bucket >= len(rank_offsets):
                            continue
                        length = spectrum_lens[param_idx]
                        rank_base = owner_rank * per_rank_total
                        upd_spec_start = (
                            rank_base + offsets["upd_spec"] + rank_offsets[owner_bucket]
                        )
                        final_norms[f"track_spectrum_update/{cleaned}"] = gathered[
                            upd_spec_start : upd_spec_start + length
                        ]
                        if "w_spec" in offsets:
                            w_spec_start = (
                                rank_base
                                + offsets["w_spec"]
                                + rank_offsets[owner_bucket]
                            )
                            final_norms[f"track_spectrum_param/{cleaned}"] = gathered[
                                w_spec_start : w_spec_start + length
                            ]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    def _gather_and_log_fsdp(
        self,
        norms_of_update,
        norms_of_weight,
        upd_spectrum_local_flat,
        w_spectrum_local_flat,
        fsdp_mesh,
        device,
        fsdp_param_names,
        world_size,
        total_buckets,
        norms_of_gram=None,
        gram_vec_local_flat=None,
        norms_of_radial=None,
    ):
        """
        Gathers FSDP norm/gram/radial/spectrum tensors from all ranks and
        logs them on rank 0. One collective for everything this step (scalar
        update/weight/gram/radial norms, fixed stride `total_buckets *
        num_types`, plus both spectrum halves and gram vectors, variable
        per-param length padded per-rank to `_fsdp_spectrum_max_total`/
        `_fsdp_gram_vec_max_total` -- see `_precompute_fsdp_metadata`/
        `_precompute_fsdp_gram_vector_metadata`), instead of a separate
        all_gather per segment -- same "single flat buffer" pattern
        `step_experts`/`step_ddp`'s Phase D use.
        """
        upd = torch.stack(norms_of_update).float().to(device)
        w = torch.stack(norms_of_weight).float().to(device) if norms_of_weight else None
        # norms_of_gram stays empty whenever gram_level==0 (see
        # gram_helper.py) -- no separate "is gram active" flag needed here.
        gram = torch.stack(norms_of_gram).float().to(device) if norms_of_gram else None
        # norms_of_radial is never empty (radial metrics are unconditional,
        # unlike gram) -- same fixed-stride list-of-padding-entries pattern.
        radial = (
            torch.stack(norms_of_radial).float().to(device) if norms_of_radial else None
        )

        local_buf, offsets = _pack_segments(
            [
                ("upd", upd),
                ("w", w),
                ("gram", gram),
                ("gram_vec", gram_vec_local_flat),
                ("radial", radial),
                ("upd_spec", upd_spectrum_local_flat),
                ("w_spec", w_spectrum_local_flat),
            ]
        )
        gathered = funcol.all_gather_tensor(local_buf, gather_dim=0, group=fsdp_mesh)
        per_rank_total = local_buf.numel()

        final_norms = {}
        if self.is_dp_rank_0:
            num_norm_types = len(self.norms_to_log)
            num_gram_types = len(self.gram_scalar_names)
            num_radial_types = len(RADIAL_METRIC_NAMES)
            cleaned_names = [
                remove_orig_mod_and_weight_for_p_name(pn) for pn in fsdp_param_names
            ]

            for param_idx, cleaned_p_name in enumerate(cleaned_names):
                owner_rank = param_idx % world_size
                bucket_idx_on_owner = param_idx // world_size
                rank_base = owner_rank * per_rank_total
                base = offsets["upd"] + bucket_idx_on_owner * num_norm_types

                for norm_idx, norm_name in enumerate(self.norms_to_log):
                    final_norms[
                        f"track_update_{norm_name}/{cleaned_p_name}"
                    ] = gathered[rank_base + base + norm_idx]
                    if "w" in offsets:
                        w_base = offsets["w"] + bucket_idx_on_owner * num_norm_types
                        final_norms[
                            f"track_param_{norm_name}/{cleaned_p_name}"
                        ] = gathered[rank_base + w_base + norm_idx]

                if "gram" in offsets:
                    gram_param_name = _gram_log_param_name(
                        cleaned_p_name, tuple(self.fsdp_params[param_idx].shape)
                    )
                    gram_base = offsets["gram"] + bucket_idx_on_owner * num_gram_types
                    for gram_idx, gram_name in enumerate(self.gram_scalar_names):
                        final_norms[
                            f"track_gram_{gram_name}/{gram_param_name}"
                        ] = gathered[rank_base + gram_base + gram_idx]
                if "radial" in offsets:
                    radial_base = (
                        offsets["radial"] + bucket_idx_on_owner * num_radial_types
                    )
                    for radial_idx, radial_name in enumerate(RADIAL_METRIC_NAMES):
                        final_norms[
                            f"track_radial_{radial_name}/{cleaned_p_name}"
                        ] = gathered[rank_base + radial_base + radial_idx]

            if "gram_vec" in offsets:
                for param_idx, cleaned_p_name in enumerate(cleaned_names):
                    owner_rank = param_idx % world_size
                    owner_bucket = param_idx // world_size
                    rank_offsets = self._fsdp_gram_vec_offsets_by_rank[owner_rank]
                    if owner_bucket >= len(rank_offsets):
                        continue
                    length = self._fsdp_gram_vec_len_by_param[param_idx]
                    rank_base = owner_rank * per_rank_total
                    v_start = (
                        rank_base + offsets["gram_vec"] + rank_offsets[owner_bucket]
                    )
                    gram_param_name = _gram_log_param_name(
                        cleaned_p_name, tuple(self.fsdp_params[param_idx].shape)
                    )
                    for vname in self.gram_vector_names:
                        final_norms[f"track_gram_{vname}/{gram_param_name}"] = gathered[
                            v_start : v_start + length
                        ]
                        v_start += length

            if "upd_spec" in offsets:
                for param_idx, cleaned_p_name in enumerate(cleaned_names):
                    owner_rank = param_idx % world_size
                    owner_bucket = param_idx // world_size
                    rank_offsets = self._fsdp_spectrum_offsets_by_rank[owner_rank]
                    if owner_bucket >= len(rank_offsets):
                        continue
                    length = self._fsdp_spectrum_len_by_param[param_idx]
                    rank_base = owner_rank * per_rank_total
                    upd_spec_start = (
                        rank_base + offsets["upd_spec"] + rank_offsets[owner_bucket]
                    )
                    final_norms[f"track_spectrum_update/{cleaned_p_name}"] = gathered[
                        upd_spec_start : upd_spec_start + length
                    ]
                    if "w_spec" in offsets:
                        w_spec_start = (
                            rank_base + offsets["w_spec"] + rank_offsets[owner_bucket]
                        )
                        final_norms[
                            f"track_spectrum_param/{cleaned_p_name}"
                        ] = gathered[w_spec_start : w_spec_start + length]

        if self.is_dp_rank_0:
            self.norms_at_current_step.update(final_norms)

    @record_function("disco.step_fsdp")
    def step_fsdp(
        self,
        fsdp_params,
        fsdp_param_names,
        workspace,
        skip_update=False,
    ):

        need_to_calculate_norm = self.need_to_calculate_norm

        fsdp_mesh = self.parallel_dims.get_optional_mesh("fsdp")
        world_size, rank = fsdp_mesh.size(), fsdp_mesh.get_local_rank()
        device = fsdp_params[0].device
        cast_dtype = self.communication_dtype

        tp_mesh = (
            self.parallel_dims.get_optional_mesh("tp") if self.tp_enabled else None
        )

        dp_replicate_mesh = (
            self.parallel_dims.get_optional_mesh("dp_replicate")
            if self.dp_replicate_enabled
            else None
        )
        fsdp_group = fsdp_mesh.get_group()

        global_updates = [None] * len(fsdp_params)
        norms_of_update, norms_of_weight, norms_of_gram = [], [], []
        norms_of_radial = []
        padding_norms = self._get_cached_padding_norms(device)
        # Empty whenever gram_level==0 (see gram_helper.py) -- no separate
        # "is gram active" flag needed, callers just check truthiness of
        # gram_padding/norms_of_gram.
        gram_padding = {
            name: self._get_cached_zero_scalar(device)
            for name in self.gram_scalar_names
        }
        # Radial-dynamics metrics (radial_helper.py) -- always computed,
        # independent of gram_level, same list-of-padding-entries pattern
        # as gram_padding above (non-owned buckets get zero placeholders
        # so norms_of_radial stays the same length on every rank).
        radial_padding = {
            name: self._get_cached_zero_scalar(device) for name in RADIAL_METRIC_NAMES
        }

        # Use pre-computed total_buckets from init.
        total_buckets = self._fsdp_total_buckets

        # Per-bucket LMO update, kept alive across both a2a modes so the
        # (now pre-update) weight-norm/gram block below -- moved to run
        # BEFORE the real apply -- can pair each bucket's full weight with
        # its update. A "bit more temporary memory" (per-bucket references
        # held a little longer), no new communication.
        u_keepalive: list[torch.Tensor | None] = [None] * total_buckets
        # Raw pre-LMO moment per bucket, kept alive the same way -- gram's
        # `U` argument must be the raw effective grad/momentum, not the
        # LMO-processed update (see readme.md).
        g_keepalive: list[torch.Tensor | None] = [None] * total_buckets

        # Singular-value spectrum: a separate flat buffer padded to the largest
        # per-rank total across all ranks (`_fsdp_spectrum_max_total`), since
        # different ranks own params of different shapes and therefore different
        # total spectrum lengths — unlike the fixed-stride scalar norm buffers.
        upd_spectrum_local_flat = None
        w_spectrum_local_flat = None
        if need_to_calculate_norm and self._fsdp_spectrum_max_total > 0:
            upd_spectrum_local_flat = torch.zeros(
                self._fsdp_spectrum_max_total, dtype=torch.float32, device=device
            )
            w_spectrum_local_flat = torch.zeros(
                self._fsdp_spectrum_max_total, dtype=torch.float32, device=device
            )

        # Gram vectors: same pad-to-rank-max reasoning as the spectrum
        # buffers above -- static offsets from
        # _precompute_fsdp_gram_vector_metadata. Pre-zeroed, written by
        # absolute offset only for owned buckets below -- unlike
        # norms_of_gram (a plain list needing explicit padding entries),
        # non-owned slots simply stay zero, same as the spectrum buffers.
        gram_vec_local_flat = None
        if need_to_calculate_norm and self._fsdp_gram_vec_max_total > 0:
            gram_vec_local_flat = torch.zeros(
                self._fsdp_gram_vec_max_total, dtype=torch.float32, device=device
            )

        use_global_fast_path = self.fsdp_a2a_mode == "once"
        bucket_workspace = None

        dist.barrier(fsdp_group)

        if use_global_fast_path:
            # Use persistent workspace cache; _workspace override for external callers.
            global_grad_send = workspace["global_grad_send"]
            global_grad_recv = workspace["global_grad_recv"]
            full_g_bufs = workspace["full_g_bufs"]
            global_upd_send = None if skip_update else workspace["global_upd_send"]
            global_upd_recv = None if skip_update else workspace["global_upd_recv"]
            global_upd_recv_cast = (
                None if skip_update else workspace["global_upd_recv_cast"]
            )
            if not skip_update and (global_upd_send is None or global_upd_recv is None):
                raise RuntimeError(
                    "Missing reverse A2A buffers for FSDP once-mode update."
                )

            self._prepare_fsdp_lmo(
                global_grad_send,
                pack_dst_views=workspace.get("pack_dst_views"),
            )

            dist.all_to_all_single(
                global_grad_recv,
                global_grad_send,
                output_split_sizes=self._fsdp_global_output_splits_elems,
                input_split_sizes=self._fsdp_global_input_splits_elems,
                group=fsdp_group,
            )

            # Global prep: fill all full_g_bufs in a single dispatch before LMO loop.
            # This moves all full_g assembly outside the LMO loop entirely.
            src_flat = workspace["full_g_prep_src_flat"]
            dst_flat = workspace["full_g_prep_dst_flat"]
            if src_flat:
                torch._foreach_copy_(dst_flat, src_flat)

            # Clean LMO loop: only lmo() + optional reduce + optional norm.
            # Reverse-pack source view creation is moved outside this loop.
            bucket_norm_dicts: list[dict | None] = [None] * total_buckets

            with record_function("disco.fsdp_lmo_loop"):
                for bucket_idx in range(total_buckets):
                    start_idx, end_idx = self._fsdp_bucket_ranges[bucket_idx]
                    param_kwargs_me = self._fsdp_param_kwargs_me[bucket_idx]
                    bucket_group_indices = self._fsdp_bucket_group_indices[bucket_idx]
                    my_param_in_bucket = (start_idx + rank) < end_idx

                    # Captured before the lmo() call -- full_g_bufs[bucket_idx]
                    # is a step-persistent workspace buffer (filled once above,
                    # untouched again until next step), so this is a free
                    # alias, not a new allocation. lmo() never mutates its
                    # input in place.
                    g_keepalive[bucket_idx] = full_g_bufs[bucket_idx]
                    u = self.lmo(full_g_bufs[bucket_idx], **param_kwargs_me)

                    if dp_replicate_mesh and self.extra_reduce_for_HSDP:
                        dist.all_reduce(
                            u, group=dp_replicate_mesh, op=dist.ReduceOp.AVG
                        )

                    # Kept alive regardless of skip_update -- the (now
                    # pre-update) weight-norm/gram block below needs it
                    # whenever we're logging this step, independent of
                    # whether the real update gets applied this call.
                    u_keepalive[bucket_idx] = u

                    if need_to_calculate_norm and my_param_in_bucket:
                        lr, *_ = self.groups_info[bucket_group_indices[rank]]
                        d = calculate_norm(-lr * u, self.norms_to_log)
                        spec = d.pop("spectrum")
                        bucket_norm_dicts[bucket_idx] = d
                        if upd_spectrum_local_flat is not None and bucket_idx < len(
                            self._fsdp_spectrum_offsets
                        ):
                            off = self._fsdp_spectrum_offsets[bucket_idx]
                            upd_spectrum_local_flat[off : off + spec.numel()].copy_(
                                spec
                            )

            if need_to_calculate_norm:
                for b in range(total_buckets):
                    d = bucket_norm_dicts[b]
                    norms_of_update.extend(
                        d.values() if d is not None else padding_norms.values()
                    )

            if not skip_update:
                if any(x is None for x in u_keepalive):
                    raise RuntimeError(
                        "FSDP once-mode unexpected number of LMO outputs: "
                        f"got={sum(x is not None for x in u_keepalive)}, "
                        f"expected={total_buckets}."
                    )
                torch._foreach_copy_(workspace["u_bufs"], u_keepalive)
                # Batch pack: single _foreach_copy_ into persistent flat dst views.
                torch._foreach_copy_(
                    workspace["upd_pack_dst_views_flat"],
                    workspace["u_src_views_flat"],
                )

                with record_function("disco.fsdp_reverse_a2a"):
                    dist.all_to_all_single(
                        global_upd_recv,
                        global_upd_send,
                        output_split_sizes=self._fsdp_global_input_splits_elems,
                        input_split_sizes=self._fsdp_global_output_splits_elems,
                        group=fsdp_group,
                    )
                # Not cleared here -- the (now pre-update) weight-norm/gram
                # block below, which runs before the real apply, still needs
                # every bucket's `u`.

                if global_upd_recv_cast is not None:
                    global_upd_recv_cast.copy_(global_upd_recv)
                elif workspace.get("cast_dst_by_dtype"):
                    for dtype, dvs in workspace["cast_dst_by_dtype"].items():
                        svs = workspace["cast_src_by_dtype"][dtype]
                        torch._foreach_copy_(dvs, svs)
                upd_recv_views = workspace["upd_recv_views"]
                for j, (param_idx, _, _, _) in enumerate(
                    self._fsdp_upd_recv_param_plan
                ):
                    global_updates[param_idx] = upd_recv_views[j]
        else:
            bucket_workspace = self._allocate_fsdp_bucket_workspace(
                cast_dtype=cast_dtype,
                device=device,
                need_to_calculate_norm=need_to_calculate_norm,
                skip_update=skip_update,
            )

            grad_send_scratch = bucket_workspace["grad_send_flat"]
            grad_recv_scratch = bucket_workspace["grad_recv_flat"]
            upd_send_scratch = bucket_workspace["upd_send_flat"]
            upd_recv_scratch = bucket_workspace["upd_recv_flat"]

            for bucket_idx in range(total_buckets):
                start_idx, end_idx = self._fsdp_bucket_ranges[bucket_idx]
                recv_shapes = self._fsdp_recv_shapes[bucket_idx]
                send_shapes = self._fsdp_send_shapes[bucket_idx]
                split_rows = self._fsdp_split_rows[bucket_idx]
                tp_infos = self._fsdp_tp_gather_info[bucket_idx]
                bucket_params = self._fsdp_bucket_params[bucket_idx]
                bucket_group_indices = self._fsdp_bucket_group_indices[bucket_idx]
                param_kwargs_me = self._fsdp_param_kwargs_me[bucket_idx]
                send_numels = self._fsdp_send_numels[bucket_idx]
                recv_numels = self._fsdp_recv_numels[bucket_idx]
                my_param_in_bucket = (start_idx + rank) < end_idx

                send_chunk_offsets = self._fsdp_bucket_send_chunk_offsets[bucket_idx]
                recv_chunk_offsets = self._fsdp_bucket_recv_chunk_offsets[bucket_idx]
                send_total = self._fsdp_bucket_send_total_elems[bucket_idx]
                recv_total = self._fsdp_bucket_recv_total_elems[bucket_idx]
                grad_send_flat = grad_send_scratch[:send_total]
                grad_recv_flat = grad_recv_scratch[:recv_total]

                for i in range(world_size):
                    base = send_chunk_offsets[i]
                    numel = send_numels[i]
                    # Mirror once-mode behavior: phantom slots reuse the last
                    # real param's gradient so every rank runs LMO on a real
                    # tensor, while reverse A2A still materializes only real
                    # params via range(end_idx - start_idx) below.
                    p = bucket_params[i]
                    group_idx = bucket_group_indices[i]
                    param_idx = min(start_idx + i, end_idx - 1)
                    g = self._get_effective_grad_by_group(p, group_idx, param_idx)
                    g_local = self._maybe_unpack_dtensor(g, tp_infos[i])
                    if g_local.dtype != cast_dtype:
                        g_local = g_local.to(cast_dtype)
                    grad_send_flat[base : base + numel].copy_(g_local.reshape(-1))

                dist.all_to_all_single(
                    grad_recv_flat,
                    grad_send_flat,
                    output_split_sizes=recv_numels,
                    input_split_sizes=send_numels,
                    group=fsdp_group,
                )

                recv_views = [
                    grad_recv_flat[
                        recv_chunk_offsets[src] : recv_chunk_offsets[src]
                        + recv_numels[src]
                    ].view(recv_shapes[src])
                    for src in range(world_size)
                ]
                full_g = torch.cat(recv_views, dim=0)
                # Unlike the fast path's full_g_bufs, `full_g` here is a
                # fresh per-iteration local (torch.cat, not a workspace view)
                # that would otherwise be lost once the loop moves to the
                # next bucket_idx -- keep a second reference alive for gram.
                g_keepalive[bucket_idx] = full_g
                u = self.lmo(full_g, **param_kwargs_me)

                if dp_replicate_mesh and self.extra_reduce_for_HSDP:
                    dist.all_reduce(u, group=dp_replicate_mesh, op=dist.ReduceOp.AVG)

                # Kept alive for the (now pre-update) weight-norm/gram block
                # below, which runs before the real apply.
                u_keepalive[bucket_idx] = u

                if not skip_update:
                    upd_send_flat = upd_send_scratch[:recv_total]
                    updates_send_list = list(torch.split(u, split_rows, dim=0))
                    for dst in range(world_size):
                        base = recv_chunk_offsets[dst]
                        u_part = updates_send_list[dst]
                        if u_part.dtype != cast_dtype:
                            u_part = u_part.to(cast_dtype)
                        upd_send_flat[base : base + recv_numels[dst]].copy_(
                            u_part.reshape(-1)
                        )

                    upd_recv_flat = upd_recv_scratch[:send_total]
                    dist.all_to_all_single(
                        upd_recv_flat,
                        upd_send_flat,
                        output_split_sizes=send_numels,
                        input_split_sizes=recv_numels,
                        group=fsdp_group,
                    )
                    for i in range(end_idx - start_idx):
                        base = send_chunk_offsets[i]
                        global_updates[start_idx + i] = upd_recv_flat[
                            base : base + send_numels[i]
                        ].view(send_shapes[i])

                if need_to_calculate_norm:
                    if my_param_in_bucket:
                        lr, *_ = self.groups_info[bucket_group_indices[rank]]
                        upd_norms = calculate_norm(-lr * u, self.norms_to_log)
                        spec = upd_norms.pop("spectrum")
                        if upd_spectrum_local_flat is not None and bucket_idx < len(
                            self._fsdp_spectrum_offsets
                        ):
                            off = self._fsdp_spectrum_offsets[bucket_idx]
                            upd_spectrum_local_flat[off : off + spec.numel()].copy_(
                                spec
                            )
                    else:
                        upd_norms = padding_norms
                    norms_of_update.extend(upd_norms.values())

        # --- Calculate Weight Norms + Gram Metrics (PRE-UPDATE) ---
        # Moved to run BEFORE "Single vectorised apply" below (was "POST-
        # UPDATE"), so `full_weight` here is genuinely pre-update -- needed
        # so calculate_gram_metrics gets W and U simultaneously (see
        # gram_helper.py). `U` is the raw moment (from `g_keepalive`), not
        # the LMO-processed update (from `u_keepalive`) -- see readme.md.
        # Same all_to_all_single as before, just earlier -- no new
        # communication. track_param_* keeps its historical post-update
        # meaning via a cheap local pseudo-weight (_pseudo_post_update_weight)
        # instead of re-reading the parameter a second time after the real
        # apply.
        if need_to_calculate_norm:
            param_send_scratch = (
                bucket_workspace["param_send_flat"] if bucket_workspace else None
            )
            param_recv_scratch = (
                bucket_workspace["param_recv_flat"] if bucket_workspace else None
            )
            for bucket_idx in range(total_buckets):
                start_idx, end_idx = self._fsdp_bucket_ranges[bucket_idx]
                my_param_in_bucket = (start_idx + rank) < end_idx

                recv_shapes = self._fsdp_recv_shapes[bucket_idx]
                recv_numels = self._fsdp_recv_numels[bucket_idx]
                send_numels = self._fsdp_send_numels[bucket_idx]
                tp_infos = self._fsdp_tp_gather_info[bucket_idx]
                send_total = self._fsdp_bucket_send_total_elems[bucket_idx]
                recv_total = self._fsdp_bucket_recv_total_elems[bucket_idx]
                send_chunk_offsets = self._fsdp_bucket_send_chunk_offsets[bucket_idx]
                recv_chunk_offsets = self._fsdp_bucket_recv_chunk_offsets[bucket_idx]
                if param_send_scratch is None or param_recv_scratch is None:
                    param_send_flat = torch.empty(
                        send_total, dtype=cast_dtype, device=device
                    )
                    param_recv_flat = torch.empty(
                        recv_total, dtype=cast_dtype, device=device
                    )
                else:
                    param_send_flat = param_send_scratch[:send_total]
                    param_recv_flat = param_recv_scratch[:recv_total]

                for i in range(world_size):
                    param_idx = start_idx + i
                    p = fsdp_params[param_idx if param_idx < end_idx else end_idx - 1]

                    tp_info = tp_infos[i]
                    p_local = self._maybe_unpack_dtensor(p, tp_info)
                    if p_local.dtype != cast_dtype:
                        p_local = p_local.to(cast_dtype)
                    base = send_chunk_offsets[i]
                    param_send_flat[base : base + send_numels[i]].copy_(
                        p_local.reshape(-1)
                    )

                dist.all_to_all_single(
                    param_recv_flat,
                    param_send_flat,
                    output_split_sizes=recv_numels,
                    input_split_sizes=send_numels,
                    group=fsdp_group,
                )
                recv_views = [
                    param_recv_flat[
                        recv_chunk_offsets[src] : recv_chunk_offsets[src]
                        + recv_numels[src]
                    ].view(recv_shapes[src])
                    for src in range(world_size)
                ]
                full_weight = torch.cat(recv_views, dim=0)  # pre-update

                if my_param_in_bucket:
                    u = u_keepalive[bucket_idx]
                    bucket_group_indices = self._fsdp_bucket_group_indices[bucket_idx]
                    lr, _, _, wd, _ = self.groups_info[bucket_group_indices[rank]]
                    pseudo_w = _pseudo_post_update_weight(full_weight, u, lr, wd)
                    w_norms = calculate_norm(pseudo_w, self.norms_to_log)
                    w_spec = w_norms.pop("spectrum")
                    if w_spectrum_local_flat is not None and bucket_idx < len(
                        self._fsdp_spectrum_offsets
                    ):
                        off = self._fsdp_spectrum_offsets[bucket_idx]
                        w_spectrum_local_flat[off : off + w_spec.numel()].copy_(w_spec)

                    # Always called -- cheap no-op when gram_level==0 (see
                    # gram_helper.py), so no extra "is gram active"
                    # flag/branch is needed. `g_raw` (not `u`) -- see comment
                    # above.
                    g_raw = g_keepalive[bucket_idx]
                    gram_metrics = calculate_gram_metrics(
                        full_weight, g_raw, pseudo_w, level=self.gram_level
                    )
                    gram_scalar_values = [
                        gram_metrics[name] for name in self.gram_scalar_names
                    ]
                    if gram_vec_local_flat is not None and bucket_idx < len(
                        self._fsdp_gram_vec_offsets
                    ):
                        off = self._fsdp_gram_vec_offsets[bucket_idx]
                        for vname in self.gram_vector_names:
                            vec = gram_metrics[vname]
                            gram_vec_local_flat[off : off + vec.numel()].copy_(vec)
                            off += vec.numel()

                    owned_param = fsdp_params[start_idx + rank]
                    radial_metrics = calculate_radial_metrics(
                        full_weight,
                        pseudo_w,
                        self._radial_state_by_param_id[id(owned_param)],
                    )
                    radial_values = [
                        radial_metrics[name] for name in RADIAL_METRIC_NAMES
                    ]
                else:
                    w_norms = padding_norms
                    gram_scalar_values = list(gram_padding.values())
                    radial_values = list(radial_padding.values())
                norms_of_weight.extend(w_norms.values())
                norms_of_gram.extend(gram_scalar_values)
                norms_of_radial.extend(radial_values)

        # Single vectorised apply. Runs AFTER weight-norm/gram calculation
        # above (moved from before it) so the full-weight reads above see
        # genuinely pre-update weights.
        if not skip_update:
            self.update_bucket_params(
                fsdp_params,
                global_updates,
                0,
                len(fsdp_params),
                tp_mesh=tp_mesh,
            )

        if need_to_calculate_norm and norms_of_update:
            self._gather_and_log_fsdp(
                norms_of_update,
                norms_of_weight,
                upd_spectrum_local_flat,
                w_spectrum_local_flat,
                fsdp_mesh,
                device,
                fsdp_param_names,
                world_size,
                total_buckets,
                norms_of_gram=norms_of_gram,
                gram_vec_local_flat=gram_vec_local_flat,
                norms_of_radial=norms_of_radial,
            )

        if dp_replicate_mesh is not None:
            dist.barrier(group=dp_replicate_mesh.get_group())

    @record_function("disco._prepare_gradients_and_momentum")
    @torch.no_grad()
    def prepare_gradients_and_momentum(self) -> None:
        """
        Fused pre-pass that updates momentum buffers for *all* parameters
        with available grads:
            buf <- (1 - m) * buf + m * g
        It performs foreach-kernel updates per (device, dtype, momentum).

        Uses pre-built momentum buffer lists (_momentum_state) to avoid
        per-step state dict lookups.

        ************************************************************************
        Notes:
        - momentum schedule can change at runtime (e.g. 0.0 -> 0.9)
        - params without grad in the current step are skipped
        ************************************************************************
        """
        self._refresh_momentum_execution_plan()

        for m, bufs, params in self._momentum_execution_plan:
            active_bufs = []
            active_grads = []
            for buf, p in zip(bufs, params):
                g = p.grad
                if g is not None:
                    active_bufs.append(buf)
                    active_grads.append(g)
            if not active_bufs:
                continue
            torch._foreach_mul_(active_bufs, 1.0 - m)
            torch._foreach_add_(active_bufs, active_grads, alpha=m)

    def _raw_effective_grad(self, p, group_idx):
        """
        grad, or momentum-blended buffer if momentum is in (0,1) -- no
        pre-norm, no gather/dtype-cast. Shared by get_momentum_or_grad
        (which adds row pre-norm + optional gather on top) and
        _apply_reduce_pre_norm_pass (which adds col/mat pre-norm via the
        batched all-reduce path). Looked up per-param (not once per shape
        group) since a shape group can mix params from different groups
        that happen to share the same pre_norm/shape/eps.
        """
        g = p.grad
        if g is None or not p.requires_grad:
            return None
        _, nesterov, momentum, _, _ = self.groups_info[group_idx]
        use_momentum = (not self.is_light) and (0.0 < momentum < 1.0)
        if not use_momentum:
            return g
        buf = self._momentum_buffer_by_param_id.get(id(p))
        if buf is None:
            raise ValueError(
                "Momentum buffer missing; ensure pre-pass ran before calling _raw_effective_grad."
            )
        return buf if not nesterov else torch.lerp(buf, g, momentum)

    @record_function("disco.apply_reduce_pre_norm_pass")
    def _apply_reduce_pre_norm_pass(self):
        """
        Col/mat pre-norm: computes the pre-normed effective gradient for
        every param whose group's pre_norm is a col/mat variant, caching
        results in self._pre_normed_grad_cache (keyed by id(p)) for
        get_momentum_or_grad/get_momentum_or_grad_list/
        _get_effective_grad_by_group to return directly. Row category needs
        no pass -- applied inline in those 3 functions instead.

        Batched by shape group (self._pre_norm_reduce_shape_groups, built
        once at init): one torch.stack + one vectorized reduction per group
        instead of a per-param Python loop, and every FSDP-sharded group's
        partial sum-of-squares is packed into ONE buffer for a single
        dist.all_reduce, regardless of how many groups/params exist.

        Known limitation (TP composition is out of scope for this pass, per
        the FSDP+TP scoping decision -- the same applies here to DDP+TP):
        the `sharded=False` ("already full") branch below does a plain
        `.to_local()` unwrap, not the TP-aware gather `_prepare_ddp_lmo`
        uses for its own LMO input. A DDP or non-sharded-embed param that is
        ALSO TP-sharded would see only its local TP shard here, not the
        true full matrix -- col/mat pre-norm on such a param is
        approximate, not solved in this pass.

        Peak memory: for every col/mat param, `torch.stack` (into `stacked`)
        and the apply step (into `normed`) each allocate a full copy the
        same size as that param's local effective-gradient shard -- on top
        of the original grad/momentum-buffer tensor, which stays alive too
        (nothing here frees it). So a param going through this pass briefly
        holds ~2-3x its shard size in memory rather than 1x, for however
        long `group_raws`/`group_partials`/the cache entry stay referenced.
        The 3 fetchers `.pop()` (not `.get()`) their cache entry, so a
        param's `_pre_normed_grad_cache` entry is released the moment it's
        consumed rather than lingering until next step's cache reset --
        that bounds the cache's own contribution, but `stacked`/`group_raws`
        for a whole shape group stay alive until every entry in that group
        has been through Phase 2, so the transient 2-3x is real if every
        parameter uses col/mat pre_norm. Row category has none of this
        overhead (no stack, no cache, applied directly to the fetched
        tensor).
        """
        self._pre_normed_grad_cache = {}
        if not self._pre_norm_reduce_shape_groups:
            return

        group_raws: dict[tuple, torch.Tensor] = {}
        group_entries: dict[tuple, list[tuple]] = {}
        group_partials: dict[tuple, torch.Tensor] = {}

        for key, entries in self._pre_norm_reduce_shape_groups.items():
            sharded, pre_norm, _shape, eps = key
            raws: list[torch.Tensor] = []
            kept_entries: list[tuple] = []
            for p, group_idx in entries:
                g = self._raw_effective_grad(p, group_idx)
                if g is None:
                    continue
                raws.append(g.to_local() if isinstance(g, DTensor) else g)
                kept_entries.append((p, group_idx))
            if not raws:
                continue
            stacked = torch.stack(raws)
            if sharded:
                group_raws[key] = stacked
                group_entries[key] = kept_entries
                group_partials[key] = PRE_NORM_PARTIAL_FUNCTIONS[pre_norm](stacked)
            else:
                # Already the full tensor (ddp/experts/non-sharded embed) --
                # apply directly, no all-reduce needed.
                normed = PRE_NORM_FULL_FUNCTIONS[pre_norm](stacked, eps)
                for i, (p, _gidx) in enumerate(kept_entries):
                    self._pre_normed_grad_cache[id(p)] = normed[i]

        if group_partials:
            # _pack_segments concatenates along dim 0 (torch.cat), so
            # multi-dim / differently-shaped partials (col partials are
            # [N, cols], mat partials are [N]) must be flattened first --
            # reshaped back via partial.shape after the all-reduce below.
            keys = list(group_partials.keys())
            packed, offsets = _pack_segments(
                [(str(i), group_partials[k].reshape(-1)) for i, k in enumerate(keys)]
            )
            dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=self._fsdp_group)
            for i, key in enumerate(keys):
                _sharded, pre_norm, _shape, eps = key
                partial = group_partials[key]
                reduced = packed[
                    offsets[str(i)] : offsets[str(i)] + partial.numel()
                ].view(partial.shape)
                normed = PRE_NORM_SHARDED_APPLY_FUNCTIONS[pre_norm](
                    group_raws[key], reduced, eps
                )
                for j, (p, _gidx) in enumerate(group_entries[key]):
                    self._pre_normed_grad_cache[id(p)] = normed[j]

    def _apply_full_pre_norm_if_needed(self, g, group_idx):
        """
        For gather_to_local=True callers: after gathering to the full
        tensor, apply col/mat pre-norm fresh (no all-reduce needed once
        fully materialized -- this is the only caller of this shape today,
        step_embedding's norm-logging re-fetch, gated behind
        need_to_calculate_norm, not the hot path). Row is already applied
        earlier (works identically pre- or post-gather); identity is a
        no-op.
        """
        pre_norm = self.groups_pre_norm.get(group_idx, "identity")
        if pre_norm == "identity":
            return g
        if pre_norm_category(pre_norm) in ("col", "mat"):
            return PRE_NORM_FULL_FUNCTIONS[pre_norm](
                g, self.groups_pre_norm_eps[group_idx]
            )
        return g

    @record_function("disco.get_momentum_or_grad")
    def get_momentum_or_grad(
        self, p, momentum, nesterov, group_idx, gather_to_local=False
    ):
        """
        Retrieves the effective gradient for a parameter.
        Assumes the momentum buffer has already been updated in a pre-pass.
        """
        if not gather_to_local:
            cached = self._pre_normed_grad_cache.pop(id(p), None)
            if cached is not None:
                return cached

        g = p.grad
        if g is None or not p.requires_grad:
            return None

        use_momentum = momentum > 0 and momentum < 1

        if not self.is_light and use_momentum:
            buf = self._momentum_buffer_by_param_id.get(id(p))
            if buf is None:
                raise ValueError(
                    "Momentum buffer missing; ensure pre-pass ran before calling get_momentum_or_grad."
                )
            g = buf if not nesterov else torch.lerp(buf, g, momentum)

        g = self._apply_row_pre_norm(g, group_idx)

        if gather_to_local and isinstance(g, DTensor):
            g = g.redistribute(placements=[Replicate()] * g.device_mesh.ndim).to_local()
            g = self._apply_full_pre_norm_if_needed(g, group_idx)

        return g

    @record_function("disco.get_momentum_or_grad_list")
    def get_momentum_or_grad_list(
        self,
        params,
        momentum,
        nesterov,
        group_idx,
        gather_to_local: bool = False,
        to_local: bool = False,
    ):
        """
        Batched variant of get_momentum_or_grad for a parameter list.
        Returns effective gradients in input order (None where unavailable).
        """
        outputs: list[torch.Tensor | None] = [None] * len(params)
        use_momentum = (not self.is_light) and (0.0 < momentum < 1.0)

        for i, p in enumerate(params):
            if not gather_to_local:
                cached = self._pre_normed_grad_cache.pop(id(p), None)
                if cached is not None:
                    outputs[i] = cached
                    continue

            g = p.grad
            if g is None or not p.requires_grad:
                continue

            if use_momentum:
                buf = self._momentum_buffer_by_param_id.get(id(p))
                if buf is None:
                    raise ValueError(
                        "Momentum buffer missing; ensure pre-pass ran before calling get_momentum_or_grad_list."
                    )
                g = buf if not nesterov else torch.lerp(buf, g, momentum)

            g = self._apply_row_pre_norm(g, group_idx)

            if gather_to_local and isinstance(g, DTensor):
                g = g.redistribute(
                    placements=[Replicate()] * g.device_mesh.ndim
                ).to_local()
                g = self._apply_full_pre_norm_if_needed(g, group_idx)
            elif to_local and isinstance(g, DTensor):
                g = g.to_local()
            outputs[i] = g

        return outputs

    @record_function("disco.update_bucket_params")
    def update_bucket_params(self, params, updates, start_idx, end_idx, tp_mesh=None):
        slice_params = params[start_idx:end_idx]
        slice_updates = updates[: (end_idx - start_idx)]

        prepared = []
        if tp_mesh is not None:
            # Use pre-computed TP slicing info to skip per-param isinstance/placements/tp_axis
            # and avoid calling p.to_local() just for a shape check.
            tp_slice_info = self._tp_slice_info
            for p, u in zip(slice_params, slice_updates):
                if u is None:
                    prepared.append((p, None))
                    continue
                info = tp_slice_info.get(id(p))
                if info is not None:
                    _, _, _, slicer, local_shape = info
                    # Apply TP slicing only when the update is the full (non-sharded) tensor.
                    if u.shape != local_shape:
                        u = u[slicer]
                prepared.append((p, u))
        else:
            prepared = list(zip(slice_params, slice_updates))

        # ------------- Phase 2: foreach buckets -------------
        # Use a plain dict (pre-sized) instead of a defaultdict with a lambda closure.
        buckets: dict[tuple, dict] = {}
        param_group_map = self.parameters_to_groups

        for p, u in prepared:
            if u is None:
                continue
            lr, _, _, wd, _ = self.groups_info[param_group_map[id(p)]]
            p_local = self._get_param_local_view(p)

            # Robust shape check after TP slicing.
            if p_local.shape != u.shape:
                # DTensor local views are expected to be stable, but refresh lazily if
                # shape changed due to an unexpected runtime redistribution.
                if isinstance(p, DTensor):
                    p_local = p.to_local()
                    self._param_local_views[id(p)] = p_local
                if p_local.shape != u.shape:
                    raise ValueError(
                        f"Shape mismatch between parameter shard {p_local.shape} and update slice {u.shape}. "
                        f"Ensure you pass the same DDP/FSDP window and slice TP updates. "
                    )

            # dtype/device consistency for foreach.
            if u.dtype != p_local.dtype:
                u = u.to(p_local.dtype)

            key = (p_local.device, p_local.dtype, float(lr), float(wd))
            if key not in buckets:
                buckets[key] = {
                    "orig": [],
                    "locals": [],
                    "updates": [],
                    "lr": lr,
                    "wd": wd,
                }
            b = buckets[key]
            b["orig"].append(p)
            b["locals"].append(p_local)
            b["updates"].append(u)

        for (_, _, lr, wd), data in buckets.items():
            if not data["locals"]:
                continue
            if wd != 0.0:
                torch._foreach_mul_(data["locals"], 1.0 - wd * lr)
            torch._foreach_add_(data["locals"], data["updates"], alpha=-lr)
