# DiSCO optimizer

`DiSCO` (`disco.py`) is a Muon/orthogonalized-update-style optimizer: instead of applying the raw
gradient, it computes an LMO ("linear minimization oracle") update via `AbstractDiSCO.lmo`
(`abstract_disco.py`) — typically a Newton-Schulz zeropower iteration that orthogonalizes the
gradient matrix (or a cheaper per-row normalization for embeddings, see below) — then applies
`w = w*(1 - wd*lr) - lr*u`.

This file documents **how DiSCO handles different parameter types under different parallelism
strategies**, which is most of what's structurally interesting about `disco.py`. Norm/gram/spectrum
tracking (a secondary concern layered on top) is covered at the end.

## Files

- `disco.py` — the `DiSCO` optimizer itself; all parallelism-specific logic lives here.
- `abstract_disco.py` — `AbstractDiSCO` base class: `lmo()`, `normalise_grad()` (the actual
  update-shaping math per `norm_factor`), and norm/gram tracking *state*
  (`need_to_calculate_norm`, `norms_to_log`, `gram_metrics_to_log`, `norms_at_current_step`).
- `norm_helper.py` / `gram_helper.py` — see "Norm/gram/spectrum tracking" below.
- `pre_norm_helper.py` — see "Pre-norm: a stage before LMO" below.
- `spectrum_logging.py` — turns raw `track_spectrum_*` tensors into W&B images / Parquet export.
- `readme.md` — this file.

## Why parameter type matters here (not just "which mesh shards it")

A parameter is routed to one of 5 handlers, and **the routing is driven by the LMO algorithm the
parameter needs, not simply by which mesh(es) shard it**. This is the thing most worth
internalizing before touching this file:

| Handles | Routed by | Algorithm | Needs full (unsharded) matrix for the *update itself*? |
|---|---|---|---|
| `step_scalar` | `p.numel() == 1` | `sign(grad)` | n/a (scalar) |
| `step_embedding` | `backend == "identity"` and `norm_factor` starts with `embed`/`unembed` | per-**row** L2 normalization (`fused_embed_linear` etc., `abstract_disco.py`) | **No** — row-separable |
| `step_experts` | structural: `ndim == 3` (MoE routed experts) | Newton-Schulz orthogonalization, per expert | No extra gather — EP shards along the *expert* axis, so each rank already holds each of its owned experts' full 2-D matrix |
| `step_ddp` | structural: not FSDP/EP-sharded | Newton-Schulz orthogonalization | Cheap — `dp_replicate` means every rank already has a full replica (at most a TP-only gather) |
| `step_fsdp` | structural: FSDP-sharded | Newton-Schulz orthogonalization | **Yes, expensive** — FSDP shards along the matrix's own row dimension, so the whole matrix must be reconstructed via `all_to_all_single` before the algorithm can run |

Routing is decided in `_build_param_lists` (called once at optimizer construction, cached
thereafter): scalars first, then `_is_embed_group(group)` (checks `backend`/`norm_factor` on the
param's *group*, i.e. a config/algorithm choice — **completely independent of how the parameter is
actually sharded**), then `get_param_type(p, fsdp_enabled, expert_enabled)` (a structural check —
`ndim == 3` → Expert, else FSDP or DDP depending on `fsdp_enabled`) for everything else.

### The key correction: `step_embedding` params *can* be FSDP-sharded

An earlier version of this doc claimed `step_embedding` params never carry an FSDP shard. That's
wrong. Embedding/unembedding parameters are typically large (vocab-sized) and commonly *are*
FSDP-sharded (row-sharded, same as any other FSDP param) in real configs — `step_embedding` never
routes through the structural `get_param_type` check at all (it short-circuits on the group's
`backend`/`norm_factor` before that check ever runs), so an FSDP-sharded embedding weight still
ends up in `step_embedding`, not `step_fsdp`.

The reason `step_embedding` doesn't need `step_fsdp`'s expensive bucketed `all_to_all_single`
reconstruction is **not** "no sharding survives to this path" — it's that the embedding/unembedding
LMO (`fused_embed_linear`/`fused_embed_sqrt`/`fused_unembed_linear`/`fused_unembed_sqrt` in
`abstract_disco.py`) normalizes **per row** (`row_l2_norm = g.pow(2).sum(dim=-1, ...).sqrt()`,
purely local to each row) — unlike the Newton-Schulz orthogonalization `step_ddp`/`step_fsdp`/
`step_experts` use, which is not row-separable and genuinely needs the whole matrix at once. Since
FSDP shards along dim 0 (rows), each rank's local shard already contains everything the embedding
LMO needs for *its own rows* — zero communication required for the update itself. `step_embedding`
still calls `p.full_tensor()` / `get_momentum_or_grad(..., gather_to_local=True)`, but only inside
the *norm-logging* loop (SVD-based norms genuinely do need the whole matrix), never for the update
path. That's a real, separate collective per parameter, not batched the way `step_fsdp` batches
multiple params into one bucket-wide `all_to_all_single` — a plausible future optimization if the
embed param count/sharding ever makes it matter, not something built today.

## Per-path structure (high level — see the docstrings/comments in each `step_*` for exact mechanics)

- **`step_scalar`**: trivial — `sign(grad)`, `@torch.compile()`-decorated, no distributed
  reconstruction of anything.
- **`step_embedding`**: gradients fetched per-param (`gather_to_local=False` for the actual
  update — local shard is enough), LMO applied locally, update applied via
  `_update_embed_params_fast` (batched by shape where possible for `_foreach_*` fusion). Norm/gram
  logging (when `need_to_calculate_norm`) additionally gathers full tensors per param — see below.
- **`step_experts`**: MoE routed-expert weights, shape `(num_experts_per_block, D, D')`, grouped
  into same-shape "blocks" (`_expert_blocks`, `_precompute_experts_metadata`) so all experts in a
  block LMO together as one batched call. Expert-Parallel shards along the *expert* axis over the
  FSDP mesh (`ep_per_rank = ceil(num_experts / fsdp_mesh.size())`) — each rank's owned experts are
  already complete 2-D matrices, no reconstruction collective needed.
- **`step_ddp`**: DDP-replicated params (`ndim <= 2` only — MoE/3-D params are forced through
  `step_experts`/`step_fsdp` instead, see the `invalid_ddp_params` check in
  `_precompute_ddp_metadata`). Each rank computes LMO for its own "owned" subset
  (`_ddp_owned_indices`, a round-robin partition purely for *avoiding duplicate work/logging*
  across replicas — not sharding, since `dp_replicate` means full replicas), then one flat
  `all_gather` distributes everyone's computed updates to everyone (Phase B) before the batched
  apply (Phase C).
- **`step_fsdp`**: the expensive path. Params are grouped into buckets of `world_size` params each
  (`_fsdp_bucket_ranges`); each bucket does a forward `all_to_all_single` to reconstruct the full
  gradient for whichever param this rank owns in that bucket, runs LMO, then a reverse
  `all_to_all_single` to re-scatter the update back to shards for the apply. Two implementations
  exist, chosen by `DISCO_FSDP_A2A_MODE` env var (default `"once"`):
  - `"once"` (`use_global_fast_path`): all buckets' forward/reverse communication is batched into
    **one** pair of global `all_to_all_single` calls (`_prepare_fsdp_lmo` fills one big send
    buffer for everything up front) — the fast path, and the only one exercised by default.
  - `"bucket"`: one `all_to_all_single` pair *per bucket* — a fallback, apparently rarely
    exercised (see the bug below, which only this path hit).

## Pre-norm: a stage before LMO

`AbstractDiSCO.lmo` fuses "orthogonalize" (Newton-Schulz, needs the full matrix) with "post-norm"
(`normalise_grad`). **Pre-norm** is a third, earlier stage applied to the effective gradient (raw
grad, or momentum-blended buffer if `momentum > 0`) **before any communication for LMO** — so it
runs on the raw tensor in its original dtype, before the communication-dtype downcast. Configured
per group via `pre_norm` (defaults to `"identity"`, a no-op), same override mechanism
(`extra_param_group_split_rules`) as `norm_factor`/`backend`.

Only matters for `step_fsdp`/`step_embedding` — `step_ddp`/`step_experts` already have the full
matrix locally (DDP replication; EP shards along the expert axis only), so pre-norm there is a
direct computation, no special handling.

Config values look like `"row-l2"`/`"col-l2"`/`"mat-l2"` — the prefix before the first `-`
(`row`/`col`/`mat`) selects the communication strategy, the full string selects the formula
(`pre_norm_helper.py`'s `PRE_NORM_*` registries), so later variants (`"col-rms"`, ...) slot in
without touching dispatch code:

- **row** (`row-l2`): reduces along dim=-1, which is never the FSDP-sharded dimension — a local
  shard already holds complete rows, so this is **zero communication**. Applied inline, directly to
  `g` (DTensor or plain Tensor, whichever it already is) at the exact point it's fetched in
  `get_momentum_or_grad`/`get_momentum_or_grad_list`/`_get_effective_grad_by_group`
  (`_apply_row_pre_norm`) — no separate pass, no cache, no `.to_local()` unwrap. dim=-1 reduction on
  a DTensor dispatches locally per shard, same as any other local op.
- **col**/**mat** (`col-l2`/`mat-l2`): reduce along the FSDP-sharded dimension / the whole matrix —
  genuinely need combining across ranks. `_apply_reduce_pre_norm_pass` (called once per step, right
  after `prepare_gradients_and_momentum`) does this in two batched phases instead of a per-param
  loop:
  1. Groups every col/mat param by `(is_fsdp_row_sharded, exact pre_norm string, local_shard_shape,
     eps)` (`_precompute_pre_norm_metadata`, once at init — mirrors the existing
     `_embed_extra_shape_groups`/FSDP-bucket shape-grouping pattern), then computes **one**
     `torch.stack` + one vectorized reduction per shape group instead of N individual per-param
     calls.
  2. Every FSDP-sharded group's partial sum-of-squares gets packed into **one** buffer
     (`_pack_segments`, same helper `step_ddp`/`step_fsdp` already use for norm-logging fusion) for
     a **single** `dist.all_reduce`, regardless of how many groups or params exist that step.
     Non-sharded groups (DDP/experts/non-FSDP-sharded embed) skip the all-reduce entirely — their
     local view is already the full tensor.

  Results are cached in `self._pre_normed_grad_cache` (keyed by `id(p)`); the 3 effective-grad
  fetchers check this cache first and return directly instead of recomputing. `gather_to_local=True`
  callers (only `step_embedding`'s norm-logging re-fetch, gated behind `need_to_calculate_norm`, not
  the hot path) bypass the cache and reapply the full-tensor formula fresh once already gathered —
  mathematically identical, no all-reduce needed once the data is whole.

**Known limitation — TP composition is out of scope.** A param that's *both* FSDP/DDP-sharded and
TP-sharded isn't handled correctly: row-norm assumes dim=-1 isn't TP-sharded (breaks under
TP col-parallel, `Shard(dim=1)`); col/mat's "already full" branch for DDP/experts/non-sharded embed
does a plain `.to_local()`, not `_prepare_ddp_lmo`'s TP-aware gather. Pre-norm is approximate under
TP composition, not solved in this pass — same scoping decision made explicitly up front, not
discovered as a gap.

**Known limitation — 1-D params aren't supported.** Row/col/mat all assume a matrix shape
`[rows, cols]` where FSDP shards dim 0 and dim=-1 is a separate, unsharded axis. For a genuinely
1-D param (e.g. a bias vector — a real, supported case elsewhere, see `lmo()`'s `ndim==1` branch),
dim=-1 *is* dim 0 *is* the sharded dim, so row's "dim=-1 is never sharded" premise and col/mat's
row-vs-column distinction both collapse — worse, silently stacking several different 1-D params
together would reduce *across params* instead of within one. `_precompute_pre_norm_metadata` raises
a clear `ValueError` at init if a non-`identity` `pre_norm` is configured on a `<2`-D param (and
scalar/`step_scalar` groups assert `pre_norm == "identity"` outright) rather than computing
something silently wrong — use `"identity"` for bias-like params for now.

## A real, pre-existing bug found while reviewing this file

`step_fsdp`'s bucketed (`"bucket"` mode) branch read `device=workspace["device"]` when allocating
`bucket_workspace`. **`"device"` was never actually a key in the `workspace` dict, in either
mode** — `_create_fsdp_step_workspace`/`_allocate_fsdp_once_workspace` never set it, and `step()`'s
dispatch only even *builds* a `workspace` dict when `fsdp_a2a_mode == "once"` (it stays `None`
otherwise). So setting `DISCO_FSDP_A2A_MODE=bucket` with any FSDP params crashed on the very first
optimizer step (`TypeError: 'NoneType' object is not subscriptable`) — **the bucketed fallback mode
was completely non-functional**. Fixed by using `step_fsdp`'s own local `device` variable (already
computed at the top of the function from `fsdp_params[0].device`) instead of threading it through
`workspace`. This bug predates all other work in this doc (confirmed via `git show HEAD`) — it
wasn't introduced by the norm/gram work below, just found while reading through the whole file.
If you ever need `"bucket"` mode for real, re-verify it end-to-end — this path looks
under-exercised (this bug would have been immediately obvious the first time it actually ran).

## Norm/gram/spectrum tracking

Every path also optionally logs, per parameter, gated by a single flag
`self.need_to_calculate_norm` (set externally via `calculate_norm_at_next_step()`, driven by
`config.metrics.log_norm_freq`):

- `track_update_*` — scalar norms (`norm_helper.calculate_norm`) of the update actually applied
  this step (`-lr * u`).
- `track_param_*` — scalar norms of the weight **after** this step's update. In 3 of 4 paths
  (`step_embedding`/`step_ddp`/`step_fsdp`) this is a cheap **derived pseudo-value**
  (`_pseudo_post_update_weight(w, u, lr, wd)`, mirroring the real apply formula) rather than a
  second real read, because the true pre-update `w` is needed in-scope for gram metrics (see
  below) — `step_experts` is pre-update natively, a known, deliberately-unfixed inconsistency
  across paths.
- `track_spectrum_*` — raw singular-value vectors (`update`/`param`), consumed by
  `spectrum_logging.py`.
- `track_gram_*` — functions of `(W, U)` pairs (`gram_helper.calculate_gram_metrics`), e.g.
  alignment between a weight and its update. `gram_helper.GRAM_METRIC_FUNCTIONS` is **currently
  empty** — pure plumbing, no formulas decided yet. Every call site calls
  `calculate_gram_metrics` unconditionally (cheap when empty — no SVD, returns `{}` instantly) and
  only branches on the result's truthiness where required (e.g. before `torch.stack(...)` for a
  collective) — there is deliberately no separate "is gram active" flag anywhere.

### W/U simultaneity (why 3 of 4 paths got reordered)

`calculate_gram_metrics(W, U)` needs the full weight and its update simultaneously in scope.
Historically `track_param_*` was computed *after* the real update was applied, so `U` was already
out of scope. Fixed by reordering — moving the point where the real update gets applied to run
*after* norm/gram calculation instead of before, in `step_embedding`, `step_ddp`, `step_fsdp`
(`step_experts` already computed weight-norm pre-update, no change needed). Critically, **no
collective changed** in any of these — only the order of two already-existing blocks. `U` needed a
small amount of extra lifetime in some paths (`step_fsdp` needed a new `u_keepalive` list; `step_ddp`'s
`local_updates` was already alive long enough) — "a bit more temporary memory held a little
longer," not new communication.

**Real bug found and fixed here**: when first reordering `step_ddp`'s weight-norm loop, an
`if u is None: continue` was copied from the neighboring update-norm loop. That's wrong for
weight-norm specifically: `local_updates[my_idx] is None` (no gradient this step) does not mean
the real apply skips the param — Phase B substitutes a zero update (`zero_by_shape`) and weight
decay still applies. Skipping would have silently zeroed `track_param_*` for any DDP param lacking
a gradient on a logging step. **Why can `u` be `None` at all?** `p.grad is None` happens in
ordinary situations — gradient accumulation boundaries, pipeline-parallel stages not executed this
micro-batch, `requires_grad=False` params still tracked, or a param genuinely untouched by this
step's forward/backward. Fix: substitute `torch.zeros_like(w)` and let the computation proceed
normally, matching what the real apply does, rather than skipping the param.

### Communication fusion

`step_experts` already fused everything (update/weight/gram norms + both spectrum halves) into one
`torch.cat` + one `all_gather_tensor`. `step_ddp`'s Phase D and `step_fsdp` used to issue up to 5
separate `all_gather_tensor` calls each; both now use the same pattern via a shared helper,
`_pack_segments(segments) -> (buffer, offsets)`, which derives offsets from what was *actually*
packed rather than hand-computing them (hand-derived offsets are exactly the kind of thing that
caused the `step_ddp` bug above) — see `_pack_segments`'s docstring and the `_gather_and_log_fsdp`
method for the pattern.

### If you're adding a real gram metric formula

1. Add it to `gram_helper.py` and register: `GRAM_METRIC_FUNCTIONS["my_metric"] = my_metric`. `W`/`U`
   arrive already unwrapped (DTensor/Parameter → plain local tensor), diag-embedded if originally
   1-D, transposed if requested — same contract as `norm_helper.NORM_FUNCTIONS`.
2. No `disco.py` call-site changes needed — `self.gram_metrics_to_log` (built from
   `GRAM_METRIC_FUNCTIONS.keys()`) picks it up automatically.
3. No vector-valued ("spectrum-analog") gram output exists yet — only scalars are wired up; that's
   new plumbing if a metric ever needs it.
4. Decide what your formula should do with an all-zero `U` (the `step_ddp` zero-substitution case
   above feeds a genuine zero update into `calculate_gram_metrics` when a param had no gradient) —
   handle it explicitly rather than relying on incidental float behavior.
