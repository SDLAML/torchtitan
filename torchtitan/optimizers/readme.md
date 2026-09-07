# DiSCO optimizer

`DiSCO` (`disco.py`) is a Muon/orthogonalized-update-style optimizer: instead of applying the raw
gradient, it computes an LMO ("linear minimization oracle") update via `AbstractDiSCO.lmo`
(`abstract_disco.py`) — typically a Newton-Schulz zeropower iteration that orthogonalizes the
gradient matrix (or a cheaper per-row normalization for embeddings, see below) — then applies
`w = w*(1 - wd*lr) - lr*u`.

## DDP-only AUS research prototype

The optional AUS path replaces the scalar DiSCO learning rate for each 2-D
matrix with

`eta_t = AUS(t) * N(W_t) / N(U_t - (W_t / N(W_t)) * phi_t(U_t))`.

AUS is a first-order angular target. At a finite learning rate, the actual
change in normalized weight direction need not equal the scheduled AUS.

Enable the paired optimizer and scheduler settings:

```text
--optimizer.aus_enabled
--lr_scheduler.schedule_type aus
--lr_scheduler.aus_coefficient 0.5
```

The coefficient defaults to `0.5`, giving the shared schedule
`AUS(t) = 0.5 / sqrt(t)` with `t=1` for the first optimizer update. The
scheduler deliberately resets every corrected parameter group to this shared
coefficient; separate coefficients per group are not part of this prototype.

When resuming a full checkpoint, the saved step number is retained and the
current run's `aus_coefficient` determines the LR, including the first resumed
update. This also applies when switching from WSD or changing the coefficient;
`checkpoint.reconfigure_lrs` is not required for AUS.

`aus_enabled` is taken from the current run's configuration and is omitted
from TorchTitan's flattened optimizer checkpoint schema, so checkpoints
predating AUS do not need this field. Other existing checkpoint requirements
(such as momentum and radial state) still apply.

The correction is recomputed from the current weight and post-LMO update
immediately before every update. `spectral`/`rmnp_row_norm_rms_rms` use the
RMS-to-RMS operator norm, `rmnp_row_norm` and `unembed_*` use RMS-to-infinity,
and `embed_*` uses l1-to-RMS on the logical transposed embedding matrix. Vector
and scalar parameters use the nominal AUS directly.

This path intentionally supports only dense replicated DDP (or a single
process): FSDP, TP, expert/MoE tensors, split logical matrices, and tensors
above rank 2 fail during optimizer construction. Corrected DDP displacements
are communicated in float32. On norm-logging steps, `track_aus_correction/*`,
`track_aus_eta/*`, and `track_aus_valid/*` expose the chosen values. A validity
of zero means the norm was nondifferentiable or the tangent was degenerate
(including zero and purely radial updates), so the correction fell back to
one. Non-finite matrix inputs or correction calculations fail before any
parameter weights are updated, including embeddings. Momentum preparation
has already run at that point; a failed step must not be retried in place.

This file documents **how DiSCO handles different parameter types under different parallelism
strategies**, which is most of what's structurally interesting about `disco.py`. Norm/gram/spectrum
tracking (a secondary concern layered on top) is covered at the end.

## Files

- `disco.py` — the `DiSCO` optimizer itself; all parallelism-specific logic lives here.
- `abstract_disco.py` — `AbstractDiSCO` base class: `lmo()`, `normalise_grad()` (the actual
  update-shaping math per `norm_factor`), and norm/gram tracking *state*
  (`need_to_calculate_norm`, `norms_to_log`, `gram_level`/`gram_scalar_names`/`gram_vector_names`,
  `norms_at_current_step`).
- `norm_helper.py` / `gram_helper.py` — see "Norm/gram/spectrum tracking" below.
- `pre_norm_helper.py` — see "Pre-norm: a stage before LMO" below.
- `spectrum_logging.py` — turns raw `track_spectrum_*` tensors into W&B images / Parquet export.
- `gram_vector_logging.py` — turns raw `track_gram_*` vector tensors into W&B atlas-grid images
  (index vs. value line plots, one grid per gram metric name — copy-and-adapted from
  `spectrum_logging.py`'s grid/layout machinery, kept as its own module so `spectrum_logging.py`
  stays untouched) and/or a Parquet export (mirrors `spectrum_logging._export_spectrum`) — see
  "Norm/gram/spectrum tracking" below.
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
- `track_param_*` — scalar norms of the weight **after** this step's update. All 4 paths use a
  cheap **derived pseudo-value** (`pseudo_w = _pseudo_post_update_weight(w, u, lr, wd)`, mirroring
  the real apply formula) rather than a second real read, because the true pre-update `w` is needed
  in-scope for gram metrics (see below) — `step_experts` didn't compute this until it also needed
  `pseudo_w` as gram's `W_after` (see "Norm/gram/spectrum tracking" → gram below); now all 4 paths
  are consistent.
- `track_spectrum_*` — raw singular-value vectors (`update`/`param`), consumed by
  `spectrum_logging.py`.
- `track_gram_*` — functions of `(W_before, V_raw, W_after)` triples (`gram_helper.
  calculate_gram_metrics`): `W_before` is the pre-update weight, `V_raw` is the **raw** effective
  grad/momentum (whatever's fed into `self.lmo()` -- see "V_raw is the raw moment, not the LMO
  update" below), and `W_after` is `pseudo_w` (the same post-update approximation `track_param_*`
  already uses, not a fresh real read). `U = W_after - W_before` (the exact realised displacement)
  and `A = -U` are derived internally -- see `gram_helper.py`'s module docstring and
  `gram_matrix.md` for the full formula catalogue and the "which tensor answers which question"
  framing. Gated by a single cumulative `self.gram_level: int` (0 = off, no-op; 1/2/3 = increasingly
  expensive, each level includes all lower levels), set alongside `norms_to_log` via
  `calculate_norm_at_next_step(norms_to_log, gram_level)`, driven by `config.metrics.gram_level`
  (mirrors `config.metrics.norms_to_log`/`log_norm_freq` exactly -- same cadence, no separate gate).
  `calculate_gram_metrics` returns a mix of 0-d scalar and 1-d vector tensors (fixed key SET per
  level, independent of parameter shape -- disco.py's DDP/FSDP/experts packing code relies on this;
  ~121 keys at level 3, comparing 4 tensors -- `W_before`, `W_after`, `V_raw`, `U` -- pairwise).
  Every call site calls `calculate_gram_metrics` unconditionally (cheap no-op at `gram_level=0` --
  returns `{}` instantly) and only branches on the result's truthiness where required (e.g. before
  `torch.stack(...)` for a collective) -- there is deliberately no separate "is gram active" flag
  anywhere. Vector-valued entries get popped and rendered as W&B atlas-grid images (index vs. value
  line plots, one grid per metric name, gated by `config.optimizer.enable_gram_plot`) and/or
  exported to a Parquet file uploaded as a W&B Artifact (gated by
  `config.optimizer.enable_gram_export`, mirrors `spectrum_logging._export_spectrum` exactly), by
  `gram_vector_logging.py` before reaching a scalar logger (copy-and-adapted from the grid/layout
  machinery `spectrum_logging.py` provides for `track_spectrum_*`, kept as a separate module -- see
  that file's docstring for why). A small, opt-in set of metric names (currently just `V_R_raw`,
  see `_MEAN_MIN_MAX_METRIC_NAMES`) also get 3 cheap derived scalars unconditionally
  (`..._mean`/`..._min`/`..._max`, e.g. `track_gram_V_R_raw_mean/...`). Scalars otherwise need no
  handling at all, they're already valid logger values.

### `V_raw` is the raw moment, not the LMO update

`calculate_gram_metrics(W_before, V_raw, W_after, ...)`'s `V_raw` argument is the **raw** effective
grad/momentum -- the exact tensor about to be passed into `self.lmo()` -- not the LMO-processed
update `u = self.lmo(...)` that every path already computes for the real parameter update. This
matches the "study optimizer-state geometry" framing in `gram_matrix.md` (as opposed to "study
weight dynamics", which is what the *realised displacement* `U = W_after - W_before` -- derived
internally inside `gram_helper.py`, not the same `U` as `self.lmo()`'s `u` -- is for).

Every path already computes the raw grad immediately before calling `self.lmo()`, so capturing it
for gram is "keep one more reference alive a little longer", not new compute or communication --
mirrors the existing `u_keepalive`/`pseudo_w` pattern (see "simultaneity" below) exactly, just one
step earlier:
- `step_embedding`: the raw grad `g` is already in scope right where `u = self.lmo(g, ...)` is
  called, a few lines before the gram call -- no new variable needed, just pass `g` instead of `u`.
- `step_ddp`: `lmo_inputs` (from `_prepare_ddp_lmo`, Phase A) already holds the raw per-owned-index
  grad and is never mutated afterward -- reused directly at the later gram call site.
- `step_experts`: a new `all_raw_grads` list, populated from `big_g` (captured right before
  `self.lmo(big_g, ...)`) alongside the existing `all_updates` list.
- `step_fsdp`: a new `g_keepalive` list, populated alongside `u_keepalive` in both the fast path
  (aliases the step-persistent `full_g_bufs[bucket_idx]` workspace buffer -- free) and the slow path
  (keeps a second reference to the freshly `torch.cat`'d `full_g`, which would otherwise be
  discarded once the loop moves to the next bucket).

Verified `AbstractDiSCO.lmo()` never mutates its input tensor in place (every backend, eager and
Triton, only ever rebinds to new tensors or writes into separately-allocated `out=` buffers) --
aliasing the raw-grad reference this way is safe; it will always reflect the true pre-LMO value.

### `W_before`/`W_after`/`V_raw` simultaneity (why all 4 paths compute `pseudo_w`)

`calculate_gram_metrics` needs the pre-update weight, its raw moment, and the post-update weight
simultaneously in scope. Historically `track_param_*` was computed *after* the real update was
applied, so the pre-update weight was already out of scope. Fixed by reordering — moving the point
where the real update gets applied to run *after* norm/gram calculation instead of before, in
`step_embedding`/`step_ddp`/`step_fsdp` (`step_experts` already computed weight-norm pre-update, no
reordering needed there). Critically, **no collective changed** in any of these — only the order of
two already-existing blocks. The update needed a small amount of extra lifetime in some paths
(`step_fsdp` needed a new `u_keepalive` list, now also `g_keepalive`; `step_ddp`'s
`local_updates`/`lmo_inputs` were already alive long enough) — "a bit more temporary memory held a
little longer," not new communication.

All 4 paths pass `pseudo_w = _pseudo_post_update_weight(w, u, lr, wd)` as gram's `W_after` (the same
tensor already used for `track_param_*`). `step_experts` didn't compute `pseudo_w` at all until the
3-tensor gram spec needed it — it used the raw pre-update `p_local[ep_idx]` for both weight-norm and
gram directly. Fixed to fetch `lr`/`wd` once per block (`self.groups_info[self._expert_block_group_idx[block_idx]]`)
and compute `pseudo_w` per-expert, same formula as the other 3 paths.

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

`gram_helper.py` is a cumulative, per-level design (mirrors `norm_helper.fused_metrics`'s
shared-computation pattern): `_build_gram_core` builds every self-/cross-Gram and correlation matrix
once (`G_Wm`/`G_Wp`/`G_V`/`G_U`, `C_Wm`/`C_Wp`/`C_V`/`C_U`, `C_WV`/`C_WU`/`C_VA` — `Wm`/`Wp` = weight
before/after, `V`/`U` = raw momentum / realised displacement), and
`_level1_metrics`/`_level2_metrics`/`_level3_metrics` derive that level's metrics from shared state
(level N is cumulative with levels < N). There's no per-metric registry (`GRAM_METRIC_FUNCTIONS` is
gone).

1. Add the computation inside the right `_level{1,2,3}_metrics` function (or extend `_GramCore`/
   `_Level2Extras` if it needs new shared intermediates).
2. Add its name to the matching level in `GRAM_SCALAR_NAMES_BY_LEVEL` (0-d output) or
   `GRAM_VECTOR_NAMES_BY_LEVEL` (1-d output) — these two dicts are the source of truth for the
   fixed, shape-independent key set `calculate_gram_metrics(..., level=L)` returns; every vector is
   length `m` (see `gram_helper.py`'s module docstring — NOT `min(m, n)`, don't reuse
   `norm_helper`'s spectrum-length tables for gram vectors). Naming convention:
   `{tensor_prefix}_{field}` (`V`/`U`/`Wm`/`Wp` for the 4 self-Gram tensors, `VA`/`WV`/`WU` for the 3
   cross-Gram pairs, `G_*`/`C_*` for eigenspectra, `K_V`/`K_U`/`J`/`Q_*` for level-3 whitened
   quantities) — keep new metrics consistent with this so `gram_vector_logging.py`'s per-metric-name
   atlas grids stay readable.
3. No `disco.py` call-site changes needed for scalars. A **new vector-valued** metric needs its
   contribution counted in the three `_precompute_*_gram_vector_metadata` methods' `n_vec =
   len(self.gram_vector_names)` — already automatic, since those methods re-read
   `self.gram_vector_names` fresh each time they run, but double check the offset math if you
   change a vector's *length formula* rather than just adding another same-length vector.
4. Decide what your formula should do when `W_after == W_before` (the `step_ddp` zero-gradient case
   feeds `u = torch.zeros_like(w)`, so `pseudo_w == w`, hence `U_actual = 0` inside
   `calculate_gram_metrics`) — verified all existing level 1-3 formulas degrade gracefully to
   finite, `eps`-guarded values for this case (traced through by hand + covered in the standalone
   verification script); handle any new formula's zero-input behavior explicitly rather than relying
   on incidental float behavior.
5. Vector metrics automatically get their own atlas grid (dense + MoE) via
   `gram_vector_logging.py` — no changes needed there either, since it discovers grid names
   dynamically from tracked key names (see that file's docstring).

### `step_embedding` gram tracking: large-vocab OOM fixed, cost-gated behind `DISCO_TRACK_EMBED_GRAM`

`gram_helper._gram(X) = X @ X.T` forms an `m x m` matrix, where `m` is the row-count after
orientation. `step_embedding`'s `embed_params` includes the `output`/lm_head weight
(`[vocab_size, hidden_dim]`) alongside `tok_embeddings` (same shape) -- previously, `need_T =
CONST_NAME_OF_EMBEDDING in p_name` only transposed for params literally named `"tok_embeddings"`,
not `"output"`, so the lm_head weight's `m` stayed at `vocab_size` instead of being reduced to
`hidden_dim` -- at a ~200k vocab this was a `[200_000, 200_000]` fp32 matrix (~160GB), an immediate
CUDA OOM (hit in practice, not hypothetical).

**Fixed**: `calculate_gram_metrics` now decides orientation unconditionally from shape (`rows <=
cols`, ignoring any caller-supplied `transpose`/`need_T` -- see `gram_helper.py`'s module docstring
and `calculate_gram_metrics`'s own docstring), so both `tok_embeddings` and `output` always get
`m = hidden_dim`, never `vocab_size`. This is a deliberate, session-wide semantic choice (not
special-cased for embeddings): any parameter with `D_out > D_in` gets the same treatment, e.g. FFN
up-projections now track input-feature-wise dynamics instead of output-channel-wise.

**Still gated, though, on cost rather than correctness**: `tok_embeddings`/`output`'s gram
computation remains far more expensive than any other tracked param, even without the OOM --
`_build_gram_core` materializes several full-size `[hidden_dim, vocab_size]` copies (row-normalized
factors, `U_actual`, `A_actual`), which at a ~200k vocab and multi-thousand hidden dim is on the
order of tens of GB of transient memory and multiple seconds of forced-fp32 GEMM (the small
`[hidden, hidden]` Gram matrix itself is cheap; building it from the full-size factors is not) --
*per parameter, per logging event*, for exactly these two parameters. `calculate_norm_at_next_step`
already lets you tune `gram_level` (and hence gram tracking's cost) per step; `DISCO_TRACK_EMBED_GRAM`
(default `"1"`, or set `optimizer.track_embed_gram = False` directly at runtime) is a second,
independent switch specifically for these two expensive params, so you can e.g. run `gram_level > 0`
every step for cheap layers while only enabling embed/output gram at sparse checkpoints. `step_ddp`/
`step_fsdp`/`step_experts` are unaffected either way (never touch vocab-scale dimensions).

One conceptual note worth keeping in mind when reading `tok_embeddings`'s (as opposed to `output`'s)
gram metrics: `tok_embeddings` is a lookup table, not a jointly-computed Linear layer -- each row's
gradient depends only on whether that token appeared in the batch, with no forward-pass coupling
between different vocab rows. Transposing doesn't change that underlying gradient structure, but it
does change what the resulting `[hidden, hidden]` Gram matrix answers: `G[a, b] = Σ_j
tok_embeddings[j, a] · tok_embeddings[j, b]`, summed over the whole vocabulary, asks about
correlation/redundancy *between hidden dimensions* across the embedding table (a real, studied
quantity -- embedding anisotropy/dimension collapse), not "is this output channel dominant" in the
sense the same metric name means for e.g. `attention.wq`. `output`/lm_head doesn't have this caveat
-- it's a genuine jointly-computed Linear layer, so its transposed (input-feature-wise) reading is
exactly as valid as any other `D_out > D_in` layer's.

### Radial-dynamics metrics (`radial_helper.py`) -- always-on, independent of gram/norm config

A third tracked-metric family, alongside norm/spectrum and gram, living in its own module
(`radial_helper.py`). Unlike gram (row-wise Gram-matrix framework, needs `V_raw`, gated behind
`gram_level`), radial metrics ask a different, simpler question: how does a weight's *whole-tensor*
norm and direction evolve step to step. Every output is a scalar (never a row-wise vector), computed
from just `W_before`/`W_after` (the same pair already used as gram's `W_before`/`W_after`). The
original metrics use Frobenius reductions; the RMS-to-infinity and l1-to-RMS update radialities add
row/column L2 reductions. `calculate_radial_metrics` is called **unconditionally** whenever any
per-param logging fires, independent of `gram_level` and `norms_to_log`, at all four `step_*` call
sites (including `step_embedding`, ungated by `track_embed_gram`). Exact RMS-to-RMS radiality is
substantially more expensive because it requires an SVD of the current weight and an operator norm
of the update.

Four running accumulators (`raw_A2`, `angular_A1`, `angular_A2`, `R1`) track the parameter's
cumulative history and persist across checkpoint save/restore -- stored in
`self.state[p]["radial_state"]` (the same place `momentum_buffer` lives), included automatically in
`torch.optim.Optimizer`'s default `state_dict()`/`load_state_dict()`, no extra plumbing. 3-D (expert)
params get one accumulator set *per expert index* (each expert has its own `W_before`/`W_after`
pair), shape `(num_local_experts,)`.

Two things worth knowing if you're reading the formulas or extending this:
- `R2(t)` (from the "radial error" identity below) is exactly the same running sum as `raw_A2` --
  only one accumulator is kept, not two.
- `relative_step` is the same formula as `gram_helper.py`'s `U_relative_step_fro`. Not an accidental
  duplicate -- this one is unconditional, that one is gated behind `gram_level`.

The canonical `angle` (angle between consecutive weight directions `q_t`/`q_t+1`) is computed via
`atan2(a_t*tangent_fraction, r_t + a_t*radial_cosine)`, not `arccos(<q_t, q_t+1>)` -- `arccos`'s
derivative blows up near `cos=1`, so small angles (most training steps) lose precision in fp32;
`atan2` doesn't have that issue. Both formulas compute the exact same geometric quantity (verified via
the underlying 2D-trigonometry identity), so the `arccos` version is kept too, as `angle_from_cos`, purely
as an independent sanity check -- not fed into the accumulators.

**Relative-degeneracy floor on `radial_cosine`/`tangent_fraction`/`radial_ratio`:** these divide by
`a_t` or `a_t^2`, so once the actual update is numerically negligible relative to the weight's own
scale -- e.g. a near-zero-lr step at the tail of a decay schedule -- the division is noise divided by
noise, and that noise floor is genuinely different between DDP and FSDP (different collectives: DDP's
all-reduce vs FSDP's all-gather sum gradients in a different order, and floating-point summation isn't
associative). Observed in practice as those 3 metrics disagreeing hugely between a DDP run and an FSDP
run specifically at a near-zero-lr step, while `radial_first_order` (no division, `2*dot_wu`) only
differed slightly, and `angle`/`angle_from_cos` didn't disagree at all -- atan2's inputs stay
well-conditioned as `a_t -> 0` (numerator `-> 0`, denominator `-> r_t > 0`), so `angle` doesn't inherit
this instability the way a division by `a_t` does. Fixed by widening `valid_wu` from a bare `a_t > 0`
to `a_t > 1e-6 * r_t` (`radial_helper._REL_DEGENERACY_EPS`) -- a step below that relative threshold now
deterministically reports the same degenerate sentinel (`radial_cosine=0`, `tangent_fraction=1`,
`radial_ratio=0`) regardless of which parallelism strategy computed it, while a genuinely small-but-real
step (e.g. `relative_step ~ 1e-3`) is well above the threshold and reports real signal, unclamped.

The induced-operator-norm update radialities apply the same relative threshold in their own primal
geometry: they report `NaN` when `N(W_before) == 0` or
`N(U) <= 1e-6 * N(W_before)`. These are undefined or numerically degenerate cases for
`<D_{N*}(W_before), U/N(U)>`; `NaN` keeps them distinct from the valid radiality `0`, which means
that the selected outward normal and normalized update are orthogonal.

`alpha_fit`/`tau_fit` (fitting the angle-decay power law `theta_t = C*(t+tau)^-alpha` from the logged
`(t, angle)` history) is a deliberately deferred, offline/analysis-time follow-up, not optimizer
state: unlike the fixed set of reductions above, fitting this needs some bounded history of past
angles and periodic (not per-step) nonlinear refitting to stay cheap at scale, and
`tau` enters the fit nonlinearly, so there's no simple closed-form running update for it.

**Checkpoint compatibility:** resuming from a checkpoint saved *before* `radial_state` existed
crashes. TorchTitan's checkpoint load goes through `torch.distributed.checkpoint`'s default
`LoadPlanner`, which has `allow_partial_load=False`; since `radial_state` is always present in the
live optimizer's `state_dict()` skeleton (lazy-inited unconditionally in `_build_param_lists`), an old
checkpoint missing that key raises `RuntimeError: Missing key in checkpoint state_dict: ...` during
DCP's own planning phase -- before `DiSCO.load_state_dict` ever runs, so nothing on the optimizer side
can catch or work around it. A model-only load (optimizer state excluded entirely, e.g.
`--checkpoint.initial_load_in_hf` / `initial_load_model_only`) sidesteps this, at the cost of *all*
optimizer state (fresh momentum too, not just `radial_state`) -- there's no way to keep momentum while
dropping only `radial_state` short of relaxing `allow_partial_load` checkpoint-wide, which was
deliberately not done here since that would also silently paper over genuinely missing keys elsewhere.

**`DiSCO.load_state_dict` override** (`disco.py`) exists for two reasons, unrelated to the crash above:
1. `torch.optim.Optimizer.load_state_dict` overwrites every param_group key (besides `"params"`) with
   whatever the checkpoint saved, including config-derived keys (`eps`/`norm_factor`/`backend`/etc.,
   see `_CONFIG_ONLY_GROUP_KEYS`) that come from this run's config, not from training -- so resuming
   after a deliberate config change (e.g. switching `norm_factor`) would otherwise silently revert it.
   The override snapshots those keys before the base call and restores them after, warning on any
   mismatch.
2. `_momentum_buffer_by_param_id`/`_radial_state_by_param_id` (built once, normally at `__init__`) are
   refreshed afterward. This turned out to be defense-in-depth rather than a fix for an active bug: in
   TorchTitan's actual `dcp.load()` resume path, `OptimizersContainer.state_dict()` returns
   `self.state[p]`'s tensors *by reference* (`Optimizer.state_dict()` never clones), and DCP fills them
   in place before `load_state_dict` ever runs -- confirmed via an actual `dcp.save`/`dcp.load`
   round-trip, not just by reading source. So the caches stay valid on their own in that path; the
   refresh only matters for a load path that bypasses DCP's in-place fill (e.g. a direct/manual
   `load_state_dict()` call with a hand-built or detached state dict).

### Known limitation: `all_gather` sends to every rank, only one needs it

Every collective in this norm/gram/spectrum pipeline (`step_experts`'s single `all_gather_tensor`,
`step_ddp`/`step_fsdp`'s Phase-D `all_gather_tensor` via `_pack_segments`) gathers to **all** ranks,
but only the logging rank (`is_dp_rank_0` / FSDP-mesh rank 0) ever reads the result — every other
rank receives (and immediately discards) the full per-rank payload for nothing. Switching to a
root-only `dist.gather` would cut that wasted receive traffic without losing any fidelity (unlike
reducing the logged data itself, which isn't an option once you want the full vectors — see
`gram_vector_logging.py`'s docstring). Not done here: NCCL's support for plain `gather`-to-root is
limited/version-dependent, and this same inefficiency predates the gram-vector work (it already
applied to `track_spectrum_*`/scalar norms too) — fixing it is a genuine follow-up, not scoped into
either pass, and would need verifying against whatever backend/PyTorch version is actually in use
before landing.
