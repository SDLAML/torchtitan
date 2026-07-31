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

### Known limitation: gram tracking is disabled for `step_embedding` (large-vocab OOM)

`gram_helper._gram(X) = X @ X.T` forms an `m x m` matrix, where `m` is the row-count of the
prepped/transposed tensor -- fine for typical attention/FFN matrices (`m` ~ hidden_dim, thousands),
but `step_embedding`'s `embed_params` also includes the `output`/lm_head weight
(`[vocab_size, hidden_dim]`). `need_T = CONST_NAME_OF_EMBEDDING in p_name` only transposes for
params literally named `"tok_embeddings"`, not `"output"`, so the lm_head weight's `m` stays at
`vocab_size` instead of being reduced to `hidden_dim` -- at a ~200k vocab this is a
`[200_000, 200_000]` fp32 matrix (~160GB), an immediate CUDA OOM (hit in practice, not
hypothetical).

**Current state**: the `calculate_gram_metrics` call in `step_embedding` is commented out
(`gram_metrics = {}` unconditionally) until this is fixed properly. `step_ddp`/`step_fsdp`/
`step_experts` are unaffected (never touch vocab-scale dimensions) and keep tracking gram normally.

Two real fixes, not done yet:
1. A size guard in `gram_helper.calculate_gram_metrics` (e.g. `if W_before.shape[0] >
   _MAX_GRAM_M: return {}`, same "cheap no-op for ill-defined input" precedent as the existing
   `m < 2` guard) — general, protects every call site against any future oversized-`m` case, not
   just this one. Would let `tok_embeddings`'s gram tracking keep working (its `m` is already
   correctly reduced to `hidden_dim` via `need_T`) while only skipping the lm_head weight.
2. Fix `need_T` to also transpose for `"output"` (if that's semantically correct — needs checking
   against how `norm_factor`'s embed/unembed row-wise treatments and `abstract_disco.py`'s
   `fused_unembed_*` functions expect the axes oriented; this is unrelated pre-existing logic, not
   something introduced by the gram work).

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
