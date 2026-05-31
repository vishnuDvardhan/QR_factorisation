# Phase 2 — Classic Tiled QR with DAG-Priority HVX Context Scheduling

## Context / why
Phase 1 proved the thesis on *unblocked* QR: on Hexagon the scarce resource is the
**HVX context (~4 in 128B mode), not the thread**, and scheduling contexts by DAG
critical-path priority beats the barrier baseline (1.06–1.39× over v0, margin grows
with size; flat scaling past 4 contexts where v0 degrades). But unblocked QR is
memory-bound and the DAG is thin (one critical column per step), which caps both the
absolute performance and the scheduling headroom — a reviewer's first question is
"why not tiled?".

**Phase 2 answers that.** Classic tiled QR (PLASMA/Buttari-style — the exact "future
work" the souyajit paper names) gives:
1. **A rich 2-D DAG** with many independent tile-tasks ≫ 4 contexts — the regime where
   context-priority scheduling should *dominate*, not just edge out the baseline.
2. **High arithmetic intensity** (the apply kernels are GEMM-like) → HVX becomes
   compute-bound → much better absolute GFLOP/s and roofline story.
3. **The contribution that lifts this from short to full paper.**

Decisions locked with user: **classic tiled QR** (not WY-panel); **pure ISS dev now**
(HW expected in a few weeks — keep the door open but don't pay for portability yet).

## Background: the tiled QR algorithm (what we're implementing)
Matrix A is split into a grid of `b×b` tiles, `A[i][j]`, `0≤i<MT`, `0≤j<NT`
(MT = ceil(m/b), NT = ceil(n/b)). Four kernels, operating on tiles:

| Kernel | Role | Reads / writes | DAG analogue |
|---|---|---|---|
| **GEQRT(k)** | QR-factor the diagonal tile `A[k][k]` → in-place `R` + reflectors `V`, computes T-factor `T` | A[k][k] → A[k][k],V,T | `T(k,k)` pivot, critical path |
| **ORMQR(k,j)** | apply `(I−VTᵀVᵀ)` from GEQRT(k) to a trailing tile `A[k][j]`, j>k | A[k][j],V,T → A[k][j] | `T(k,j)` row-update |
| **TSQRT(i,k)** | couple-factor stacked `[A[k][k]; A[i][k]]` (i>k) → updates both, new reflectors `V'`,`T'` | A[k][k],A[i][k] → both,V',T' | below-diagonal annihilation |
| **TSMQR(i,k,j)** | apply TSQRT(i,k)'s reflectors to the stacked pair `[A[k][j]; A[i][j]]` | A[k][j],A[i][j],V',T' → both | coupled trailing update |

(Names follow LAPACK/PLASMA: GEQRT, ORMQR/LARFB, TSQRT, TSMQR/SSRFB.)

### Dependencies (the DAG)
For panel column k = 0..NT-1:
- `GEQRT(k)` depends on `TSMQR(k, k-1, k)` (its tile finalized by prev panel).
- `ORMQR(k,j)` (j>k) depends on `GEQRT(k)` **and** `TSMQR(k,k-1,j)`.
- `TSQRT(i,k)` (i>k) depends on `GEQRT(k)` **and** `TSQRT(i-1,k)` (the elimination
  chain down column k is sequential: i = k+1, k+2, …, MT-1).
- `TSMQR(i,k,j)` (i>k, j>k) depends on `TSQRT(i,k)`, `ORMQR(k,j)` (i=k+1) or
  `TSMQR(i-1,k,j)`, and `TSMQR(i,k-1,j)`.

**Critical path** = the diagonal chain `GEQRT(0) → TSQRT(*,0) → GEQRT(1) → …`. The
panel factorization (GEQRT + the TSQRT elimination chain) is the serial spine;
everything to its right (ORMQR/TSMQR on trailing tiles) is the parallel mass. This is
exactly the structure where a **bottom-level / critical-path priority** on the scarce
HVX contexts pays off: feed contexts to GEQRT/TSQRT first so the spine never stalls,
then spill the rest to the trailing TSMQR mass.

## Implementation: `v8_tiled_dag_hvx.c`

### Reuse from Phase 1 (do not reinvent)
- Persistent worker pool + per-task `go[]` signals + atomic `done_count`
  (`qr_pool_hh_unblocked.c` / `v7_dag_hvx_sched.c` pool).
- **Atomic HVX-budget allocator** (`hvx_claim`/`hvx_worker_units` + budgeted
  `qurt_hvx_lock`) — Phase 1's key fix; the deprecated `qurt_hvx_try_lock` is banned.
- `include/hvx_utils.h` HVX primitives (`hvx_dot_col_v`, `hvx_reduce_qf32`,
  `hvx_col_norm_sq`); extend with tile-level GEMM-ish helpers.
- Cycle metric `get_cycles()` (c15:14); correctness via `verify_R_against` +
  `subdiag_residual`; deterministic `fill_test_matrix`; `qr_cpu_householder` ground
  truth (already in v7).
- Makefile generic `vN` rules (`make v8_tiled_dag_hvx`, `make sim-v8_tiled_dag_hvx`).
- inparams reuse the existing `nb` field as the **tile size b**:
  `m n nt chunk sched nb` → b = nb (must divide evenly / pad; b multiple of 32 for HVX,
  start with b=32 or 64).

### New pieces (build + verify in this order — each step independently testable)
1. **Tile kernels, scalar first (correctness skeleton).**
   `geqrt(b)`, `ormqr(b)`, `tsqrt(b)`, `tsmqr(b)` as plain scalar tile ops on `b×b`
   blocks with a compact-WY `T` factor. Drive them with a **sequential** tiled QR loop
   (no pool). Verify R against `qr_cpu_householder` for several (m,n,b). *Gate: bit-for-
   bit-ish correct (R-err < 1e-3, sub-diag < 1e-5) before any parallelism or HVX.*
2. **HVX-ize the two apply kernels** (`ormqr`, `tsmqr` — the GEMM-like, high-AI, high-
   payoff ones) using `hvx_utils`. Keep GEQRT/TSQRT scalar at first (they're on the
   serial spine and small — b×b). Re-verify.
3. **Static task graph + ready-tracking.** Represent tasks as
   `{kind, k, i, j}` with an integer **priority = bottom-level** (precomputed:
   diagonal-chain tasks highest; trailing TSMQR lowest, decreasing with distance from
   the spine — closed-form, like Phase 1's ascending-column rule, no runtime graph
   walk). A per-task atomic `dep_count`; when it hits 0 the task is "ready". On task
   completion, decrement successors' `dep_count`.
4. **Priority dispatch with budgeted HVX.** Workers pull the **highest-priority ready
   task** (a small lock-free/atomic priority structure or bucketed ready-lists keyed by
   priority — keep it simple: a few priority buckets, not a full heap, to avoid the
   rebalancing overhead the paper warned about). Winner of the HVX budget runs the HVX
   kernel; over-budget workers run the scalar fallback. Reuse Phase 1's allocator.
5. **Scheduler ablation switches** (mirror Phase 1 + the paper's ladder):
   - `SCHED_BARRIER` — panel-by-panel barriers (PLASMA "static" / our v0 analogue).
   - `SCHED_ASYNC` — async dependency-driven, **no priority** (round-robin contexts).
   - `SCHED_ASYNC_PRIO` — async + **bottom-level context priority** ← the contribution.

### Concurrency contract (the risky part — design up front)
- One writer per tile per task; the `dep_count` edges above guarantee no two
  concurrent tasks touch the same tile. **TSQRT chain down a column is strictly
  sequential** (i increasing) — encode as a dependency, not a lock.
- Publish tile/T-factor writes with a memory barrier before decrementing successor
  `dep_count` (same pattern as Phase 1's `col_done` barrier).
- Reflector/T storage: each GEQRT/TSQRT writes its own `V`,`T` slab (indexed by
  (k) or (i,k)); no aliasing, no ring needed (unlike Phase 1, since tiles are distinct).
- Guard MT·NT ≤ MAX_TASKS; assert b | alignment; handle ragged last tile (pad to b or
  special-case partial tiles — start by requiring m,n multiples of b).

## Experiments (Phase 2)
- **P1 correctness:** all 3 sched modes, b∈{32,64}, (m,n) ∈ a few sizes → R-err +
  sub-diag OK. Gate before perf.
- **P2 head-to-head @ a fixed large size** (e.g. m=n=1024, b=64, nt=4): tiled barrier
  vs tiled async vs tiled async_prio vs **Phase-1 best (v7 unblocked)** vs v0. Show
  tiling + priority wins, and by how much.
- **P3 tile-size sweep** b∈{32,64,128} — find the AI/parallelism sweet spot.
- **P4 nt sweep** {1,2,4,6,8} — confirm the "flat past 4 contexts" scaling story holds
  and is *stronger* than unblocked (more independent tiles to absorb scalar fallback).
- **P5 priority ablation** async vs async_prio across sizes — quantify the priority win
  in the many-tiles regime (expected larger than unblocked's, *inverting* the paper).
- **P6 (cheap, do alongside):** parse `build/hexagon/pmu_stats.txt` for HVX utilization
  / stall counters → mechanism story + simulator credibility.
- **Roofline:** convert cycles → GFLOP/s (QR flops ≈ 2mn²−(2/3)n³), plot vs HVX peak.

## HW-readiness (cheap hooks now, full validation when hardware lands)
Pure-ISS dev, but two no-cost habits so the device port is a config flip later:
- Keep timing in one `get_cycles()`-based harness block; don't scatter ISS-isms.
- Keep the result-print format identical to Phase 1 so device runs drop into the same
  parser / notebook. When HW arrives: add a fidelity (ISS-vs-silicon) section — one
  validation point de-risks the whole paper.

## Definition of done (Phase 2)
- `v8_tiled_dag_hvx.c` builds clean; 3 sched modes correct across the P1 grid.
- P2–P5 measured; tiled async_prio is the best variant and beats Phase-1 v7.
- Numbers + a tiled-DAG figure recorded in `hvx_householder_research_plan.md`.
- Then decide short vs full paper based on the magnitude of the tiled win.

## Files
- **New:** `v8_tiled_dag_hvx.c`; extend `include/hvx_utils.h` with tile GEMM helpers.
- **Reuse:** pool/allocator from `v7_dag_hvx_sched.c`; verify/ground-truth from same.
- **Update:** `Makefile` VARIANTS += v8; `hvx_householder_research_plan.md` results.
- **Untouched:** all Phase-1 baselines and variants (needed for comparison).
