# PLAN — Vectorize the reflector & rescue `async_prio_cc` (do later)

Status: **DONE 2026-06-03 — the rescue HYPOTHESIS WAS REFUTED, but the HVX
reflector turned into a separate, useful win on SQUARE matrices.** See the
"## RESULTS" section at the bottom. The plan body below is the original
(pre-experiment) reasoning, kept for the record.

---

Status (original): **not started** — captured 2026-06-02, to implement later.

## The insight (why this matters)

The reflector computation in v7 is **entirely scalar** — both O(m) norm reductions
(`nxsq = Σ A[i,k]²`, `nvsq = Σ v[i]²`) and the column copy run as scalar loops on the
main thread (`compute_reflector`, [v7_dag_hvx_sched.c:172-188]). The HVX norm helper
`hvx_col_norm_sq` exists in `include/hvx_utils.h:59` and is `#include`d but **never
called**.

This is the bug behind `SCHED_ASYNC_PRIO_CC` underperforming (it was 1.3–2.6× slower
than `async_prio`). Mechanism, confirmed by reading `qr_dag_async_prio_cc`
([v7_dag_hvx_sched.c:480-532]):

- Main locks ONE HVX context up front ([:497]) and holds it for the whole k-loop.
- Per step main does: (1) `apply_col_hvx` on the critical column k+1 — **uses** the
  context; (2) dispatch workers; (3) `compute_reflector(k+1)` — **SCALAR**; (4)
  `pool_wait_done()` — idle.
- Main only USES HVX in phase 1 (one column). Phases 3–4 (the bulk of main's time,
  dominated by the slow scalar reflector) the held context is **locked but idle**.
- Cost of the design: workers drop 4→3 contexts ([:491]) to reserve a context for main
  that then sits idle through the scalar reflector. On square (trailing-dominated,
  workers are the bottleneck) that's a straight ~25% loss → the measured slowdown.

So: **main reserves a context to accelerate the critical path, then computes that
critical path in scalar, leaving the context idle.** Vectorizing the reflector fills
main's idle context with useful HVX work — closing the loop the design was built around.

## Why it specifically helps tall-skinny (the model-backed claim)

- **Tall-skinny (large m, small n):** trailing work is small (few columns) → workers
  are NOT the bottleneck, the scalar reflector chain (∝ m, on the critical path) IS.
  Stealing one worker context costs ~nothing; vectorizing the reflector on main's held
  context demolishes the actual bottleneck. → `async_prio_cc` + HVX reflector should
  WIN BIG here.
- **Square (m≈n):** trailing dominates, workers are the bottleneck; giving up a context
  still hurts and a fast reflector helps less (reflector was never the bottleneck).
  `async_prio_cc` may still trail `async_prio`, but by much less.

Prediction to test: **`async_prio_cc` + vectorized reflector overtakes `async_prio` on
tall-skinny at large m, with the crossover falling out as a function of m/n.** This
converts a negative result ("async_prio_cc underperforms") into a sharp, defensible one
("co-holding requires a vectorized critical path; with it, co-holding wins for
m/n ≥ threshold").

## Implementation steps

1. **HVX reflector.** Add `compute_reflector_hvx` (or a flag on `compute_reflector`)
   that computes both norm reductions with `hvx_col_norm_sq` / `hvx_dot_col_v`-style
   HVX tree reductions instead of scalar loops. Keep the scalar version for the
   ablation. Note both O(m) passes: `‖A[:,k]‖²` and `‖v‖²`. The column copy can stay
   scalar or be vectorized too.
2. **Wire into `async_prio_cc` on main's already-held context.** Main already holds one
   HVX context ([:497]); no per-step lock needed — just call the HVX reflector there.
   This is the natural home: in barrier/async_prio main does NOT hold a context, so
   vectorizing there would need a per-step lock/unlock (defer that).
3. **Keep other schedulers on the scalar reflector** for a controlled comparison
   (isolates the reflector-vectorization effect).
4. **Re-measure tall-skinny sweep** (Exp 2: n=64, m ∈ {256,512,1024,2048,4096}, nt=4)
   with all four schedulers: barrier / async / async_prio / async_prio_cc. Find the
   m/n crossover where async_prio_cc overtakes async_prio.
5. **Optional follow-on:** if the HVX reflector wins broadly, also offer it to
   barrier/async_prio via a brief main-side lock, and re-check square.

## Correctness / gotchas
- `hvx_col_norm_sq` requires the start index to be lane-aligned (multiple of 32); pivot
  columns satisfy this since sub-pivot entries are zeroed. Verify the partial first
  vector contributes correctly for the `‖v‖²` pass (v[0..k-1] are zeroed in
  `compute_reflector`, so a full [0,m) HVX reduction is valid).
- Main using HVX concurrently with workers is already proven safe — the earlier
  "async_prio_cc deadlock" was a timeout artifact, not a real deadlock.
- Gate: `max|R|-err < 1e-3`, sub-diag `< 1e-5` vs scalar reference, at nt=1 and nt=4,
  128² and 256², before trusting timings.

## Files
- Edit: `v7_dag_hvx_sched.c` — add HVX reflector, wire into `qr_dag_async_prio_cc`.
- Reuse: `include/hvx_utils.h` `hvx_col_norm_sq` (already present, currently unused).
- Measure: `qr_experiments_v2.ipynb` Exp 2 (tall-skinny sweep).
- Record: `hvx_householder_research_plan.md`.

## Paper angle
Highest-leverage fix regardless of venue: a paper *about* accelerating the QR critical
path must not compute that path in scalar. Fixing it (a) improves the headline
tall-skinny numbers, (b) rehabilitates async_prio_cc into a model-backed win, and
(c) closes the most obvious reviewer hole.

---

## RESULTS (2026-06-03)

Implemented `compute_reflector_hvx` (HVX-vectorised norm reductions; the O(m)
column COPY is still scalar). Addition-only split for `||A[k:m,k]||^2` (scalar
head over the <=31-element unaligned prefix + HVX over the aligned bulk) to avoid
cancellation. Correctness verified: nt=1 (no deadlock) and nt=4, 128^2/256^2,
R-err <= 5.6e-4.

### Finding 1 — the cc rescue hypothesis was WRONG
`async_prio_cc` (sched=3) with the HVX reflector is still SLOWER than
`async_prio`, on both shapes (nt=4):

| matrix   | async_prio | async_prio_cc (HVX refl) | cc result   |
|----------|-----------:|-------------------------:|-------------|
| 256x256  |    719,967 |                  888,881 | 1.23x slower|
| 1024x64  |    665,169 |                1,403,729 | 2.11x slower|

Why the premise failed: at n=64 the trailing update dominates the reflector by
~n/2 = 32x, so WORKERS are the bottleneck, not the reflector chain. cc gives up
a worker context (4->3) AND adds a serial O(m) critical column on main; both
costs scale with m, so tall is hit hardest. The reflector would only dominate at
n <= ~8. Also: main holds its context for the WHOLE loop and only releases at the
very end — and in Householder QR the critical path is coextensive with the whole
factorisation, so the "release once critical path done" returns the context when
there is no work left. cc is a dead end for practical aspect ratios.

### Finding 2 — the HVX reflector is a real win on SQUARE (new variant)
Computing the reflector on HVX SERIALLY inside the critical-column lock window
(workers keep all 4 contexts) helps when the reflector is EXPOSED on the critical
path (square / small m, where the trailing range shrinks and late steps are
worker-light) and hurts when workers fully hide it (tall / large m):

| matrix   | async_prio (scalar) | async_prio_hvx (HVX) | change |
|----------|--------------------:|---------------------:|--------|
| 256x256  |             724,253 |              662,733 | -8.5%  |
| 1024x64  |             665,169 |              732,290 | +10%   |

Shipped as a SEPARATE scheduler mode so the baseline is preserved:
- `sched=0` `async_prio`      — scalar, overlapped reflector (unchanged baseline)
- `sched=5` `async_prio_hvx`  — HVX, serial-in-window reflector (new)
`qr_dag_async` now takes an `hvx_reflector` flag; all other modes untouched.
Added `v7_async_prio_hvx` to the experiments notebook.

### Open follow-ups
- The +10% tall regression is entirely the un-vectorised O(m) COPY in the serial
  window. Vectorising that copy (HVX load/store, ~32x fewer instructions) may
  shrink/erase the regression -> a possible win on BOTH shapes.
- Map the m/n crossover where async_prio_hvx overtakes async_prio (full sweep).
