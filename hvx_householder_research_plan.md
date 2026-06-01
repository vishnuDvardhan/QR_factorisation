# HVX Householder QR Acceleration on Hexagon v69 — Research & Implementation Plan

## Context

Starting point: `pool_hh_unblocked` — a persistent thread-pool Householder QR on
Hexagon v69 with HVX trailing-column updates. This is the current best performer.
Goal: systematically eliminate every known bottleneck, characterize each fix, and
produce a publishable HiPC 2026 short paper.

---

## Progress Log — 2026-05-31 (later) — Phase 2 step 1: scalar tiled QR (v8)

Started Phase 2 (classic tiled QR, see `phase2_tiled_dag_plan.md`). New file
`v8_tiled_dag_hvx.c`. **Step 1 = scalar correctness skeleton** (no HVX/threads yet):
four tile kernels GEQRT / ORMQR / TSQRT / TSMQR built on two generic Householder
primitives (`house_qr_panel`, `house_apply_panel`), driven by a sequential tiled loop.
TSQRT/TSMQR use a dense 2B×B stacked scratch (correct; structure-exploiting form is a
later optimization).

**Correctness gate PASSED — 10/10 configs** vs `qr_cpu_householder` ground truth
(`v8_verify.txt`, sweep `scripts_v8_verify.sh`). All sub-diagonal residuals **exactly
0.0** (kernels write exact zeros); max |R|-err 9.2e-4 at the largest size (1024×128),
within the 1e-3 gate:
| m×n (B) | R-err | sub-diag |
|---|---|---|
| 128×128 (32/64) | 6.7e-5 | 0 |
| 256×128 (32/64) | 1.2e-4 / 2.8e-4 | 0 |
| 256×256 (64) | 2.8e-4 | 0 |
| 512×64 (32) | 3.0e-4 | 0 |
| 512×256 (64) | 2.6e-4 | 0 |
| 384×96 (32) | 2.3e-4 | 0 |
| 640×64 (32) | 7.4e-4 | 0 |
| 1024×128 (64) | 9.2e-4 | 0 |

Note: the scalar tiled skeleton is (expectedly) *slower* than the unblocked scalar ref
in raw cycles — it does dense flops in the stacked TSQRT/TSMQR and isn't vectorized.
That is fine: this step only locks correctness. Speed comes from steps 2–4 (HVX apply
kernels → task graph → priority scheduling).

### Steps 2–4 done (v8 parallel scheduler — committed 4321906, 8b92bc3)
- **Step 2 (HVX apply):** `house_apply_panel_hvx` vectorizes ORMQR/TSMQR over the
  row dim (exploits `V[r<c]=0` to run the full tile-aligned range). ~1.6–1.9× over
  the scalar skeleton; still correct (sub-diag exactly 0).
- **Step 3+4 (DAG + priority HVX scheduler):** persistent worker pool over a task
  DAG built by sequential-trace + per-tile last-writer (race-free by construction);
  bottom-level critical-path priority; budgeted HVX-context allocator (first
  min(nt,#units) workers hold a context → HVX apply, rest → scalar). Modes
  async_prio / async / barrier / seq. **8/8 configs correct** (sub-diag 0):

| mode | nt | 128×128 B32 cyc | vs seq |
|---|---|---|---|
| seq | 1 | 2,377,823 | 1.00× |
| async_prio | 1 | 2,426,692 | 0.98× |
| async_prio | 2 | 1,418,828 | 1.68× |
| barrier | 4 | 1,420,562 | 1.67× |
| async | 4 | 1,422,110 | 1.67× |
| async_prio | 4 | 1,419,316 | 1.68× |

**Honest read:** at 128×128 the tiled DAG is too thin (16 tiles, ~28 tasks) to feed
4 workers, and the panel factor (GEQRT/TSQRT) is still *scalar* → the serial
bottleneck. So nt=4 saturates at ~1.68× and the three modes tie — priority can't
differentiate yet. The scheduling story should separate at larger sizes (independent
tiles ≫ 4 contexts), which is step 5.

### Hard-won lessons (worth a paper footnote)
- The bring-up "deadlocks" were **host overload** (stray 100-min ISS processes
  starving multi-second runs past the timeout), NOT scheduler bugs. Always kill
  stray `hexagon-sim` and check load before trusting an ISS timeout.
- Real bugs found+fixed: DAG predecessor-count corruption (macro var-shadow);
  `qurt_sem_init` leaves count unspecified → use `qurt_sem_init_val(.,0)`;
  semaphore teardown race after join.
- `fflush(stdout)` after each printf is mandatory on this ISS to see progress
  before a timeout-kill.

### Step 5 — v7 vs v8 head-to-head (DONE; `compare_v7_v8.{txt,csv}`, `scripts_compare_v7_v8.sh`)
nt=4, B=32, cycles (c15:14), all correct:

| variant / mode | 128×128 | 256×256 |
|---|---|---|
| colmajor_mt (spawn) | 1,168,982 | 2,260,307 |
| v0_pool (barrier) | 341,377 | 938,095 |
| v1_hvx_red (P1) | 301,989 | 792,991 |
| v3_persist (P4) | 338,018 | 931,954 |
| **v7 unblk async_prio** | 289,327 | **724,195** |
| **v7 unblk barrier** | **287,766** | 735,121 |
| v8 tiled seq | 1,406,270 | 6,996,080 |
| v8 tiled barrier | 622,405 | 2,225,774 |
| v8 tiled async | 620,131 | 2,623,194 |
| v8 tiled async_prio | 622,404 | 2,235,011 |

**Headline finding (robust, defensible, NOT a bug):** the **unblocked v7 beats the
tiled v8**, and the gap *widens* with size — 2.2× at 128, **3.1× at 256**. This
inverts the classic CPU/HPC "tiling wins at scale" intuition, for two
platform-specific reasons:
1. **Structure-unaware coupled kernels.** Our TSQRT/TSMQR factor/apply the stacked
   2B×B pair densely, so the tiled algorithm does ~9× more arithmetic than unblocked
   (v8 seq = 6.996M vs v7 ≈ 0.73M at 256). HVX+scheduling claw most of it back (to
   ~2.23M) but cannot beat v7's lean unblocked kernel.
2. **Only 4 HVX contexts.** Tiling's extra task-level concurrency can't be cashed in
   beyond 4 vector units, so the tiling overhead isn't repaid by parallel speedup.

**v8 scheduler modes tie** (barrier 2.226M ≈ async_prio 2.235M ≈ async 2.623M at 256;
same at 128, 256-B64, 512-B32) — with 4 contexts, scheduling *policy* does not move
the tiled DAG. Priority differentiation needs either ≫4 vector units or
structure-exploiting kernels that lower the arithmetic floor.

**Paper implication:** lead with the **v7 result** (HVX-context-aware DAG scheduling
for unblocked QR; vector context is the scarce resource; 1.3–1.4× over pool, graceful
past 4 threads). Frame the **tiled v8 as a boundary/negative result**: naive tiling
does not pay off on a 4-context DSP because structure-unaware coupled kernels inflate
arithmetic faster than the limited vector parallelism can absorb. A structure-exploiting
TSQRT/TSMQR (triangular V, internal-blocking) is the path to make tiling competitive —
future work, and the strongest lever if real hardware exposes more contexts.

### Remaining (optional polish)
- PMU counters + roofline (GFLOP/s vs HVX peak) for both v7 and v8.
- Structure-exploiting TSQRT/TSMQR (would change the v8 verdict).
- Real-hardware validation point when the board arrives.

---

## Progress Log — 2026-05-31 — DAG-priority HVX scheduling (v7)

**Goal of this phase:** bring the DAG task-scheduling ideas of Chatterjee et al.
(`souyajit-paper.pdf`, arXiv 2506.09463) to our HVX Householder QR, and schedule the
scarce HVX *vector contexts* (4 in 128B mode) to tasks by DAG priority. New file:
`v7_dag_hvx_sched.c`. Full plan: `~/.claude/plans/delegated-fluttering-chipmunk.md`.

### What the paper does / doesn't give us
It is a **multicore CPU** paper (Intel TBB, 26–104 threads) — **no HVX / DSP / vector
unit notion**. Reusable parts: (1) the task DAG `T(k,k)`=reflector, `T(k,j>k)`=trailing
update, edges `T(k,j)←T(k,k)` and `T(k,j)←T(k-1,j)`; (2) the scheduling ladder
barrier → async dual-queue → priority. Our novel bit is making the *HVX context* the
scheduled resource. **Critical path = update of column k+1 each step** (reflector k+1
waits only on it), so critical-path priority = "do column k+1 first, compute reflector
k+1 while workers finish k+2…n-1." Priority is closed-form (ascending column index) —
no runtime graph analysis.

### Bugs found & fixed (both important, worth writing up)
1. **`qurt_hvx_try_lock` is deprecated on v65+ and does NOT grant a context** → every
   worker silently fell to the scalar path → **~5× slowdown** (3.2M cyc) in *both*
   scheduler modes. Fixed with an **atomic HVX-budget allocator**: a worker claims a
   slot with one `qurt_atomic_inc_return`; only if `slot ≤ worker_units` does it call
   the (blocking) `qurt_hvx_lock` — so the lock is only ever taken when a context is
   provably free. Clean scalar fallback for workers beyond the budget. *This is the
   "allocate the 4 HVX units by priority" mechanism.*
2. **Main thread holding a context for its lifetime** starved a worker exactly at
   `nt == #contexts` (nt=4: 933,644 cyc, *worse* than v0). Fixed: main locks a context
   only *briefly* for the single critical column k+1 (between the previous join and the
   next dispatch, when no worker is active), then releases it so workers get all 4.

### Results (m=640, n=64, chunk=4; cycles via c15:14; all correct)
All variants verified: max|R|-err ≈ 2.2e-4, sub-diagonal residual ≈ 3.5e-7.

**E1 head-to-head @ nt=4:**
| Variant | Cycles | vs v0_pool | vs scalar(11.77M) |
|---|---|---|---|
| colmajor_mt (spawn-per-step) | 1,635,916 | 2.59× slower | 7.2× |
| v0_pool (barrier) | 630,569 | 1.00× | 18.7× |
| v7 barrier (=v0 algo + HVX tree-reduce) | 567,798 | 0.90× | 20.7× |
| **v7 async_prio (final)** | **473,042** | **0.75× (1.33× speedup)** | **24.9×** |

**E3 nt sweep — async_prio vs v0_pool (lower = better):**
| nt | v0_pool | v7 async_prio (1st: main holds ctx) | v7 async_prio (FINAL: brief lock) |
|---|---|---|---|
| 1 | 1,244,001 | 859,956 | 884,149 |
| 2 | 817,203 | 564,410 | 588,172 |
| 3 | — | 486,553 | 500,897 |
| 4 | 630,087 | 933,644 ← collapse | **473,042** ← best |
| 6 | 685,033 | 598,943 | 497,721 |
| 8 | 719,837 | 594,832 | 522,616 |

### Conclusions
- **HVX tree-reduction (P1) is a free ~10%** — `v7 barrier` vs `v0` is the same
  algorithm except the dot-product reduction is in-register (no store-to-temp[32]).
- **The scarce resource on Hexagon is the HVX context (≈4), not the thread.** v0_pool
  *degrades* past nt=4 (630K→720K @ nt=8) as surplus threads thrash the lock; v7
  async_prio stays flat (473K–523K) because over-budget workers do useful scalar work
  instead of blocking. This is the headline DSP-specific result.
- **Critical-path pivot overlap + correct context budgeting wins at every nt.** Final
  v7 best = **473,042 @ nt=4 = 1.33× over v0_pool, 3.46× over colmajor_mt, 24.9× over
  scalar.** Inverts the paper's CPU finding that priority barely helped — because
  contexts are far scarcer than the paper's 26–104 threads, allocation matters more.
- **Busy-waiting is catastrophic here** (first cut spun main on `col_done[k+1]` → 5×):
  on an in-order SMT DSP a spinning thread steals an HW slot + memory BW from HVX
  workers. The fix is to give the main thread *real work* (the critical column).

### Reproduce
`scripts_bench.sh` (E1+E3, all variants) and `scripts_resweep.sh` (v7 nt sweep) →
`results_bench.txt`, `resweep.txt`. Per-run config in `build/inparams.txt`
(`m n nt chunk sched nb`; sched 0=async_prio, 1=barrier).

### E4 size sweep — DONE (nt=4, chunk=4; 24/24 configs correct; see `e4_results.txt`)
v7_async_prio speedup over v0_pool (cycles); the margin **grows with size**:
| | m=128 | m=256 | m=512 | m=1024 |
|---|---|---|---|---|
| n=32 | 1.06× | 1.34× | 1.39× | 1.30× |
| n=64 | 1.12× | 1.22× | 1.31× | 1.39× |
Largest matrix m=1024 n=64: v0 920,205 → v7_async_prio **661,979 (1.39×)**; barrier
839,239. Confirms the 640×64 result is not a fluke and the advantage scales.

### Next
- Phase 2: tiled DAG (`v8_tiled_dag_hvx.c`) — GEQRT/TSQRT/LARFB 2-D tile DAG, same
  priority HVX allocator. *Strengthens paper but not strictly required.*
- (Optional, cheap) parse `pmu_stats.txt` alongside cycles for a mechanism story.

---

## Progress Log — 2026-05-30

### Done today
- **Environment verified.** `hexagon-clang 19.0.07` + `hexagon-sim` from the SDK at
  `/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.5.0.0` (`HEXAGON_SDK_ROOT` is set in
  the shell, so the Makefile default path is overridden). Full
  build→simulate→cycle-count pipeline confirmed working.
- **Adapted to the existing flat repo** (the idealized `hvx_hh_qr/` tree does not
  exist). v0 baseline = the existing `qr_pool_hh_unblocked.c` (left untouched).
  - Added `include/hvx_utils.h`: `hvx_reduce_qf32`, `hvx_col_norm_sq`,
    `hvx_vec_norm_sq`, `hvx_dot_col_v`. (`Q6_V_vror_VR` rotates by *bytes*, so the
    5 reduction stages use 64/32/16/8/4 — verified against the SDK protos header.)
  - Makefile: added `-Iinclude` and generic variant rules — `make vN`,
    `make sim-vN`, `make variants` (VARIANTS = v1..v6).
- **v3_persistent_hvx.c (Fix P4)** — implemented + built + run. Worker now acquires
  HVX once on first job and holds it for its lifetime.
- **v1_hvx_reduction.c (Fix P1)** — implemented + builds clean. **Not yet run.**

### Measurements so far (m=640, n=64, nt=4, chunk=4; metric = `get_cycles` c15:14)
| Variant | Cycles | max\|R\|-err | vs v0 |
|---------|--------|-----------|-------|
| scalar_ref (CPU) | 11,768,356 | — | — |
| v0 pool_hh_unblocked | 630,569 | 2.060e-04 | 1.00× (18.7× vs scalar) |
| v3 persistent_hvx (P4) | 622,550 | 2.060e-04 | **1.013× (~1.3% faster)** |

### Key finding (important — revise paper claims)
**P4 (persistent HVX) saved only ~1.3%, not the predicted 20–30%.** The plan's
"5K cyc/lock × n × nt = 1.28M cycles wasted" estimate is self-inconsistent: 1.28M
exceeds the *entire* 630K-cycle baseline runtime. On this QuRT/ISS the per-dispatch
`qurt_hvx_lock`/`unlock` pair is far cheaper than assumed. The "Expected Saving"
table in this doc needs to be re-derived from measurement, not a priori estimates.

### Rest for today / next up
1. Run **v1_hvx_reduction** at the same config; record cycles + err.
2. Run **v3 at nt=8** (the plan's second P4 data point) to see if locking matters
   more under context contention.
3. Run **Experiment A** microbenchmark (isolate horizontal-reduction cost) — this is
   what actually justifies P1, and we need the standalone number.
4. Then proceed to v2 (P2 pivot norm), v4 (P5 adaptive chunk), v5 (P3 DAG pipeline),
   v6 (combined) per the implementation order below.

---

Platform: Hexagon v69 DSP, Hexagon ISS (--timing), 8 QuRT threads, 4 HVX contexts,
128-byte HVX registers (32 float lanes), VTCM available.
Metric: Pcycles via `c15:14`.

---

## Repository Layout

```
hvx_hh_qr/
├── inparams.txt                  # m n nt chunk use_static nb
├── src/
│   ├── common.h                  # Matrix type, MAT macro, get_cycles, fill_test_matrix
│   ├── scalar_ref.c              # CPU scalar Householder — ground truth
│   ├── v0_pool_baseline.c        # Original pool_hh_unblocked (DO NOT MODIFY)
│   ├── v1_hvx_reduction.c        # Fix P1: tree-reduce dot product in HVX
│   ├── v2_hvx_pivot.c            # Fix P2: vectorize pivot norm on main thread
│   ├── v3_persistent_hvx.c       # Fix P4: hold HVX context across dispatches
│   ├── v4_adaptive_chunk.c       # Fix P5: adaptive chunk size per iteration
│   ├── v5_dag_pipeline.c         # Fix P3: DAG-pipelined pivot k+1 overlap
│   └── v6_combined.c             # All fixes combined — final submission variant
├── include/
│   └── hvx_utils.h               # hvx_reduce_qf32, hvx_dot_col, etc.
├── scripts/
│   ├── run_sweep.sh              # Sweeps m, n, nt, chunk; writes results/
│   └── parse_results.py          # Reads results/, produces tables + plots
└── results/                      # Raw cycle counts per variant per config
```

---

## Shared Utilities to Build First (`include/hvx_utils.h`)

Before touching any variant, implement these helpers once and reuse everywhere.

### 1. `hvx_reduce_qf32` — tree reduction of 32-lane qf32 vector to scalar

```c
// Replace the store-to-array + 32-iteration scalar loop that appears in
// pool_do_col_hh. No memory traffic. 5 rotate-and-add pairs = 10 instructions.
static inline float hvx_reduce_qf32(HVX_Vector vqsum) {
    HVX_Vector v = Q6_Vsf_equals_Vqf32(vqsum);   // qf32 → sf

    // 5 stages: each halves the number of independent partial sums
    HVX_Vector vhi;
    vhi = Q6_V_vror_VR(v, 64);   v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));
    vhi = Q6_V_vror_VR(v, 32);   v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));
    vhi = Q6_V_vror_VR(v, 16);   v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));
    vhi = Q6_V_vror_VR(v,  8);   v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));
    vhi = Q6_V_vror_VR(v,  4);   v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));

    float result;
    memcpy(&result, &v, sizeof(float));  // lane 0 holds the full sum
    return result;
}
```

### 2. `hvx_col_norm_sq` — ‖A[k:m, k]‖² using HVX

```c
// Used in pivot step. m must be multiple of 32. k is start row.
// Caller must hold HVX lock.
static inline float hvx_col_norm_sq(const float* col, int m, int k_start) {
    HVX_Vector vacc = Q6_V_vzero();
    for (int i = k_start; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector vc = *(const HVX_VectorU*)(col + i);
        vacc = Q6_Vqf32_vadd_Vqf32Vqf32(vacc, Q6_Vqf32_vmpy_VsfVsf(vc, vc));
    }
    return hvx_reduce_qf32(vacc);
}
```

### 3. `hvx_vec_norm_sq` — ‖v[k:m]‖² (same pattern, separate name for clarity)

```c
static inline float hvx_vec_norm_sq(const float* v, int m, int k_start) {
    return hvx_col_norm_sq(v, m, k_start);  // identical operation
}
```

### 4. `hvx_dot_col_v` — dot product of A[:,j] with v, fully in HVX

```c
// Replaces the vSum accumulation + scalar reduction in pool_do_col_hh.
static inline float hvx_dot_col_v(const float* col_j, const float* v, int m) {
    HVX_Vector vacc = Q6_V_vzero();
    for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector va = *(const HVX_VectorU*)(col_j + i);
        HVX_Vector vv = *(const HVX_VectorU*)(v + i);
        vacc = Q6_Vqf32_vadd_Vqf32Vqf32(vacc, Q6_Vqf32_vmpy_VsfVsf(va, vv));
    }
    return hvx_reduce_qf32(vacc);
}
```

---

## Fix P4 — Hold HVX Across Dispatches (`v3_persistent_hvx.c`)

**Do this first. Highest return, lowest effort.**

**Problem:** Current code calls `qurt_hvx_lock / qurt_hvx_unlock` on every dispatch
inside `pool_worker`. For n columns, each worker does n lock/unlock pairs.
Cost: ~5K cycles per lock × n × nt threads.
For m=640, n=64, nt=4: 4 × 64 × 5K = **1.28M cycles wasted on locking alone**.

**Fix:** Each worker acquires HVX once on first dispatch and holds it until shutdown.

```c
// Modified pool_worker — acquire HVX once, hold for entire QR run
static void pool_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    qurt_signal_t* my_go = &g_pool.go[tid];
    int hvx_held = 0;

    while (1) {
        qurt_signal_wait(my_go, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
        qurt_signal_clear(my_go, 1u);

        if (g_pool.kind == POOL_JOB_SHUTDOWN) break;

        // Acquire HVX once on first real job; hold for lifetime
        if (!hvx_held && g_pool.kind != POOL_JOB_NONE) {
            hvx_held = (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0);
        }

        if (hvx_held) {
            switch (g_pool.kind) {
                case POOL_JOB_COL_HH: pool_do_col_hh(); break;
                default: break;
            }
        }

        unsigned int remaining = qurt_atomic_add_return(&g_pool.done_count, -1u);
        if (remaining == 0)
            qurt_signal_set(&g_pool.done_master, 1u);
    }

    if (hvx_held) qurt_hvx_unlock();
    qurt_thread_exit(0);
}
```

**Note on the 4-context limit:** With nt=8 workers all trying to hold HVX, only 4 will
succeed. The remaining 4 get `hvx_held=0` and will do scalar fallback for every
dispatch. This is the same behavior as before but without the per-dispatch lock
overhead. If nt <= 4, all workers hold HVX — optimal.

**Measurement target:** Run v0_pool_baseline vs v3_persistent_hvx for:
- m=640, n=64, nt=4, chunk=4
- m=640, n=64, nt=8, chunk=4
Record total cycles and locking overhead separately if possible.

---

## Fix P1 — HVX Tree Reduction for Dot Product (`v1_hvx_reduction.c`)

**Problem:** `pool_do_col_hh` accumulates vSum in qf32 (correct) but then does:
```c
float temp[32]; *(HVX_VectorU*)temp = Q6_Vsf_equals_Vqf32(vSum);
float dot = 0.f;
for (int lane = 0; lane < 32; lane++) dot += temp[lane];  // scalar, 32 ops + memory
```
This stores 128 bytes to the stack and loops 32 times. Cost: ~32 scalar adds +
1 store + 1 load = ~40 extra cycles per column j.
Total wasted: 40 × (n × (n-1) / 2) columns processed.
For n=64: 40 × 2016 = **80,640 cycles per thread**.

**Fix:** Replace with `hvx_dot_col_v` from hvx_utils.h. Rewrite `pool_do_col_hh`:

```c
static void pool_do_col_hh(void) {
    int m     = g_pool.hh_A->m;
    int chunk = g_pool.hh_chunk_size;

    while (1) {
        unsigned int ci = qurt_atomic_inc_return(&g_pool.hh_counter) - 1;
        int col_base = g_pool.hh_col_start + (int)ci * chunk;
        if (col_base >= g_pool.hh_col_end) break;
        int col_end_c = col_base + chunk;
        if (col_end_c > g_pool.hh_col_end) col_end_c = g_pool.hh_col_end;

        for (int j = col_base; j < col_end_c; j++) {
            float* col_j = &MAT(*g_pool.hh_A, 0, j);

            // FIXED: fully HVX dot product + tree reduction (no scalar loop)
            float dot = hvx_dot_col_v(col_j, g_pool.hh_v, m);

            float w = g_pool.hh_tau * dot;
            HVX_Vector vw = Q6_V_vsplat_R(*(uint32_t*)&w);

            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)(col_j + i);
                HVX_Vector vv = *(const HVX_VectorU*)(g_pool.hh_v + i);
                HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, Q6_V_vzero());
                HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
                *(HVX_VectorU*)(col_j + i) =
                    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp));
            }
        }
    }
}
```

**Measurement target:** Isolate the dot product cost by timing a microbenchmark:
- Loop: 1000 × hvx_dot_col_v(col, v, m=640)
- Compare vs the original store+scalar-loop version
- Report cycles per reduction

---

## Fix P2 — Vectorize Pivot Norm on Main Thread (`v2_hvx_pivot.c`)

**Problem:** The main thread computes two norms in `qr_hh_unblocked_pool` as scalar:
```c
// Pass 1: ‖A[k:m, k]‖²  — scalar loop over m-k elements
float nxsq = 0.f;
for (int i = k; i < m; i++) { float val = MAT(*A, i, k); nxsq += val*val; }

// Pass 2: ‖v‖²  — scalar loop over m-k elements  
float nvsq = 0.f;
for (int i = k; i < m; i++) nvsq += v[i]*v[i];
```
Both run while all worker threads are idle (between dispatches).
For m=640, k=0: each pass is 640 scalar multiply-accumulates ≈ 640 cycles.
Across n=64 pivot steps: 2 × 64 × ~320 average = **~41K cycles scalar norm work**
that could be HVX-vectorized.

**Fix:** Main thread acquires HVX before entering the k-loop and uses
`hvx_col_norm_sq` and `hvx_vec_norm_sq`:

```c
static void qr_hh_unblocked_pool_v2(Matrix* A, int chunk_size) {
    int m = A->m, n = A->n;
    float* v = (float*)memalign(128, (size_t)m * sizeof(float));

    // Main thread acquires HVX for pivot steps
    // Workers will also acquire HVX inside pool_worker
    // Total demand = 1 (main) + nt (workers) — may exceed 4-context limit
    // Strategy: main acquires last, after workers are sleeping between dispatches
    // OR: main uses scalar norm if HVX unavailable (fallback)
    int main_hvx = (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0);

    for (int k = 0; k < n; k++) {
        // --- Pivot step (main thread) ---
        float nxsq, nx, alpha, nvsq, tau;

        if (main_hvx) {
            nxsq = hvx_col_norm_sq(&MAT(*A, 0, k), m, k);
        } else {
            nxsq = 0.f;
            for (int i = k; i < m; i++) {
                float val = MAT(*A, i, k); nxsq += val * val;
            }
        }
        nx = sqrtf(nxsq);
        if (nx < 1e-7f) continue;

        alpha = -(MAT(*A, k, k) >= 0.f ? 1.f : -1.f) * nx;
        for (int i = 0; i < k; i++) v[i] = 0.f;
        for (int i = k; i < m; i++) v[i] = MAT(*A, i, k);
        v[k] -= alpha;

        if (main_hvx) {
            nvsq = hvx_vec_norm_sq(v, m, k);
        } else {
            nvsq = 0.f;
            for (int i = k; i < m; i++) nvsq += v[i] * v[i];
        }
        if (nvsq < 1e-10f) continue;
        tau = 2.f / nvsq;

        // --- Dispatch trailing updates ---
        pool_dispatch_col_hh(A, v, tau, k + 1, n, chunk_size);

        MAT(*A, k, k) = alpha;
        for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
    }

    if (main_hvx) qurt_hvx_unlock();
    free(v);
}
```

**HVX contention note:** When main holds HVX + nt=4 workers hold HVX = 5 contexts
requested, but only 4 available. One will fall back to scalar. Measure whether
giving main thread HVX (and losing one worker's HVX) is net positive or negative.
This is an empirical question — run both configurations.

---

## Fix P5 — Adaptive Chunk Size (`v4_adaptive_chunk.c`)

**Problem:** Fixed chunk_size for all k. At late iterations:
```
k = n-4: only 3 trailing columns remain
chunk=4, nt=8: only 1 chunk exists, 7 threads do nothing
```

**Fix:** Compute adaptive chunk in `pool_dispatch_col_hh`:

```c
static void pool_dispatch_col_hh(Matrix* A, const float* v, float tau,
                                 int col_start, int col_end,
                                 int base_chunk) {
    if (col_end <= col_start || !g_pool_running) return;

    int remaining = col_end - col_start;
    // Give each thread at least one chunk; don't go below base_chunk
    // (too-small chunks lose HVX efficiency on short columns)
    int adaptive = (remaining + g_pool.nt - 1) / g_pool.nt;
    int chunk = (adaptive > base_chunk) ? adaptive : base_chunk;

    g_pool.hh_chunk_size = chunk;
    g_pool.kind          = POOL_JOB_COL_HH;
    g_pool.hh_A          = A;
    g_pool.hh_v          = v;
    g_pool.hh_tau        = tau;
    g_pool.hh_col_start  = col_start;
    g_pool.hh_col_end    = col_end;
    g_pool.hh_counter    = 0;
    g_pool.done_count    = g_pool.nt;

    qurt_signal_clear(&g_pool.done_master, 1u);
    for (int t = 0; t < g_pool.nt; t++)
        qurt_signal_set(&g_pool.go[t], 1u);
    qurt_signal_wait(&g_pool.done_master, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
}
```

---

## Fix P3 — DAG-Pipelined Pivot Overlap (`v5_dag_pipeline.c`)

**This is the research contribution. Implement last, after P1/P2/P4/P5 are stable.**

**The insight:** After dispatching trailing updates for step k, the main thread
blocks waiting for ALL n-k-1 columns to finish. But column k+1 is the ONLY
column needed to compute pivot k+1. If we track per-column completion, the
main thread can start computing reflector k+1 as soon as column k+1 is done —
without waiting for columns k+2 ... n-1.

### Data Structure: Per-Column Completion Flags

```c
// Add to ThreadPool struct:
volatile unsigned int col_done[MAX_N];  // col_done[j] = 1 when col j fully updated

// Add to pool_do_col_hh, after updating column j:
g_pool.col_done[j] = 1;
```

### Modified Driver: Overlap Pivot k+1 with Tail of Trailing Updates for k

```c
static void qr_hh_dag_pipeline(Matrix* A, int chunk_size) {
    int m = A->m, n = A->n;
    float* v     = (float*)memalign(128, (size_t)m * sizeof(float));
    float* v_pre = (float*)memalign(128, (size_t)m * sizeof(float));  // prefetched next reflector

    // Pre-compute reflector 0 before entering main loop
    float alpha_pre, tau_pre;
    int   k_pre = 0;
    compute_reflector(A, m, k_pre, v_pre, &alpha_pre, &tau_pre);

    for (int k = 0; k < n; k++) {
        // Dispatch trailing updates for step k using pre-computed reflector
        memcpy(v, v_pre, m * sizeof(float));
        float tau   = tau_pre;
        float alpha = alpha_pre;

        // Reset per-column done flags for columns k+1 .. n-1
        for (int j = k + 1; j < n; j++) g_pool.col_done[j] = 0;

        pool_dispatch_col_hh_nonblocking(A, v, tau, k + 1, n, chunk_size);

        // While workers update columns k+1..n-1,
        // wait ONLY for column k+1, then compute next reflector
        if (k + 1 < n) {
            // Spin until col k+1 is done (it will be among the first dispatched)
            while (!g_pool.col_done[k + 1]) { /* yield or nop */ }

            // Compute reflector k+1 now — column k+1 is fully updated
            compute_reflector(A, m, k + 1, v_pre, &alpha_pre, &tau_pre);
            k_pre = k + 1;
        }

        // Now wait for ALL trailing updates of step k to complete
        pool_wait_done();

        // Apply diagonal/subdiagonal cleanup
        MAT(*A, k, k) = alpha;
        for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
    }

    free(v); free(v_pre);
}
```

### Helper: `compute_reflector`

```c
// Encapsulate the scalar (or HVX) reflector computation for reuse
static void compute_reflector(Matrix* A, int m, int k,
                               float* v_out,
                               float* alpha_out, float* tau_out) {
    float nxsq = 0.f;
    for (int i = k; i < m; i++) {
        float val = MAT(*A, i, k);
        nxsq += val * val;
    }
    float nx = sqrtf(nxsq);
    if (nx < 1e-7f) { *tau_out = 0.f; return; }

    float alpha = -(MAT(*A, k, k) >= 0.f ? 1.f : -1.f) * nx;
    for (int i = 0; i < k; i++) v_out[i] = 0.f;
    for (int i = k; i < m; i++) v_out[i] = MAT(*A, i, k);
    v_out[k] -= alpha;

    float nvsq = 0.f;
    for (int i = k; i < m; i++) nvsq += v_out[i] * v_out[i];
    if (nvsq < 1e-10f) { *tau_out = 0.f; return; }

    *alpha_out = alpha;
    *tau_out   = 2.f / nvsq;
}
```

**Synchronization contract:**
- `pool_dispatch_col_hh_nonblocking`: fires all worker go-signals, returns immediately
- `pool_wait_done`: blocks on `done_master` signal (same as current `pool_dispatch`)
- Workers set `col_done[j] = 1` after finishing each column j

**Race condition to avoid:** Column k+1 must be fully written before main thread
reads it in `compute_reflector`. The `col_done[k+1] = 1` flag (set by worker after
the HVX store loop for column j=k+1 completes) provides this guarantee —
but use a memory barrier before setting it:

```c
// In pool_do_col_hh, after updating column j:
__asm__ __volatile__("" ::: "memory");  // compiler barrier
g_pool.col_done[j] = 1;                // visible to main thread spin
```

---

## Fix P6 — β-Row Batching for Better Instruction-Level Parallelism

**This is optional but worth trying after P3.**

**Observation:** Currently each chunk processes columns independently in a scalar
outer loop. For each column j, the pattern is:
```
dot  = hvx_dot_col_v(col_j, v)     ← 1 vector dot product
A[:,j] -= w * v                     ← 1 vector saxpy
```

Processing β columns simultaneously allows the dot products to be interleaved
with the saxpy updates, hiding qf32 latency:

```c
// Process 2 columns at once (β=2 unroll)
for (int j = col_base; j < col_end_c; j += 2) {
    HVX_Vector vacc0 = Q6_V_vzero();
    HVX_Vector vacc1 = Q6_V_vzero();
    float* col0 = &MAT(*A, 0, j);
    float* col1 = &MAT(*A, 0, j+1);

    for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector vv  = *(const HVX_VectorU*)(v + i);
        HVX_Vector va0 = *(HVX_VectorU*)(col0 + i);
        HVX_Vector va1 = *(HVX_VectorU*)(col1 + i);
        vacc0 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc0, Q6_Vqf32_vmpy_VsfVsf(va0, vv));
        vacc1 = Q6_Vqf32_vadd_Vqf32Vqf32(vacc1, Q6_Vqf32_vmpy_VsfVsf(va1, vv));
    }
    float dot0 = hvx_reduce_qf32(vacc0);
    float dot1 = hvx_reduce_qf32(vacc1);
    // ... saxpy for both columns
}
```

Try β = 2, 4, 8. The optimal β is likely 2 or 4 on v69 due to register pressure
(32 HVX registers shared across all live vectors).

---

## Experiments to Run

### Experiment A — Microbenchmark: HVX Reduction Cost

**Purpose:** Establish baseline cost of horizontal reduction — justifies P1.

```
Config: m=640, n=1 (single column)
Measure:
  - Old: store vSum to temp[32] + scalar loop (32 adds)  → X cycles
  - New: hvx_reduce_qf32 (5 rotate-add pairs)           → Y cycles
  - Ratio X/Y = reduction speedup
```

### Experiment B — Ablation: Each Fix in Isolation

Run all 6 variants for the same config to isolate each fix's contribution:

```
Config: m=640, n=64, nt=4, chunk=4

Variant               Cycles      Speedup over v0
v0_pool_baseline      [measure]   1.0×
v1_hvx_reduction      [measure]   P1 only
v2_hvx_pivot          [measure]   P2 only
v3_persistent_hvx     [measure]   P4 only
v4_adaptive_chunk     [measure]   P5 only
v5_dag_pipeline       [measure]   P3 only
v6_combined           [measure]   all fixes
```

### Experiment C — Sweep: Matrix Size

```
Fix: nt=4, chunk=adaptive
Sweep: m ∈ {128, 256, 384, 512, 640, 768, 896, 1024}, n ∈ {16, 32, 64}
Variants: scalar_ref, v0_pool_baseline, v6_combined
Metric: cycles, cycles/flop, speedup
```

### Experiment D — Sweep: Thread Count

```
Fix: m=640, n=64, chunk=adaptive
Sweep: nt ∈ {1, 2, 3, 4, 5, 6, 7, 8}
Variants: v0_pool_baseline, v6_combined
Key question: does v6 show better scaling beyond nt=4 than v0?
Expected: v6 degrades more gracefully because DAG pipelining keeps
          scalar threads (nt>4, no HVX) busy on pivot prefetch
```

### Experiment E — Sweep: Chunk Size (for v6_combined, fixed chunk variant)

```
Fix: m=640, n=64, nt=4
Sweep: chunk ∈ {1, 2, 4, 8, 16, 32}
Purpose: validate that adaptive chunk (P5) finds the right value
```

---

## Correctness Verification

Every variant must pass this check (already in v0, keep it everywhere):

```c
float err = verify_R_against(&A_cpu, &A_alg);
printf("max |R|-err: %.3e  %s\n", err, err < 1e-3f ? "OK" : "FAIL");
```

Additionally add:
```c
// Check R is upper triangular (no residual below diagonal)
float sub_err = 0.f;
for (int j = 0; j < A->n; j++)
    for (int i = j+1; i < A->m; i++)
        sub_err = fmaxf(sub_err, fabsf(MAT(*A, i, j)));
printf("sub-diagonal residual: %.3e  %s\n", sub_err, sub_err < 1e-5f ? "OK" : "FAIL");
```

---

## Expected Results and Paper Claims

Based on analysis of v0's bottlenecks:

| Fix | Expected Saving | Basis |
|-----|----------------|-------|
| P4: persistent HVX | 20-30% of total | 5K cyc/lock × n × nt |
| P1: tree reduction | 5-10% of total | 40 cyc × n(n-1)/2 columns |
| P2: HVX pivot norm | 2-5% of total | scalar norm ÷ HVX norm |
| P5: adaptive chunk | 2-5% at tail | idle thread elimination |
| P3: DAG pipeline | 10-20% of total | pivot overlap with trailing tail |
| Combined | 35-55% over v0 | Amdahl composition |

Combined speedup over **scalar_ref** target: **30-50× for m=640, n=64, nt=4**.

**The publishable claim:** First characterization of Householder QR on Hexagon v69
HVX with: (1) quantified horizontal reduction cost on HVX, (2) DAG-pipelined
pivot overlap exploiting per-column completion tracking, (3) HVX-context-aware
persistent thread pool. Demonstrates X× over scalar baseline and Y× over
naive HVX+MT (v0), with analytical model predicting the reduction-bottleneck
ceiling.

---

## Implementation Order (for Claude Code)

```
Day 1:
  1. Create repository layout and common.h
  2. Copy v0_pool_baseline.c verbatim — do not touch
  3. Implement hvx_utils.h (hvx_reduce_qf32, hvx_col_norm_sq, hvx_dot_col_v)
  4. Implement v3_persistent_hvx.c (P4) — highest ROI, lowest risk
  5. Run Experiment B for v0 vs v3. Confirm cycle reduction.

Day 2:
  6. Implement v1_hvx_reduction.c (P1) — apply hvx_dot_col_v in pool_do_col_hh
  7. Implement v2_hvx_pivot.c (P2) — HVX norm in main thread pivot step
  8. Implement v4_adaptive_chunk.c (P5) — one-liner in dispatch function
  9. Run Experiment A (microbenchmark reduction cost)
  10. Run Experiment B for all single-fix variants

Day 3:
  11. Implement v5_dag_pipeline.c (P3)
      - Add col_done[] array to ThreadPool
      - Implement pool_dispatch_col_hh_nonblocking + pool_wait_done
      - Implement compute_reflector helper
      - Implement qr_hh_dag_pipeline driver
  12. Verify correctness carefully — race conditions likely here
  13. Run Experiment B for v5_dag_pipeline in isolation

Day 4:
  14. Implement v6_combined.c — all fixes together
  15. Run Experiments C, D, E (full sweeps)
  16. Run scripts/parse_results.py — generate tables and plots
  17. Optional: implement P6 β-row batching if time permits
```

---

## Notes for Claude Code

- All source files share the same `inparams.txt` format: `m n nt chunk use_static nb`
- The `use_static` and `nb` fields are currently unused — reserve them for future
  blocked variants
- All matrices must have m divisible by 32 (HVX alignment requirement)
- Column-major storage: `MAT(A, i, j) = A.data[j*m + i]`
- HVX loads must be 128-byte aligned: use `memalign(128, ...)` for all matrix allocs
- The ISS timing flag `--timing` must be set for `get_cycles()` to return real counts
- Do NOT use `printf` inside hot loops — it serializes and invalidates cycle counts
- Verify each variant independently before combining into v6_combined
