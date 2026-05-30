# QR Algorithm Comparison — Hexagon HVX + QuRT

Latest results combining the original variants from `qr_compare_all.c` with the new
**persistent-thread-pool** variants in separate files.

| File | Variant | Notes |
|---|---|---|
| [qr_compare_all.c](../qr_compare_all.c) | 5 algorithms in one binary | Per-spawn HH (legacy thread model) + MGS2 + Givens + Tile-QR |
| [qr_pool_hh_unblocked.c](../qr_pool_hh_unblocked.c) | `pool_hh_unblocked` | **Current winner.** HH unblocked + persistent thread pool |
| [qr_pool_hh_wy_blocked.c](../qr_pool_hh_wy_blocked.c) | `pool_hh_wy_blocked` | WY-blocked + pool for trail update |

All verified against CPU scalar Householder reference; max |R| abs-diff well below 1e-3.

Plots: [compare_64.png](compare_64.png), [compare_128.png](compare_128.png)

---

## Final scoreboard

### m=64, n=32

| # | Variant | Cycles | vs CPU |
|---|---|---|---|
| — | CPU scalar HH (ref) | 311K | 1.0× |
| **1** | **`pool_hh_unblocked`** | **115K** | **2.7×** ← winner |
| 2 | `pool_hh_wy_blocked` | 146K | 2.1× |
| 3 | HH WY-blocked (no pool) | 263K | 1.2× |
| 4 | MGS2 HVX (1 thr) | 277K | 1.1× |
| 5 | Givens HVX (1 thr) | 452K | 0.69× |
| 6 | HH unblocked HVX-MT (per-spawn) | 677K | 0.46× |
| 7 | Tile QR (2-way TSQR) | 1.77M | 0.18× |

### m=128, n=128

| # | Variant | Cycles | vs CPU |
|---|---|---|---|
| — | CPU scalar HH (ref) | 3.8M | 1.0× |
| **1** | **`pool_hh_unblocked`** | **342K** | **11.1×** ← winner |
| 2 | HH unblocked HVX-MT (per-spawn) | 1.14M | 3.3× |
| 3 | `pool_hh_wy_blocked` | 1.30M | 2.9× |
| 4 | HH WY-blocked (no pool) | 1.87M | 2.0× |
| 5 | Givens HVX (1 thr) | 1.96M | 1.9× |
| 6 | MGS2 HVX (1 thr) | 4.94M | 0.77× |

---

## Headline finding

**Thread-spawn overhead — not anything algorithmic — was the dominant bottleneck.**
Replacing per-Householder `qurt_thread_create`/`qurt_thread_join` with a persistent
thread pool (workers wait on `qurt_signal_t`, do their slice, signal done, sleep)
collapsed cycles by 3–6× on the *same kernel*. At m=128 the simplest algorithm
(HH unblocked) now runs **11× faster than scalar CPU** — matching the headline
speedup the referenced [paper](../souyajit-paper.pdf) reported on Intel CPUs
with 26–104 threads, achieved here with just 4 Hexagon threads (~2 HVX contexts
effective).

## Why blocking still doesn't win on Hexagon at these sizes

`pool_hh_wy_blocked` is faster than its non-pool predecessor (1.30M vs 1.87M)
but slower than `pool_hh_unblocked` (1.30M vs 342K). Decomposing the m=128
budget:

- 16 panels × 8 intra-panel Householders = 128 single-thread HVX column updates
  ≈ **800K cyc** (the panel phase is inherently serial since the col range is
  too narrow to thread)
- 16 build_T calls (scalar) ≈ **160K cyc**
- 16 block-update dispatches × pool overhead ≈ **16K cyc**
- Block-update HVX useful work (4-way threaded) ≈ **300K cyc**

The single-threaded panel phase is the killer. For blocking to win on this
hardware we'd need either (a) a much larger matrix where the trail-update GEMM
dominates and the panel phase is a negligible fraction, or (b) a way to extract
parallelism from within a narrow panel (e.g., row-tile parallelism inside the
panel — bigger restructuring).

## Implementation notes — persistent thread pool

- **QuRT signals are single-waiter.** Multiple workers waiting on the same
  `qurt_signal_t` triggers a kernel assertion (`qurt_futex_c.c:73`). Fix: one
  signal per worker (worker T waits on `g_pool.go[T]`, set by main).
- **Done synchronization** uses one shared signal (`done_master`) that *only the
  main thread* waits on. Workers atomic-decrement a counter (`qurt_atomic_add_return`
  with `-1u`); the worker that observes `0` sets `done_master`.
- **HVX lock per job, not per worker lifetime.** Workers re-acquire HVX inside
  the per-job body so other code paths (single-threaded HVX algorithms,
  scalar CPU code) can use HVX between dispatches.

## What we did NOT pursue (and why)

- **Async DAG scheduler with two lock-free queues** (the paper's actual
  scheduling contribution). At our matrix sizes the pool dispatch already
  gets ~1K cyc/dispatch — the marginal win of work-stealing critical-path-first
  scheduling on 4 threads with 2 HVX contexts is small (~1.5–2× at best). Worth
  pursuing if we ever move to a denser-thread chip or bigger matrices.
- **Pool-based MGS2 / Givens.** Single-thread HVX is already competitive (or
  better) for these algorithms at our sizes since they have no per-step
  reduction overhead to amortize. Threading them adds row-partition reduction
  overhead that likely costs more than it saves at m≤256.
- **Proper CAQR (Communication-Avoiding QR).** Would be ~500 LOC of substantial
  new code. Real domain is tall-skinny QR (m ≫ n), not the square problems we
  care about.

## Recommendation

For production QR on Hexagon v69 with 4 threads, m≈64–256 square:

**Use `qr_pool_hh_unblocked`.**

It's the simplest implementation, hits 11× speedup over scalar at m=128 (target
size), uses minimal scratch memory, and the kernel is straightforward to extend
(e.g., add L2 prefetch, or fold in the WY panel structure once trail update
dominates total work — likely m ≥ 512).
