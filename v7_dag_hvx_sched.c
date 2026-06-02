/* =====================================================================
 *  v7_dag_hvx_sched — DAG-priority HVX scheduling for Householder QR.
 *
 *  Brings the task-graph scheduling ideas of Chatterjee et al.
 *  ("Efficient Task Graph Scheduling for Parallel QR Factorization in
 *   SLSQP", arXiv 2506.09463) to Hexagon v69 HVX, and adds the platform's
 *  novel twist: the scarce resource is the *HVX vector context* (4 of them
 *  in 128B mode), allocated to tasks by DAG priority.
 *
 *  Task DAG (column-major Householder):
 *    T(k,k)  = compute reflector for column k          (critical path)
 *    T(k,j)  = trailing update of column j>k
 *  Edges: T(k,j) depends on T(k,k)  (reflector before update, intra-step)
 *               and  on T(k-1,j)    (column j updated in prev step, inter-step)
 *
 *  Critical path: T(0,1) -> T(1,1) -> T(1,2) -> T(2,2) -> ...  i.e. at step k
 *  the ONLY critical-path trailing task is the update of column k+1, because
 *  reflector k+1 cannot start until column k+1 has been updated.  Hence
 *  "critical-path priority" == "update column k+1 first, then compute
 *  reflector k+1 while columns k+2..n-1 of step k are still updating."
 *
 *  Two schedulers (runtime-selectable via the 5th inparams field):
 *    sched=0  SCHED_ASYNC_PRIO  : pivot k+1 overlaps the tail of step k
 *                                 (waits only on col_done[k+1], not on all);
 *                                 columns claimed in ascending (priority) order;
 *                                 HVX contexts grabbed with qurt_hvx_try_lock so
 *                                 the >4 case degrades to scalar instead of
 *                                 blocking — i.e. contexts go to the first
 *                                 (highest-priority) claimants.
 *    sched=1  SCHED_BARRIER     : v0-equivalent (reflector k+1 only after the
 *                                 whole step-k barrier) — sanity / ablation ref.
 *    sched=2  SCHED_ASYNC       : same async overlap as ASYNC_PRIO but WITHOUT
 *                                 critical-path priority — main does no special
 *                                 critical column and workers claim columns
 *                                 descending.  Isolates the priority policy
 *                                 (everything else identical to ASYNC_PRIO).
 *    sched=3  SCHED_ASYNC_PRIO_CC: async_prio with a CONCURRENT critical column.
 *                                 Main retains one HVX context, dispatches the
 *                                 workers (budget = units-1) first, then crunches
 *                                 column k+1 on its own context in parallel —
 *                                 so all 4 contexts are busy during the critical
 *                                 column instead of just 1.  Wins when the
 *                                 critical column dominates (tall-skinny).
 *
 *  inparams.txt format: m n nt [chunk_size] [sched_mode] [nb]
 * ===================================================================== */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include "qurt.h"
#include "qurt_thread.h"
#include "qurt_atomic_ops.h"
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

#include "hvx_utils.h"   /* hvx_dot_col_v, hvx_reduce_qf32, hvx_col_norm_sq */

QURT_DECLARE_STATIC_HEAP(16777216 * 2);
unsigned int heapSize = 16777216;

#define HVX_FLOAT_LANES 32
#define MAX_THREADS     8
#define MAX_N           4096
#define STACK_SIZE      (16 * 1024 * 2)
#define DEFAULT_CHUNK   4
#define DEFAULT_NB      8

enum { SCHED_ASYNC_PRIO = 0, SCHED_BARRIER = 1, SCHED_ASYNC = 2,
       SCHED_ASYNC_PRIO_CC = 3, SCHED_CC_QUEUE = 4 };

static const char* VARIANT_NAME = "dag_hvx_sched";

/* Number of 128B HVX contexts the chipset provides (bits 15:8 of get_units;
 * e.g. 0x400 == QURT_HVX_HW_UNITS_4X128B_0X64B == 4).  Filled in main(). */
static int g_hvx_units_128b = 4;
static inline int hvx_units_128b(void) {
    int u = (qurt_hvx_get_units() >> 8) & 0xff;
    return (u > 0) ? u : 4;
}

static char thread_stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8)));

typedef HVX_Vector __attribute__((aligned(4))) HVX_VectorU;

typedef struct { float* data; int m; int n; } Matrix;
#define MAT(mat, i, j) ((mat).data[(j) * (mat).m + (i)])

static inline uint64_t get_cycles(void) {
    uint64_t c;
    __asm__ __volatile__("%0 = c15:14" : "=r"(c) : : "memory");
    return c;
}

/* ---------------- Matrix helpers ---------------- */
static Matrix create_matrix(int m, int n) {
    if (m % HVX_FLOAT_LANES != 0) {
        printf("ERROR: m must be a multiple of %d\n", HVX_FLOAT_LANES);
        exit(1);
    }
    Matrix mat = { NULL, m, n };
    mat.data = (float*)memalign(128, (size_t)m * n * sizeof(float));
    if (!mat.data) { printf("ERROR: alloc failed\n"); exit(1); }
    memset(mat.data, 0, (size_t)m * n * sizeof(float));
    return mat;
}
static Matrix copy_matrix(const Matrix* src) {
    Matrix dst = create_matrix(src->m, src->n);
    memcpy(dst.data, src->data, (size_t)src->m * src->n * sizeof(float));
    return dst;
}
static void free_matrix(Matrix* mat) {
    if (mat->data) { free(mat->data); mat->data = NULL; }
}
static void fill_test_matrix(Matrix* A) {
    for (int i = 0; i < A->m; i++)
        for (int j = 0; j < A->n; j++)
            MAT(*A, i, j) = (float)((i*3 + j*7) % 11) - 5.f;
}
static float verify_R_against(const Matrix* ref, const Matrix* alg) {
    float max_err = 0.f;
    for (int j = 0; j < ref->n; j++)
        for (int i = 0; i <= j && i < ref->m; i++) {
            float d = fabsf(fabsf(MAT(*ref, i, j)) - fabsf(MAT(*alg, i, j)));
            if (d > max_err) max_err = d;
        }
    return max_err;
}
/* Largest magnitude strictly below the diagonal — should be ~0 for valid R. */
static float subdiag_residual(const Matrix* A) {
    float e = 0.f;
    for (int j = 0; j < A->n; j++)
        for (int i = j + 1; i < A->m; i++) {
            float a = fabsf(MAT(*A, i, j));
            if (a > e) e = a;
        }
    return e;
}

/* ---------------- CPU scalar HH reference ---------------- */
static void qr_cpu_householder(Matrix* A) {
    int m = A->m, n = A->n;
    float* v = (float*)malloc((size_t)m * sizeof(float));
    for (int k = 0; k < n; k++) {
        float nxsq = 0.f;
        for (int i = k; i < m; i++) { float val = MAT(*A, i, k); nxsq += val*val; }
        float nx = sqrtf(nxsq); if (nx < 1e-7f) continue;
        float alpha = -(MAT(*A, k, k) >= 0.f ? 1.f : -1.f) * nx;
        for (int i = k; i < m; i++) v[i] = MAT(*A, i, k);
        v[k] -= alpha;
        float nvsq = 0.f;
        for (int i = k; i < m; i++) nvsq += v[i]*v[i];
        if (nvsq < 1e-10f) continue;
        float tau = 2.f / nvsq;
        for (int j = k + 1; j < n; j++) {
            float dot = 0.f;
            for (int i = k; i < m; i++) dot += v[i] * MAT(*A, i, j);
            float w = tau * dot;
            for (int i = k; i < m; i++) MAT(*A, i, j) -= w * v[i];
        }
        MAT(*A, k, k) = alpha;
        for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
    }
    free(v);
}

/* compute_reflector — fills v[0..m), returns alpha/tau for column k.
 * v[0..k-1] are zeroed so the HVX kernels can iterate i in [0,m) uniformly. */
static int compute_reflector(Matrix* A, int m, int k,
                             float* v, float* alpha_out, float* tau_out) {
    float nxsq = 0.f;
    for (int i = k; i < m; i++) { float val = MAT(*A, i, k); nxsq += val*val; }
    float nx = sqrtf(nxsq);
    if (nx < 1e-7f) { *tau_out = 0.f; return 0; }
    float alpha = -(MAT(*A, k, k) >= 0.f ? 1.f : -1.f) * nx;
    for (int i = 0; i < k; i++) v[i] = 0.f;
    for (int i = k; i < m; i++) v[i] = MAT(*A, i, k);
    v[k] -= alpha;
    float nvsq = 0.f;
    for (int i = k; i < m; i++) nvsq += v[i]*v[i];
    if (nvsq < 1e-10f) { *tau_out = 0.f; return 0; }
    *alpha_out = alpha;
    *tau_out   = 2.f / nvsq;
    return 1;
}

/* =====================================================================
 *  Persistent thread pool with per-column completion tracking
 * ===================================================================== */
typedef enum { POOL_JOB_NONE = 0, POOL_JOB_COL_HH, POOL_JOB_SHUTDOWN } PoolJobKind;

typedef struct {
    int                   nt;
    volatile PoolJobKind  kind;

    qurt_signal_t         go[MAX_THREADS];
    qurt_signal_t         done_master;
    volatile unsigned int done_count;

    /* COL_HH job descriptor */
    Matrix*               hh_A;
    const float*          hh_v;
    float                 hh_tau;
    int                   hh_col_start;
    int                   hh_col_end;
    unsigned int          hh_counter;     /* atomic chunk-claim counter */
    int                   hh_chunk_size;
    /* Claim order: +1 = ascending columns (critical-path-first = priority),
     * -1 = descending (no priority — the SCHED_ASYNC ablation).  Isolates the
     * priority-ordering policy: everything else (kernels, overlap, HVX budget)
     * is identical between async and async_prio. */
    int                   hh_claim_dir;

    /* HVX-context allocation: at most hvx_worker_units workers may hold an
     * HVX context; the rest fall back to scalar.  A worker claims a slot with
     * one atomic inc, so qurt_hvx_lock() is only ever called when a context is
     * provably free (no blocking stalls, and no reliance on the deprecated
     * qurt_hvx_try_lock which does not grant a context on v65+). */
    volatile unsigned int hvx_claim;       /* atomic slot counter per dispatch */
    int                   hvx_worker_units; /* contexts available to workers   */

    /* DAG: per-column completion for the current dispatch (critical-path
     * pivot overlap reads col_done[k+1]). */
    volatile int          col_done[MAX_N];
} ThreadPool;

static ThreadPool    g_pool;
static qurt_thread_t g_pool_tids[MAX_THREADS];
static int           g_pool_running = 0;

/* HVX trailing-column update: A[:,j] -= (tau * v^T A[:,j]) v */
static inline void apply_col_hvx(Matrix* A, const float* v, float tau, int j, int m) {
    float* col_j = &MAT(*A, 0, j);
    float dot = hvx_dot_col_v(col_j, v, m);          /* dot + HVX tree reduce */
    float w = tau * dot;
    HVX_Vector vw = Q6_V_vsplat_R(*(uint32_t*)&w);
    for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector va = *(HVX_VectorU*)(col_j + i);
        HVX_Vector vv = *(const HVX_VectorU*)(v + i);
        HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, Q6_V_vzero());
        HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
        *(HVX_VectorU*)(col_j + i) =
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp));
    }
}

/* Scalar fallback for workers that did not win an HVX context (nt > 4). */
static inline void apply_col_scalar(Matrix* A, const float* v, float tau, int j, int m) {
    float* col_j = &MAT(*A, 0, j);
    float dot = 0.f;
    for (int i = 0; i < m; i++) dot += v[i] * col_j[i];
    float w = tau * dot;
    for (int i = 0; i < m; i++) col_j[i] -= w * v[i];
}

/* Worker body: claim a chunk of columns (ascending for async_prio = priority,
 * descending for async = no priority), grab an HVX context once via the atomic
 * budget, run the update, and publish col_done[j].
 *
 * Claim order is the ONLY difference between the async and async_prio policies:
 * the i-th claimed chunk (atomic ci = 0,1,2,...) maps to the chunk that many
 * steps in from the start (dir=+1) or from the end (dir=-1) of [col_start,
 * col_end).  Same single atomic counter, so no new races. */
static void pool_do_col_hh(void) {
    Matrix* A = g_pool.hh_A;
    int m     = A->m;
    int chunk = g_pool.hh_chunk_size;
    const float* v = g_pool.hh_v;
    float tau = g_pool.hh_tau;
    int dir   = g_pool.hh_claim_dir;     /* +1 ascending (prio), -1 descending */
    int nchunk = (g_pool.hh_col_end - g_pool.hh_col_start + chunk - 1) / chunk;
    int got_hvx = 0;   /* 0 = not yet claimed, 1 = holds context, -1 = scalar */

    while (1) {
        unsigned int ci = qurt_atomic_inc_return(&g_pool.hh_counter) - 1;
        if ((int)ci >= nchunk) break;
        /* chunk index within the range, ascending or descending */
        int cidx = (dir >= 0) ? (int)ci : (nchunk - 1 - (int)ci);
        int col_base    = g_pool.hh_col_start + cidx * chunk;
        int col_end_c   = col_base + chunk;
        if (col_end_c > g_pool.hh_col_end) col_end_c = g_pool.hh_col_end;

        if (got_hvx == 0) {                       /* claim an HVX context once */
            unsigned int slot = qurt_atomic_inc_return(&g_pool.hvx_claim);
            if ((int)slot <= g_pool.hvx_worker_units &&
                qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK)
                got_hvx = 1;
            else
                got_hvx = -1;                     /* over budget -> scalar */
        }
        int use_hvx = (got_hvx == 1);

        for (int j = col_base; j < col_end_c; j++) {
            if (use_hvx) apply_col_hvx(A, v, tau, j, m);
            else         apply_col_scalar(A, v, tau, j, m);
            __asm__ __volatile__("" ::: "memory");   /* publish before flag */
            g_pool.col_done[j] = 1;
        }
    }
    if (got_hvx == 1) qurt_hvx_unlock();
}

static void pool_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    qurt_signal_t* my_go = &g_pool.go[tid];
    while (1) {
        qurt_signal_wait(my_go, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
        qurt_signal_clear(my_go, 1u);
        if (g_pool.kind == POOL_JOB_SHUTDOWN) break;
        if (g_pool.kind == POOL_JOB_COL_HH) pool_do_col_hh();
        unsigned int remaining = qurt_atomic_add_return(&g_pool.done_count, -1u);
        if (remaining == 0) qurt_signal_set(&g_pool.done_master, 1u);
    }
    qurt_thread_exit(0);
}

static void pool_init(int nt) {
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    if (nt < 1) nt = 1;
    g_pool.nt = nt;
    g_pool.kind = POOL_JOB_NONE;
    qurt_signal_init(&g_pool.done_master);
    for (int t = 0; t < nt; t++) qurt_signal_init(&g_pool.go[t]);
    qurt_thread_attr_t attr;
    for (int t = 0; t < nt; t++) {
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
        qurt_thread_attr_set_priority(&attr, 100);
        qurt_thread_create(&g_pool_tids[t], &attr, pool_worker, (void*)(intptr_t)t);
    }
    g_pool_running = 1;
}

static void pool_shutdown(void) {
    if (!g_pool_running) return;
    g_pool.kind = POOL_JOB_SHUTDOWN;
    g_pool.done_count = g_pool.nt;
    qurt_signal_clear(&g_pool.done_master, 1u);
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_set(&g_pool.go[t], 1u);
    for (int t = 0; t < g_pool.nt; t++) { int s; qurt_thread_join(g_pool_tids[t], &s); }
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_destroy(&g_pool.go[t]);
    qurt_signal_destroy(&g_pool.done_master);
    g_pool_running = 0;
}

/* Fire the trailing-column updates for columns [col_start,col_end) but DO
 * NOT block.  claim_dir: +1 = workers claim ascending columns (critical-path
 * priority), -1 = descending (no priority).  Returns 1 if workers were
 * dispatched, 0 if there was no work. */
static int pool_dispatch_nonblocking(Matrix* A, const float* v, float tau,
                                     int col_start, int col_end, int chunk_size,
                                     int claim_dir) {
    if (col_end <= col_start || !g_pool_running) return 0;
    if (chunk_size < 1) chunk_size = 1;
    for (int j = col_start; j < col_end; j++) g_pool.col_done[j] = 0;
    g_pool.kind          = POOL_JOB_COL_HH;
    g_pool.hh_A          = A;
    g_pool.hh_v          = v;
    g_pool.hh_tau        = tau;
    g_pool.hh_col_start  = col_start;
    g_pool.hh_col_end    = col_end;
    g_pool.hh_counter    = 0;
    g_pool.hvx_claim     = 0;        /* reset HVX-context slot counter */
    g_pool.hh_chunk_size = chunk_size;
    g_pool.hh_claim_dir  = claim_dir;
    g_pool.done_count    = g_pool.nt;
    qurt_signal_clear(&g_pool.done_master, 1u);
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_set(&g_pool.go[t], 1u);
    return 1;
}

static void pool_wait_done(void) {
    qurt_signal_wait(&g_pool.done_master, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
}

/* =====================================================================
 *  Drivers
 * ===================================================================== */

/* Async pivot overlap WITHOUT busy-waiting, shared by two policies:
 *
 *   prio = 1  (SCHED_ASYNC_PRIO): critical-path priority.  The main thread
 *             *executes the critical column k+1 itself* on a briefly-held HVX
 *             context, computes reflector k+1 while the workers handle the
 *             lower-priority columns k+2..n-1, and the workers claim columns
 *             ASCENDING (nearest the critical path first).
 *
 *   prio = 0  (SCHED_ASYNC): same async double-buffered overlap, but NO
 *             critical-path priority.  The main thread does NOT special-case
 *             column k+1; all trailing columns [k+1, n) go to the workers,
 *             which claim them DESCENDING (no priority ordering).  Reflector
 *             k+1 still overlaps the workers (that is the async pipelining,
 *             not the priority).  This is the ablation that isolates the
 *             *priority* policy: kernels, pool, HVX budget, and the overlap
 *             of the reflector are all identical to async_prio.
 *
 * Key lesson (both): spinning the main thread on col_done[] is catastrophic on
 * this in-order SMT DSP (~5x slowdown), so the main thread always does useful
 * work or sleeps.  Workers get ALL HVX contexts; the main thread only locks a
 * context *briefly* for the critical column in the prio path, when no worker
 * is active. */
static void qr_dag_async(Matrix* A, int chunk_size, int prio) {
    int m = A->m, n = A->n;
    float* vbuf[2];
    float  tau[2], alpha[2];
    int    valid[2];
    vbuf[0] = (float*)memalign(128, (size_t)m * sizeof(float));
    vbuf[1] = (float*)memalign(128, (size_t)m * sizeof(float));
    int cur = 0;
    int dir = prio ? +1 : -1;          /* ascending (priority) vs descending */

    g_pool.hvx_worker_units = g_hvx_units_128b;

    valid[cur] = compute_reflector(A, m, 0, vbuf[cur], &alpha[cur], &tau[cur]);

    for (int k = 0; k < n; k++) {
        int nxt = cur ^ 1;
        int dispatched = 0;
        /* prio: main does the critical col k+1 itself, workers take [k+2,n).
         * non-prio: main does no special column, workers take all of [k+1,n). */
        int col_first = prio ? (k + 2) : (k + 1);

        if (prio && valid[cur] && k + 1 < n) {
            /* Critical-path task T(k,k+1): update column k+1 first, on HVX,
             * while no workers hold a context (prev step already joined). */
            if (qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK) {
                apply_col_hvx(A, vbuf[cur], tau[cur], k + 1, m);
                qurt_hvx_unlock();
            } else {
                apply_col_scalar(A, vbuf[cur], tau[cur], k + 1, m);
            }
        }

        if (valid[cur])
            dispatched = pool_dispatch_nonblocking(A, vbuf[cur], tau[cur],
                                                   col_first, n, chunk_size, dir);

        /* Reflector k+1 reads column k+1.  In the prio path the main thread has
         * already finished column k+1 itself (above), so computing reflector
         * k+1 NOW safely overlaps the workers (who only touch k+2..n-1).  In the
         * non-prio path the workers own column k+1, so we must JOIN first to
         * avoid reading a half-updated column — overlap is therefore only
         * available to the priority policy. */
        if (prio) {
            if (k + 1 < n)
                valid[nxt] = compute_reflector(A, m, k + 1, vbuf[nxt],
                                               &alpha[nxt], &tau[nxt]);
            if (dispatched) pool_wait_done();
        } else {
            if (dispatched) pool_wait_done();   /* join: column k+1 now final */
            if (k + 1 < n)
                valid[nxt] = compute_reflector(A, m, k + 1, vbuf[nxt],
                                               &alpha[nxt], &tau[nxt]);
        }

        if (valid[cur]) {
            MAT(*A, k, k) = alpha[cur];
            for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
        }
        cur = nxt;
    }

    free(vbuf[0]); free(vbuf[1]);
}

/* SCHED_ASYNC_PRIO_CC: identical to async_prio, EXCEPT the main thread holds
 * its single HVX context across the WHOLE critical path instead of locking and
 * unlocking it once per step.  The critical path here is the diagonal chain of
 * reflectors/critical-columns the main thread walks — i.e. the entire k-loop —
 * so "critical path complete" == "the loop has finished".  Main therefore locks
 * the context once before the loop and unlocks once after it.
 *
 * Everything else is copied verbatim from async_prio (prio=1): main applies the
 * critical column k+1 itself, dispatches [k+2,n) to the workers (which use the
 * remaining contexts), and computes reflector k+1 overlapping the workers. */
static void qr_dag_async_prio_cc(Matrix* A, int chunk_size) {
    int m = A->m, n = A->n;
    float* vbuf[2];
    float  tau[2], alpha[2];
    int    valid[2];
    vbuf[0] = (float*)memalign(128, (size_t)m * sizeof(float));
    vbuf[1] = (float*)memalign(128, (size_t)m * sizeof(float));
    int cur = 0;

    /* Workers get all-but-one context; main reserves the remaining one for the
     * whole critical path. */
    int worker_units = g_hvx_units_128b - 1;
    if (worker_units < 1) worker_units = 1;
    g_pool.hvx_worker_units = worker_units;

    /* (***the only change vs async_prio***) lock the critical-path context ONCE,
     * up front, and hold it for the entire k-loop. */
    int main_hvx = (qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK);

    valid[cur] = compute_reflector(A, m, 0, vbuf[cur], &alpha[cur], &tau[cur]);

    for (int k = 0; k < n; k++) {
        int nxt = cur ^ 1;
        int dispatched = 0;

        /* critical column k+1 on main's already-held context (no per-step lock) */
        if (valid[cur] && k + 1 < n) {
            if (main_hvx) apply_col_hvx(A, vbuf[cur], tau[cur], k + 1, m);
            else          apply_col_scalar(A, vbuf[cur], tau[cur], k + 1, m);
        }

        if (valid[cur])
            dispatched = pool_dispatch_nonblocking(A, vbuf[cur], tau[cur],
                                                   k + 2, n, chunk_size, +1);

        if (k + 1 < n)
            valid[nxt] = compute_reflector(A, m, k + 1, vbuf[nxt],
                                           &alpha[nxt], &tau[nxt]);

        if (dispatched) pool_wait_done();

        if (valid[cur]) {
            MAT(*A, k, k) = alpha[cur];
            for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
        }
        cur = nxt;
    }

    /* critical path complete -> release the context. */
    if (main_hvx) qurt_hvx_unlock();

    free(vbuf[0]); free(vbuf[1]);
}

/* SCHED_BARRIER: v0-equivalent reference — reflector k computed only after
 * the previous step's full barrier (no pivot overlap). */
static void qr_dag_barrier(Matrix* A, int chunk_size) {
    int m = A->m, n = A->n;
    g_pool.hvx_worker_units = g_hvx_units_128b;   /* workers use all contexts */
    float* v = (float*)memalign(128, (size_t)m * sizeof(float));
    for (int k = 0; k < n; k++) {
        float alpha, tau;
        if (!compute_reflector(A, m, k, v, &alpha, &tau)) continue;
        int dispatched = pool_dispatch_nonblocking(A, v, tau, k + 1, n,
                                                   chunk_size, +1);
        if (dispatched) pool_wait_done();
        MAT(*A, k, k) = alpha;
        for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
    }
    free(v);
}

/* =====================================================================
 *  SCHED_CC_QUEUE — concurrent task-DAG scheduler (v8-proven model) with an
 *  HVX-accelerated critical path.
 *
 *  We pre-build the FULL task DAG up front with predecessor counts, then run
 *  it with a shared ready-queue: a task is pushed only when all its
 *  predecessors are done (no requeue, no run-ahead ring, no busy-wait).  This
 *  is the exact pattern proven correct in v8_tiled_dag_hvx.c.
 *
 *  Two task kinds (unblocked, column granularity):
 *    REFL(k)   : compute reflector k (scalar) + write the diagonal/subdiagonal
 *                of column k.  Reads column k.
 *    UPD(k,j)  : apply reflector k to trailing column j  (k < j < n).
 *
 *  Dependencies (column last-writer):
 *    REFL(k)  <- last writer of column k  (= UPD(k-1,k), the critical column)
 *    UPD(k,j) <- REFL(k)                  (reflector must exist)
 *             <- last writer of column j  (= UPD(k-1,j), inter-step order)
 *  Critical path = REFL(0) -> UPD(0,1) -> REFL(1) -> UPD(1,2) -> ...
 *
 *  HVX-accelerated critical path: the MAIN thread executes the critical-path
 *  tasks itself — every REFL(k) (scalar) and every critical column UPD(k,k+1)
 *  (on a held HVX context) — so reflectors are produced as fast as possible and
 *  downstream UPD tasks become ready sooner, widening the parallel front.  The
 *  WORKERS execute the non-critical UPD(k,j>k+1) tasks pulled from the queue.
 *
 *  Scheduling: when a task completes it decrements each successor's pred count;
 *  a successor hitting 0 is routed to MAIN (if it is a critical-path task) or to
 *  the worker queue (otherwise).  Main blocks on its own semaphore for the next
 *  critical-path task; workers block on the shared queue semaphore.
 * ===================================================================== */
enum { CK_REFL = 0, CK_UPD = 1 };

typedef struct {
    uint8_t  kind;            /* CK_REFL / CK_UPD */
    int      k, j, jend;      /* REFL uses k; UPD applies reflector k to cols [j,jend) */
    int      crit;            /* 1 = on the critical path (main executes) */
    int      npred;
    volatile int pred;        /* remaining predecessors (atomic countdown) */
    int      soff, scnt;      /* CSR successor slice */
} CTask;

static int g_ccq_chunk = 8;   /* columns per worker UPD task (amortizes sync) */
static int g_ccq_main_hvx = 0;/* 0 = main releases its context: workers get ALL
                               *     units, main does the critical column on
                               *     scalar.  Measured ~1.33x faster than main
                               *     holding a context here (the per-step
                               *     critical column is too small to justify
                               *     reserving a whole vector unit for main).
                               * 1 = main holds a context (workers get units-1). */

typedef struct {
    Matrix* A; int m, n, nt;
    CTask*  task; int ntask;
    int*    succ;             /* CSR successor ids */
    float*  vstore;           /* reflector storage: n vectors of length m */
    float*  taustore;         /* n taus */
    /* LOCK-FREE worker ready-queue: a bounded MPMC ring.  Producers (main +
     * workers, via ccq_complete) claim a unique slot with one atomic inc of
     * wpush; consumers (workers) claim with one atomic inc of wpop.  Each slot
     * has a `ready` flag so a consumer that claims a slot before the producer
     * has finished writing it spins briefly on the flag.  The counting
     * semaphore wavail is used ONLY for blocking when the queue is empty
     * (so idle workers sleep instead of busy-spinning), not for mutual
     * exclusion — there is no mutex on the hot path. wcap is a power of two. */
    int*    wq;                 /* slot payload (task id) */
    volatile int* wflag;        /* per-slot ready flag (0/1) */
    int     wcap, wmask;        /* capacity (pow2) and mask */
    volatile unsigned wpush, wpop;
    qurt_sem_t wavail;
    /* main's critical-path ready mailbox (single consumer = main) */
    qurt_sem_t mavail; volatile int mready;   /* next critical task id, -1 none */
    volatile unsigned remaining;
    volatile int done;
    volatile unsigned hvx_claim; int hvx_worker_units;
} CCQ;

static CCQ ccq;
static qurt_thread_t ccq_tids[MAX_THREADS];

static inline float* CCQ_V(int k) { return ccq.vstore + (size_t)k * ccq.m; }

/* ---- lock-free worker ready-queue (bounded MPMC ring) ---- */
static void ccq_worker_push(int id) {
    unsigned pos = qurt_atomic_inc_return(&ccq.wpush) - 1;  /* unique slot */
    int s = (int)(pos & (unsigned)ccq.wmask);
    ccq.wq[s] = id;
    __asm__ __volatile__("" ::: "memory");                 /* publish payload */
    ccq.wflag[s] = 1;                                       /* mark ready */
    qurt_sem_up(&ccq.wavail);                               /* wake a sleeper */
}
static int ccq_worker_pop(void) {
    /* The semaphore count == (#real items pushed) - (#consumed) until the run
     * ends; at completion the driver sets done=1 and posts nt extra tokens.  A
     * worker that wakes after done with no real item left exits.  Invariant:
     * a consumer only claims a wpop ticket when it KNOWS a real item exists
     * (it passed sem_down for a real push, not an exit token), so wpop can
     * never overtake wpush.  We distinguish the two by checking done+frontier
     * AFTER the wake but BEFORE claiming a ticket. */
    qurt_sem_down(&ccq.wavail);            /* block while empty (no busy-spin) */
    /* If everything is retired and we were woken to exit, leave. */
    if (ccq.done && ccq.wpop >= ccq.wpush) return -1;
    unsigned pop = qurt_atomic_inc_return(&ccq.wpop) - 1;   /* real item ticket */
    int s = (int)(pop & (unsigned)ccq.wmask);
    while (!ccq.wflag[s]) { /* tiny: producer between inc and flag store */ }
    int id = ccq.wq[s];
    __asm__ __volatile__("" ::: "memory");
    ccq.wflag[s] = 0;
    return id;
}
/* route a now-ready task to main (critical) or the worker queue (non-critical).
 * mready is a single-slot mailbox written by at most one producer at a time
 * (only one critical task is ever ready at once, by DAG construction), so a
 * plain store + semaphore suffices — no lock. */
static void ccq_make_ready(int id) {
    if (ccq.task[id].crit) {
        ccq.mready = id;
        __asm__ __volatile__("" ::: "memory");
        qurt_sem_up(&ccq.mavail);
    } else {
        ccq_worker_push(id);
    }
}

/* execute one UPD task (applies reflector k to all columns [j,jend)) */
static inline void ccq_exec_upd(int id, int use_hvx) {
    const CTask* t = &ccq.task[id];
    const float* v = CCQ_V(t->k); float tau = ccq.taustore[t->k];
    for (int jj = t->j; jj < t->jend; jj++) {
        if (use_hvx) apply_col_hvx(ccq.A, v, tau, jj, ccq.m);
        else         apply_col_scalar(ccq.A, v, tau, jj, ccq.m);
    }
}

/* on completion of task id: decrement successors, route those that hit 0, and
 * signal global done when all tasks are retired. */
static void ccq_complete(int id) {
    const CTask* t = &ccq.task[id];
    for (int e = 0; e < t->scnt; e++) {
        int s = ccq.succ[t->soff + e];
        int left = (int)qurt_atomic_add_return((unsigned*)&ccq.task[s].pred, -1u);
        if (left == 0) ccq_make_ready(s);
    }
    unsigned rem = qurt_atomic_add_return((unsigned*)&ccq.remaining, -1u);
    if (rem == 0) {
        ccq.done = 1;
        for (int x = 0; x < ccq.nt; x++) qurt_sem_up(&ccq.wavail); /* wake workers to exit */
        qurt_sem_up(&ccq.mavail);                                  /* unblock main if waiting */
    }
}

static void ccq_worker(void* arg) {
    int tid = (int)(intptr_t)arg; (void)tid;
    int got_hvx = 0;
    while (1) {
        int id = ccq_worker_pop();
        if (id < 0) break;
        if (got_hvx == 0) {
            unsigned slot = qurt_atomic_inc_return(&ccq.hvx_claim);
            if ((int)slot <= ccq.hvx_worker_units &&
                qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK) got_hvx = 1;
            else got_hvx = -1;
        }
        ccq_exec_upd(id, got_hvx == 1);
        ccq_complete(id);
    }
    if (got_hvx == 1) qurt_hvx_unlock();
    qurt_thread_exit(0);
}

/* ---- DAG construction (sequential-order trace + per-column last writer) ---- */
static int* ccq_esrc; static int* ccq_edst; static int ccq_ne;
static void ccq_add_edge(int s, int d) {
    if (s < 0) return;
    ccq_esrc[ccq_ne] = s; ccq_edst[ccq_ne] = d; ccq_ne++;
    ccq.task[d].npred++;
}
static void ccq_build_dag(int n) {
    int C = g_ccq_chunk; if (C < 1) C = 1;
    /* worker UPD tasks are C-aligned column blocks: block b = cols [b*C,b*C+C).
     * A column always lies in the same block across steps, so a per-block
     * last-writer is an exact dependency.  Per step k: REFL(k) (1), the critical
     * column k+1 (1, crit), and one chunk task per block intersecting [k+2,n).
     * Loose upper bound on task count: */
    int nblk = (n + C - 1) / C;
    int tmax = n + n /*crit cols*/ + n * (nblk + 1) + 8;
    /* edge bound: <=2 per task, plus up to C extra per dirty-block fallback
     * (once per step) -> 3*tmax + n*C is safe. */
    size_t emax = (size_t)3 * tmax + (size_t)n * C + 16;
    ccq.task = (CTask*)calloc(tmax, sizeof(CTask));
    ccq_esrc = (int*)malloc(emax * sizeof(int));
    ccq_edst = (int*)malloc(emax * sizeof(int));
    int* lw  = (int*)malloc((size_t)n * sizeof(int));    /* last writer per column */
    int* lwb = (int*)malloc((size_t)nblk * sizeof(int)); /* last writer per block  */
    for (int j = 0; j < n; j++) lw[j] = -1;
    for (int b = 0; b < nblk; b++) lwb[b] = -1;
    int nt_ = 0; ccq_ne = 0;

    for (int k = 0; k < n; k++) {
        /* REFL(k): reads column k; depends on column k's last writer */
        int rid = nt_++;
        ccq.task[rid].kind = CK_REFL; ccq.task[rid].k = k; ccq.task[rid].crit = 1;
        ccq_add_edge(lw[k], rid);
        lw[k] = rid;                              /* REFL writes column k diag */

        /* critical column k+1 (single, main executes) */
        if (k + 1 < n) {
            int cid = nt_++;
            ccq.task[cid].kind = CK_UPD; ccq.task[cid].k = k;
            ccq.task[cid].j = k + 1; ccq.task[cid].jend = k + 2; ccq.task[cid].crit = 1;
            ccq_add_edge(rid, cid);
            ccq_add_edge(lw[k + 1], cid);
            lw[k + 1] = cid;
            int b = (k + 1) / C; lwb[b] = -2;     /* mark block dirty (mixed writer) */
        }

        /* non-critical columns [k+2, n) as C-aligned chunk tasks */
        int start = k + 2;
        for (int j = start; j < n; ) {
            int b = j / C;
            int bend = (b + 1) * C; if (bend > n) bend = n;   /* block end */
            int je = bend;                                     /* this task covers [j,je) */
            int uid = nt_++;
            ccq.task[uid].kind = CK_UPD; ccq.task[uid].k = k;
            ccq.task[uid].j = j; ccq.task[uid].jend = je; ccq.task[uid].crit = 0;
            ccq_add_edge(rid, uid);
            /* predecessor = last writer of this block; if the block was last
             * touched column-wise (dirty, -2), fall back to per-column writers
             * so we never miss an edge. */
            if (lwb[b] == -2) {
                for (int c = j; c < je; c++) ccq_add_edge(lw[c], uid);
            } else {
                ccq_add_edge(lwb[b], uid);
            }
            for (int c = j; c < je; c++) lw[c] = uid;
            lwb[b] = uid;
            j = je;
        }
    }
    ccq.ntask = nt_;

    /* CSR successor lists */
    for (int x = 0; x < nt_; x++) ccq.task[x].scnt = 0;
    for (int e = 0; e < ccq_ne; e++) ccq.task[ccq_esrc[e]].scnt++;
    int off = 0;
    for (int x = 0; x < nt_; x++) { ccq.task[x].soff = off; off += ccq.task[x].scnt; }
    ccq.succ = (int*)malloc((size_t)(ccq_ne ? ccq_ne : 1) * sizeof(int));
    int* cur = (int*)malloc((size_t)nt_ * sizeof(int));
    for (int x = 0; x < nt_; x++) cur[x] = ccq.task[x].soff;
    for (int e = 0; e < ccq_ne; e++) ccq.succ[cur[ccq_esrc[e]]++] = ccq_edst[e];
    for (int x = 0; x < nt_; x++) ccq.task[x].pred = ccq.task[x].npred;
    free(lw); free(lwb); free(cur); free(ccq_esrc); free(ccq_edst);
}

static void qr_dag_cc_queue(Matrix* A, int nt) {
    int m = A->m, n = A->n;
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    if (nt < 1) nt = 1;
    if (n == 0) return;

    ccq.A = A; ccq.m = m; ccq.n = n; ccq.nt = nt;
    ccq.vstore   = (float*)memalign(128, (size_t)n * m * sizeof(float));
    ccq.taustore = (float*)memalign(128, (size_t)n * sizeof(float));
    ccq_build_dag(n);
    ccq.remaining = (unsigned)ccq.ntask;
    ccq.done = 0; ccq.hvx_claim = 0;
    /* If main holds a context, workers get units-1; else workers get all. */
    ccq.hvx_worker_units = g_ccq_main_hvx ? (g_hvx_units_128b - 1) : g_hvx_units_128b;
    if (ccq.hvx_worker_units < 1) ccq.hvx_worker_units = 1;
    /* lock-free ring: capacity = next power of two > ntask (slots are reused as
     * tasks retire, but ntask+slack guarantees no live-slot overwrite). */
    int cap = 1; while (cap <= ccq.ntask + nt) cap <<= 1;
    ccq.wcap = cap; ccq.wmask = cap - 1;
    ccq.wq    = (int*)malloc((size_t)cap * sizeof(int));
    ccq.wflag = (volatile int*)calloc((size_t)cap, sizeof(int));
    ccq.wpush = ccq.wpop = 0;
    ccq.mready = -1;
    qurt_sem_init_val(&ccq.wavail, 0);
    qurt_sem_init_val(&ccq.mavail, 0);

    /* main holds a context only if configured to; otherwise it leaves all
     * contexts to the workers and runs its critical column on scalar. */
    int main_hvx = g_ccq_main_hvx ? (qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK) : 0;

    qurt_thread_attr_t attr;
    for (int t = 0; t < nt; t++) {
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
        qurt_thread_attr_set_priority(&attr, 100);
        qurt_thread_create(&ccq_tids[t], &attr, ccq_worker, (void*)(intptr_t)t);
    }

    /* REFL(0) (task id 0) is the only initially-ready task; it is critical, so
     * main starts with it directly, then receives each next critical task via
     * the mavail semaphore. */
    int next = 0, have_next = 1;
    while (!ccq.done) {
        int id;
        if (have_next) { id = next; have_next = 0; }
        else {
            qurt_sem_down(&ccq.mavail);
            if (ccq.done) break;
            /* single-producer-at-a-time mailbox: at most one critical task is
             * ready at once (DAG critical path is a chain), so a plain read
             * with reset is safe — no lock. */
            id = ccq.mready; ccq.mready = -1;
            __asm__ __volatile__("" ::: "memory");
            if (id < 0) continue;
        }
        CTask* t = &ccq.task[id];
        if (t->kind == CK_REFL) {
            int k = t->k;
            float alpha, tau;
            int ok = compute_reflector(A, m, k, CCQ_V(k), &alpha, &tau);
            ccq.taustore[k] = ok ? tau : 0.f;
            if (ok) {
                MAT(*A, k, k) = alpha;
                for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
            } else {
                for (int i = 0; i < m; i++) CCQ_V(k)[i] = 0.f;  /* tau 0 -> UPD no-op */
            }
        } else { /* critical UPD(k,k+1) on main's held context */
            ccq_exec_upd(id, main_hvx);
        }
        ccq_complete(id);
    }

    if (main_hvx) qurt_hvx_unlock();
    for (int t = 0; t < nt; t++) { int s; qurt_thread_join(ccq_tids[t], &s); }

    qurt_sem_destroy(&ccq.wavail); qurt_sem_destroy(&ccq.mavail);
    free(ccq.vstore); free(ccq.taustore); free(ccq.task); free(ccq.succ);
    free(ccq.wq); free((void*)ccq.wflag);
}

/* =====================================================================
 *  main
 * ===================================================================== */
int main(void) {
    FILE* fp = fopen("./../inparams.txt", "r");
    if (!fp) { printf("Error: cannot open ./../inparams.txt\n"); return 1; }
    int m = 0, n = 0, nt = 1;
    int chunk = DEFAULT_CHUNK, sched = SCHED_ASYNC_PRIO, nb = DEFAULT_NB;
    int nread = fscanf(fp, "%d %d %d %d %d %d", &m, &n, &nt, &chunk, &sched, &nb);
    fclose(fp);
    if (nread < 3) { printf("Error: inparams.txt needs m n nt\n"); return 1; }
    if (m % HVX_FLOAT_LANES != 0) {
        printf("Error: m must be a multiple of %d\n", HVX_FLOAT_LANES);
        return 1;
    }
    if (n > MAX_N) { printf("Error: n exceeds MAX_N=%d\n", MAX_N); return 1; }
    (void)nb;
    g_hvx_units_128b = hvx_units_128b();

    const char* sched_name = (sched == SCHED_BARRIER)        ? "barrier" :
                             (sched == SCHED_ASYNC)          ? "async"   :
                             (sched == SCHED_ASYNC_PRIO_CC)  ? "async_prio_cc" :
                             (sched == SCHED_CC_QUEUE)       ? "cc_queue" :
                                                               "async_prio";
    printf("=== %s [%s] ===\n", VARIANT_NAME, sched_name);
    printf("Matrix: %d x %d  |  Threads: %d  |  Chunk: %d  |  HVX units(cfg): 0x%x\n\n",
           m, n, nt, chunk, qurt_hvx_get_units());

    Matrix A_orig = create_matrix(m, n);
    fill_test_matrix(&A_orig);

    Matrix A_cpu = copy_matrix(&A_orig);
    printf("[ref] CPU scalar Householder        ... ");
    uint64_t t0 = get_cycles();
    qr_cpu_householder(&A_cpu);
    uint64_t t1 = get_cycles();
    printf("%llu cycles\n", (unsigned long long)(t1 - t0));

    /* cc_queue spawns its own drain-loop workers; all other modes use the
     * persistent signal-pool. */
    int use_signal_pool = (sched != SCHED_CC_QUEUE);
    if (use_signal_pool) pool_init(nt);
    Matrix A_alg = copy_matrix(&A_orig);
    printf("[alg] %s [%s] ... ", VARIANT_NAME, sched_name);
    uint64_t t2 = get_cycles();
    if      (sched == SCHED_BARRIER)       qr_dag_barrier(&A_alg, chunk);
    else if (sched == SCHED_ASYNC)         qr_dag_async(&A_alg, chunk, 0);  /* no priority */
    else if (sched == SCHED_ASYNC_PRIO_CC) qr_dag_async_prio_cc(&A_alg, chunk); /* concurrent crit col */
    else if (sched == SCHED_CC_QUEUE)      { g_ccq_chunk = (chunk > 0 ? chunk : 8);
                                             /* nb==1 forces main to HOLD a context; any other
                                              * value uses the default (release = faster here). */
                                             g_ccq_main_hvx = (nb == 1);
                                             qr_dag_cc_queue(&A_alg, nt); }   /* concurrent queue */
    else                                   qr_dag_async(&A_alg, chunk, 1);  /* async_prio */
    uint64_t t3 = get_cycles();
    if (use_signal_pool) pool_shutdown();

    float err  = verify_R_against(&A_cpu, &A_alg);
    float sub  = subdiag_residual(&A_alg);
    printf("%llu cycles  |  max |R|-err: %.3e %s  |  sub-diag: %.3e %s\n",
           (unsigned long long)(t3 - t2),
           err, err < 1e-3f ? "OK" : "FAIL",
           sub, sub < 1e-5f ? "OK" : "FAIL");

    free_matrix(&A_orig); free_matrix(&A_cpu); free_matrix(&A_alg);
    return 0;
}
