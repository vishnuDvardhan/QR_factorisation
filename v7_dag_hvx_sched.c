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

enum { SCHED_ASYNC_PRIO = 0, SCHED_BARRIER = 1 };

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
    unsigned int          hh_counter;     /* atomic ascending-column claim */
    int                   hh_chunk_size;

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

/* Worker body: claim columns in ascending (priority) order; grab an HVX
 * context on the first claimed chunk via try_lock; publish col_done[j]. */
static void pool_do_col_hh(void) {
    Matrix* A = g_pool.hh_A;
    int m     = A->m;
    int chunk = g_pool.hh_chunk_size;
    const float* v = g_pool.hh_v;
    float tau = g_pool.hh_tau;
    int got_hvx = 0;   /* 0 = not yet claimed, 1 = holds context, -1 = scalar */

    while (1) {
        unsigned int ci = qurt_atomic_inc_return(&g_pool.hh_counter) - 1;
        int col_base    = g_pool.hh_col_start + (int)ci * chunk;
        if (col_base >= g_pool.hh_col_end) break;
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
 * NOT block.  Returns 1 if workers were dispatched, 0 if there was no work. */
static int pool_dispatch_nonblocking(Matrix* A, const float* v, float tau,
                                     int col_start, int col_end, int chunk_size) {
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

/* SCHED_ASYNC_PRIO: critical-path pivot overlap WITHOUT busy-waiting.
 *
 * Key lesson from the first cut: spinning the main thread on col_done[k+1]
 * is catastrophic on this in-order SMT DSP (it steals a HW-thread slot and
 * memory bandwidth from the HVX workers — measured ~5x slowdown).  Instead,
 * the main thread *executes the critical column k+1 itself* (the highest-
 * priority task) on a reserved HVX context, then computes reflector k+1,
 * both overlapping the workers that handle the lower-priority columns
 * k+2..n-1.  No spin: the main thread is always doing useful work or asleep.
 *
 * This is exactly "allocate an HVX vector unit to the highest-priority task":
 * one of the (~4) contexts is reserved for the main thread's critical-path
 * column; the rest are handed to workers via try_lock in ascending order. */
static void qr_dag_async_prio(Matrix* A, int chunk_size) {
    int m = A->m, n = A->n;
    float* vbuf[2];
    float  tau[2], alpha[2];
    int    valid[2];
    vbuf[0] = (float*)memalign(128, (size_t)m * sizeof(float));
    vbuf[1] = (float*)memalign(128, (size_t)m * sizeof(float));
    int cur = 0;

    /* Workers get ALL HVX contexts.  The main thread does NOT hold a context
     * for its lifetime (that starved a worker at nt == #contexts).  Instead it
     * locks a context *briefly* for the single critical column k+1 — between
     * the previous step's join and this step's dispatch, when no worker is
     * active — then releases it before the bulk dispatch + wait. */
    g_pool.hvx_worker_units = g_hvx_units_128b;

    valid[cur] = compute_reflector(A, m, 0, vbuf[cur], &alpha[cur], &tau[cur]);

    for (int k = 0; k < n; k++) {
        int nxt = cur ^ 1;
        int dispatched = 0;

        /* Critical-path task T(k,k+1): update column k+1 first, on HVX, while
         * no workers hold a context (prev step already joined). */
        if (valid[cur] && k + 1 < n) {
            if (qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK) {
                apply_col_hvx(A, vbuf[cur], tau[cur], k + 1, m);
                qurt_hvx_unlock();
            } else {
                apply_col_scalar(A, vbuf[cur], tau[cur], k + 1, m);
            }
        }

        /* Workers take the lower-priority columns [k+2, n) with all contexts. */
        if (valid[cur])
            dispatched = pool_dispatch_nonblocking(A, vbuf[cur], tau[cur],
                                                   k + 2, n, chunk_size);

        /* Reflector k+1 is pure scalar — overlaps the workers' HVX updates. */
        if (k + 1 < n)
            valid[nxt] = compute_reflector(A, m, k + 1, vbuf[nxt],
                                           &alpha[nxt], &tau[nxt]);

        if (dispatched) pool_wait_done();    /* join workers for cols k+2..n-1 */

        if (valid[cur]) {
            MAT(*A, k, k) = alpha[cur];
            for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
        }
        cur = nxt;
    }

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
        int dispatched = pool_dispatch_nonblocking(A, v, tau, k + 1, n, chunk_size);
        if (dispatched) pool_wait_done();
        MAT(*A, k, k) = alpha;
        for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
    }
    free(v);
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

    const char* sched_name = (sched == SCHED_BARRIER) ? "barrier" : "async_prio";
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

    pool_init(nt);
    Matrix A_alg = copy_matrix(&A_orig);
    printf("[alg] %s [%s] ... ", VARIANT_NAME, sched_name);
    uint64_t t2 = get_cycles();
    if (sched == SCHED_BARRIER) qr_dag_barrier(&A_alg, chunk);
    else                        qr_dag_async_prio(&A_alg, chunk);
    uint64_t t3 = get_cycles();
    pool_shutdown();

    float err  = verify_R_against(&A_cpu, &A_alg);
    float sub  = subdiag_residual(&A_alg);
    printf("%llu cycles  |  max |R|-err: %.3e %s  |  sub-diag: %.3e %s\n",
           (unsigned long long)(t3 - t2),
           err, err < 1e-3f ? "OK" : "FAIL",
           sub, sub < 1e-5f ? "OK" : "FAIL");

    free_matrix(&A_orig); free_matrix(&A_cpu); free_matrix(&A_alg);
    return 0;
}
