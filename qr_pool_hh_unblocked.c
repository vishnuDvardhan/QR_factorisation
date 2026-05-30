/* =====================================================================
 *  Persistent thread-pool variant of HH unblocked QR (column-major HVX).
 *
 *  Goal: eliminate the per-Householder qurt_thread_create + join that
 *  dominated cycles in the previous HVX-MT baseline (~30-40K cyc/spawn
 *  × n Householders × nt threads).
 *
 *  Design:
 *    - At program start, spawn `nt` worker threads ONCE.
 *    - Each worker sits on its OWN qurt_signal_t (QuRT signals are
 *      single-waiter — sharing one signal among multiple waiters
 *      triggers a kernel assertion).
 *    - For each Householder, main sets a "go" bit on every worker's
 *      signal, waits on a master "done" signal that's signalled when
 *      all workers atomically decrement a done-counter to zero.
 *    - Workers acquire HVX per job (not per lifetime) so other code
 *      paths can still use HVX between dispatches.
 *
 *  Compared to per-call thread_create/join (~30K cyc each), the pool
 *  pays only ~1K cyc per dispatch (signal set + signal wait), plus
 *  ~5K cyc for HVX lock — a net ~5-10× reduction in fixed overhead per
 *  Householder.
 *
 *  Same input matrix as other variants (deterministic), same output
 *  format so the experiment notebook can parse it.
 *
 *  inparams.txt format: m n nt [chunk_size] [use_static] [nb]
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

QURT_DECLARE_STATIC_HEAP(16777216 * 2);
unsigned int heapSize = 16777216;

#define HVX_FLOAT_LANES 32
#define MAX_THREADS     8
#define STACK_SIZE      (16 * 1024 * 2)
#define DEFAULT_CHUNK   4
#define DEFAULT_NB      8

static const char* VARIANT_NAME = "pool_hh_unblocked";

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
    for (int j = 0; j < ref->n; j++) {
        for (int i = 0; i <= j && i < ref->m; i++) {
            float d = fabsf(fabsf(MAT(*ref, i, j)) - fabsf(MAT(*alg, i, j)));
            if (d > max_err) max_err = d;
        }
    }
    return max_err;
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

/* =====================================================================
 *  Persistent thread pool
 * ===================================================================== */

typedef enum {
    POOL_JOB_NONE = 0,
    POOL_JOB_COL_HH,
    POOL_JOB_SHUTDOWN,
} PoolJobKind;

typedef struct {
    int                   nt;
    volatile PoolJobKind  kind;

    /* Per-worker "go" signal (QuRT signals are single-waiter). */
    qurt_signal_t         go[MAX_THREADS];
    /* Single signal that only the main thread waits on. */
    qurt_signal_t         done_master;
    /* Workers atomic-dec this. When it hits 0, last worker sets done_master. */
    volatile unsigned int done_count;

    /* COL_HH job descriptor */
    Matrix*               hh_A;
    const float*          hh_v;
    float                 hh_tau;
    int                   hh_col_start;
    int                   hh_col_end;
    unsigned int          hh_counter;     /* atomic work-claim */
    int                   hh_chunk_size;
} ThreadPool;

static ThreadPool   g_pool;
static qurt_thread_t g_pool_tids[MAX_THREADS];
static int          g_pool_running = 0;

/* Worker job body — same kernel as the simple per-spawn version. */
static void pool_do_col_hh(void) {
    int m     = g_pool.hh_A->m;
    int chunk = g_pool.hh_chunk_size;
    while (1) {
        unsigned int ci = qurt_atomic_inc_return(&g_pool.hh_counter) - 1;
        int col_base    = g_pool.hh_col_start + (int)ci * chunk;
        if (col_base >= g_pool.hh_col_end) break;
        int col_end_c   = col_base + chunk;
        if (col_end_c > g_pool.hh_col_end) col_end_c = g_pool.hh_col_end;

        for (int j = col_base; j < col_end_c; j++) {
            /* Dot product accumulated in qf32 */
            HVX_Vector vSum = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&MAT(*g_pool.hh_A, i, j);
                HVX_Vector vv = *(HVX_VectorU*)&g_pool.hh_v[i];
                vSum = Q6_Vqf32_vadd_Vqf32Vqf32(vSum,
                          Q6_Vqf32_vmpy_VsfVsf(va, vv));
            }
            float temp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
            *(HVX_VectorU*)temp = Q6_Vsf_equals_Vqf32(vSum);
            float dot = 0.f;
            for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) dot += temp[lane];

            float w = g_pool.hh_tau * dot;
            HVX_Vector vw = Q6_V_vsplat_R(*(uint32_t*)&w);
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&MAT(*g_pool.hh_A, i, j);
                HVX_Vector vv = *(HVX_VectorU*)&g_pool.hh_v[i];
                HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, Q6_V_vzero());
                HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
                *(HVX_VectorU*)&MAT(*g_pool.hh_A, i, j) =
                    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp));
            }
        }
    }
}

static void pool_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    qurt_signal_t* my_go = &g_pool.go[tid];

    while (1) {
        /* Wait for bit 0 on my own go signal. */
        qurt_signal_wait(my_go, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
        qurt_signal_clear(my_go, 1u);

        if (g_pool.kind == POOL_JOB_SHUTDOWN) break;

        /* HVX lock per job (so non-pool HVX users can run between
         * dispatches). */
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
            switch (g_pool.kind) {
                case POOL_JOB_COL_HH: pool_do_col_hh(); break;
                default: break;
            }
            qurt_hvx_unlock();
        }

        /* Last worker out sets done_master so main wakes. */
        unsigned int remaining = qurt_atomic_add_return(&g_pool.done_count, -1u);
        if (remaining == 0) {
            qurt_signal_set(&g_pool.done_master, 1u);
        }
    }
    qurt_thread_exit(0);
}

static void pool_init(int nt) {
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    if (nt < 1) nt = 1;
    g_pool.nt   = nt;
    g_pool.kind = POOL_JOB_NONE;

    qurt_signal_init(&g_pool.done_master);
    for (int t = 0; t < nt; t++) qurt_signal_init(&g_pool.go[t]);

    qurt_thread_attr_t attr;
    for (int t = 0; t < nt; t++) {
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
        qurt_thread_attr_set_priority(&attr, 100);
        qurt_thread_create(&g_pool_tids[t], &attr, pool_worker,
                           (void*)(intptr_t)t);
    }
    g_pool_running = 1;
}

static void pool_shutdown(void) {
    if (!g_pool_running) return;
    g_pool.kind = POOL_JOB_SHUTDOWN;
    g_pool.done_count = g_pool.nt;
    qurt_signal_clear(&g_pool.done_master, 1u);
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_set(&g_pool.go[t], 1u);
    /* Workers exit immediately on SHUTDOWN; they do NOT signal done_master,
     * so just join them directly. */
    for (int t = 0; t < g_pool.nt; t++) {
        int s; qurt_thread_join(g_pool_tids[t], &s);
    }
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_destroy(&g_pool.go[t]);
    qurt_signal_destroy(&g_pool.done_master);
    g_pool_running = 0;
}

static void pool_dispatch_col_hh(Matrix* A, const float* v, float tau,
                                 int col_start, int col_end, int chunk_size) {
    if (col_end <= col_start || !g_pool_running) return;
    if (chunk_size < 1) chunk_size = 1;
    g_pool.kind          = POOL_JOB_COL_HH;
    g_pool.hh_A          = A;
    g_pool.hh_v          = v;
    g_pool.hh_tau        = tau;
    g_pool.hh_col_start  = col_start;
    g_pool.hh_col_end    = col_end;
    g_pool.hh_counter    = 0;
    g_pool.hh_chunk_size = chunk_size;
    g_pool.done_count    = g_pool.nt;

    qurt_signal_clear(&g_pool.done_master, 1u);
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_set(&g_pool.go[t], 1u);
    qurt_signal_wait(&g_pool.done_master, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
}

/* =====================================================================
 *  HH unblocked driver (uses the pool for trailing-column updates)
 * ===================================================================== */

static void qr_hh_unblocked_pool(Matrix* A, int chunk_size) {
    int m = A->m, n = A->n;
    float* v = (float*)memalign(128, (size_t)m * sizeof(float));
    for (int k = 0; k < n; k++) {
        float nxsq = 0.f;
        for (int i = k; i < m; i++) { float val = MAT(*A, i, k); nxsq += val*val; }
        float nx = sqrtf(nxsq); if (nx < 1e-7f) continue;
        float alpha = -(MAT(*A, k, k) >= 0.f ? 1.f : -1.f) * nx;
        for (int i = 0; i < k; i++) v[i] = 0.f;
        for (int i = k; i < m; i++) v[i] = MAT(*A, i, k);
        v[k] -= alpha;
        float nvsq = 0.f;
        for (int i = k; i < m; i++) nvsq += v[i]*v[i];
        if (nvsq < 1e-10f) continue;
        float tau = 2.f / nvsq;

        pool_dispatch_col_hh(A, v, tau, k + 1, n, chunk_size);

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
    int chunk = DEFAULT_CHUNK, use_static = 0, nb = DEFAULT_NB;
    int nread = fscanf(fp, "%d %d %d %d %d %d",
                       &m, &n, &nt, &chunk, &use_static, &nb);
    fclose(fp);
    if (nread < 3) { printf("Error: inparams.txt needs m n nt\n"); return 1; }
    if (m % HVX_FLOAT_LANES != 0) {
        printf("Error: m must be a multiple of %d\n", HVX_FLOAT_LANES);
        return 1;
    }
    (void)use_static; (void)nb;

    printf("=== %s ===\n", VARIANT_NAME);
    printf("Matrix: %d x %d  |  Threads: %d  |  Chunk: %d  |  nb: %d\n\n",
           m, n, nt, chunk, nb);

    Matrix A_orig = create_matrix(m, n);
    fill_test_matrix(&A_orig);

    /* CPU reference */
    Matrix A_cpu = copy_matrix(&A_orig);
    printf("[ref] CPU scalar Householder        ... ");
    uint64_t t0 = get_cycles();
    qr_cpu_householder(&A_cpu);
    uint64_t t1 = get_cycles();
    printf("%llu cycles\n", (unsigned long long)(t1 - t0));

    /* Pool-based HH unblocked */
    pool_init(nt);
    Matrix A_alg = copy_matrix(&A_orig);
    printf("[alg] %s             ... ", VARIANT_NAME);
    uint64_t t2 = get_cycles();
    qr_hh_unblocked_pool(&A_alg, chunk);
    uint64_t t3 = get_cycles();
    pool_shutdown();
    float err = verify_R_against(&A_cpu, &A_alg);
    printf("%llu cycles  |  max |R|-err: %.3e  %s\n",
           (unsigned long long)(t3 - t2), err,
           err < 1e-3f ? "OK" : "FAIL");

    free_matrix(&A_orig);
    free_matrix(&A_cpu);
    free_matrix(&A_alg);
    return 0;
}
