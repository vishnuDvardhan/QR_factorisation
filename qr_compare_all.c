/* =====================================================================
 *  QR algorithm comparison — Hexagon HVX + QuRT MT
 *
 *  One binary runs the following on the SAME input matrix:
 *    [1] CPU reference scalar Householder QR        (correctness oracle)
 *    [2] HH unblocked HVX-MT                       (current baseline)
 *    [3] HH WY-blocked HVX-MT                       (already tested)
 *    [4] MGS2 (Modified Gram-Schmidt re-ortho) HVX  (TBD)
 *    [5] Givens QR HVX-MT                          (TBD)
 *    [6] Tile QR (TSQR-style merge) HVX-MT          (TBD)
 *
 *  Each algorithm produces an upper-triangular R in-place. Verification
 *  compares |R_alg[i,j]| against |R_cpu[i,j]| on the upper triangle —
 *  this is layout-agnostic and ignores sign conventions (HH/MGS/Givens
 *  pick different signs on the diagonal).
 *
 *  Column-major throughout (matches LAPACK convention and our existing
 *  WY-blocked code).
 *
 *  inparams.txt format: m n nt chunk_size use_static nb
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

QURT_DECLARE_STATIC_HEAP(16777216 * 4);
unsigned int heapSize = 16777216;

#define HVX_FLOAT_LANES 32
#define MAX_THREADS     8
#define STACK_SIZE      (16 * 1024 * 2)
#define MAX_NB          32
#define DEFAULT_CHUNK   4
#define DEFAULT_NB      8

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

/* Compare upper triangle |R| values. Returns max abs diff. */
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

/* =====================================================================
 *  [1] CPU reference: scalar Householder QR
 * ===================================================================== */

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
 *  Persistent thread pool.
 *
 *  Workers spawn once at program start, wait on a signal, do one job
 *  (HH-col update OR WY-block update), signal done, sleep again.
 *  This eliminates the ~30K-cyc qurt_thread_create + join per dispatch
 *  that we previously paid for every Householder / every block update.
 *
 *  Workers acquire the HVX lock per job (not per lifetime) so the
 *  single-threaded algorithms (MGS2, Givens, Tile-QR leaves) can grab
 *  HVX themselves between pool dispatches.
 *
 *  Signal bit T identifies thread T (nt up to 8 = signal width plenty).
 *  Pool dispatch is called only from the main thread; workers never
 *  dispatch (no nested-pool race).
 * ===================================================================== */

typedef enum {
    POOL_JOB_NONE = 0,
    POOL_JOB_COL_HH,
    POOL_JOB_BLOCK_UPDATE,
    POOL_JOB_SHUTDOWN,
} PoolJobKind;

typedef struct {
    int                 nt;
    volatile PoolJobKind kind;
    qurt_signal_t       go[MAX_THREADS];   /* one per worker (single-waiter req) */
    qurt_signal_t       done_master;        /* only main waits on this */
    volatile unsigned int done_count;        /* workers atomic-dec; 0 → set done_master */

    /* COL_HH job */
    Matrix*             hh_A;
    const float*        hh_v;
    float               hh_tau;
    int                 hh_col_start;
    int                 hh_col_end;
    unsigned int        hh_counter;
    int                 hh_chunk_size;

    /* BLOCK_UPDATE job */
    Matrix*             bu_A;
    const float*        bu_V;
    const float*        bu_T;
    int                 bu_nb_actual;
    int                 bu_trail_start;
    int                 bu_trail_end;
    unsigned int        bu_counter;
    int                 bu_chunk_cols;
} ThreadPool;

static ThreadPool   g_pool;
static qurt_thread_t g_pool_tids[MAX_THREADS];
static int          g_pool_running = 0;

/* ---- Pool worker: COL_HH job body ---- */
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

/* ---- Pool worker: BLOCK_UPDATE job body ---- */
static void pool_do_block_update(void) {
    int m  = g_pool.bu_A->m;
    int nb = g_pool.bu_nb_actual;
    while (1) {
        unsigned int ci = qurt_atomic_inc_return(&g_pool.bu_counter) - 1;
        int j_base = g_pool.bu_trail_start + (int)ci * g_pool.bu_chunk_cols;
        if (j_base >= g_pool.bu_trail_end) break;
        int j_end_c = j_base + g_pool.bu_chunk_cols;
        if (j_end_c > g_pool.bu_trail_end) j_end_c = g_pool.bu_trail_end;

        for (int j = j_base; j < j_end_c; j++) {
            float* Aj = &MAT(*g_pool.bu_A, 0, j);
            HVX_Vector acc[MAX_NB];
            for (int l = 0; l < nb; l++) acc[l] = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&Aj[i];
                for (int l = 0; l < nb; l++) {
                    HVX_Vector vv = *(HVX_VectorU*)&g_pool.bu_V[(size_t)l * m + i];
                    HVX_Vector vp = Q6_Vqf32_vmpy_VsfVsf(vv, va);
                    acc[l] = Q6_Vqf32_vadd_Vqf32Vqf32(acc[l], vp);
                }
            }
            float z[MAX_NB], w[MAX_NB];
            float tmp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
            for (int l = 0; l < nb; l++) {
                *(HVX_VectorU*)tmp = Q6_Vsf_equals_Vqf32(acc[l]);
                float s = 0.f;
                for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) s += tmp[lane];
                z[l] = s;
            }
            /* w = T^T * z (lower-tri access of upper-tri T) */
            for (int i = 0; i < nb; i++) {
                float s = 0.f;
                for (int jj = 0; jj <= i; jj++)
                    s += g_pool.bu_T[(size_t)i * MAX_NB + jj] * z[jj];
                w[i] = s;
            }
            HVX_Vector vw_splat[MAX_NB];
            for (int l = 0; l < nb; l++)
                vw_splat[l] = Q6_V_vsplat_R(*(uint32_t*)&w[l]);
            HVX_Vector vzero = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va    = *(HVX_VectorU*)&Aj[i];
                HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, vzero);
                for (int l = 0; l < nb; l++) {
                    HVX_Vector vv = *(HVX_VectorU*)&g_pool.bu_V[(size_t)l * m + i];
                    HVX_Vector vp = Q6_Vqf32_vmpy_VsfVsf(vv, vw_splat[l]);
                    va_qf = Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp);
                }
                *(HVX_VectorU*)&Aj[i] = Q6_Vsf_equals_Vqf32(va_qf);
            }
        }
    }
}

/* ---- Pool worker thread main loop ---- */
static void pool_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    unsigned int my_bit = 1u << tid;

    while (1) {
        /* Wait for my go bit, then clear it. */
        qurt_signal_wait(&g_pool.go, my_bit, QURT_SIGNAL_ATTR_WAIT_ANY);
        qurt_signal_clear(&g_pool.go, my_bit);

        if (g_pool.kind == POOL_JOB_SHUTDOWN) break;

        /* Acquire HVX, do work, release. */
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
            switch (g_pool.kind) {
                case POOL_JOB_COL_HH:        pool_do_col_hh();        break;
                case POOL_JOB_BLOCK_UPDATE:  pool_do_block_update();  break;
                default: break;
            }
            qurt_hvx_unlock();
        }

        /* Signal done. */
        qurt_signal_set(&g_pool.done, my_bit);
    }
    qurt_thread_exit(0);
}

static void pool_init(int nt) {
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    if (nt < 1) nt = 1;
    g_pool.nt   = nt;
    g_pool.kind = POOL_JOB_NONE;
    qurt_signal_init(&g_pool.go);
    qurt_signal_init(&g_pool.done);

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
    unsigned int all_bits = (1u << g_pool.nt) - 1;
    qurt_signal_clear(&g_pool.done, all_bits);
    qurt_signal_set(&g_pool.go, all_bits);
    for (int t = 0; t < g_pool.nt; t++) {
        int s; qurt_thread_join(g_pool_tids[t], &s);
    }
    qurt_signal_destroy(&g_pool.go);
    qurt_signal_destroy(&g_pool.done);
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

    unsigned int all_bits = (1u << g_pool.nt) - 1;
    qurt_signal_clear(&g_pool.done, all_bits);
    qurt_signal_set(&g_pool.go, all_bits);
    qurt_signal_wait(&g_pool.done, all_bits, QURT_SIGNAL_ATTR_WAIT_ALL);
}

static void pool_dispatch_block_update(Matrix* A, const float* V, const float* T,
                                       int trail_start, int trail_end,
                                       int nb_actual, int chunk_cols) {
    if (trail_end <= trail_start || !g_pool_running) return;
    if (chunk_cols < 1) chunk_cols = 1;
    g_pool.kind           = POOL_JOB_BLOCK_UPDATE;
    g_pool.bu_A           = A;
    g_pool.bu_V           = V;
    g_pool.bu_T           = T;
    g_pool.bu_nb_actual   = nb_actual;
    g_pool.bu_trail_start = trail_start;
    g_pool.bu_trail_end   = trail_end;
    g_pool.bu_counter     = 0;
    g_pool.bu_chunk_cols  = chunk_cols;

    unsigned int all_bits = (1u << g_pool.nt) - 1;
    qurt_signal_clear(&g_pool.done, all_bits);
    qurt_signal_set(&g_pool.go, all_bits);
    qurt_signal_wait(&g_pool.done, all_bits, QURT_SIGNAL_ATTR_WAIT_ALL);
}

/* =====================================================================
 *  [2] HH unblocked HVX-MT (col-major)  — copied from qr_wy_blocked_mt.c
 *      with qf32 intrinsics.
 * ===================================================================== */

typedef struct {
    Matrix*       A;
    const float*  v;
    float         tau;
    int           col_start;
    int           col_end;
    unsigned int* counter;
    int           chunk_size;
} ColCtx;

static void col_worker_dyn(void* arg) {
    ColCtx* ctx = (ColCtx*)arg;
    int m     = ctx->A->m;
    int chunk = ctx->chunk_size;
    while (1) {
        unsigned int ci = qurt_atomic_inc_return(ctx->counter) - 1;
        int col_base    = ctx->col_start + (int)ci * chunk;
        if (col_base >= ctx->col_end) break;
        int col_end_c   = col_base + chunk;
        if (col_end_c > ctx->col_end) col_end_c = ctx->col_end;

        if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) continue;
        for (int j = col_base; j < col_end_c; j++) {
            HVX_Vector vSum = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&MAT(*ctx->A, i, j);
                HVX_Vector vv = *(HVX_VectorU*)&ctx->v[i];
                vSum = Q6_Vqf32_vadd_Vqf32Vqf32(vSum,
                          Q6_Vqf32_vmpy_VsfVsf(va, vv));
            }
            float temp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
            *(HVX_VectorU*)temp = Q6_Vsf_equals_Vqf32(vSum);
            float dot = 0.f;
            for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) dot += temp[lane];
            float w = ctx->tau * dot;
            HVX_Vector vw = Q6_V_vsplat_R(*(uint32_t*)&w);
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&MAT(*ctx->A, i, j);
                HVX_Vector vv = *(HVX_VectorU*)&ctx->v[i];
                HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, Q6_V_vzero());
                HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
                *(HVX_VectorU*)&MAT(*ctx->A, i, j) =
                    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp));
            }
        }
        qurt_hvx_unlock();
    }
    qurt_thread_exit(0);
}

static void run_col_workers(Matrix* A, const float* v, float tau,
                            int col_start, int col_end,
                            int nt, int chunk_size) {
    int total_cols = col_end - col_start;
    if (total_cols <= 0) return;
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    if (nt > total_cols)  nt = total_cols;
    if (nt < 1) nt = 1;
    if (chunk_size < 1) chunk_size = 1;

    ColCtx              ctx[MAX_THREADS];
    qurt_thread_t       tids[MAX_THREADS];
    qurt_thread_attr_t  attrs[MAX_THREADS];
    unsigned int        counter = 0;
    for (int t = 0; t < nt; t++) {
        ctx[t].A = A; ctx[t].v = v; ctx[t].tau = tau;
        ctx[t].col_start = col_start; ctx[t].col_end = col_end;
        ctx[t].counter = &counter; ctx[t].chunk_size = chunk_size;
        qurt_thread_attr_init(&attrs[t]);
        qurt_thread_attr_set_stack_addr(&attrs[t], thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attrs[t], STACK_SIZE);
        qurt_thread_attr_set_priority(&attrs[t], 100);
        int r = qurt_thread_create(&tids[t], &attrs[t], col_worker_dyn, &ctx[t]);
        if (r != QURT_EOK) {
            for (int i = 0; i < t; i++) { int s; qurt_thread_join(tids[i], &s); }
            return;
        }
    }
    for (int t = 0; t < nt; t++) { int s; qurt_thread_join(tids[t], &s); }
}

static void qr_hh_unblocked(Matrix* A, int nt, int chunk_size) {
    int m = A->m, n = A->n;
    (void)nt;  /* now driven by the persistent thread pool */
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
 *  [4] MGS2 — Modified Gram-Schmidt with one re-orthogonalization pass.
 *
 *  Column-major, in-place: A becomes Q during the loop; we accumulate
 *  R[j,k] on the side and copy back into A's upper triangle at the end.
 *  Single-threaded HVX: each dot/axpy/norm/scale is one m-element pass.
 *
 *  Per col k: 2(k) dots + 2(k) axpys + 1 norm + 1 scale.
 *  Total flops ≈ 4·m·n² — about 2× more than Householder, but with the
 *  simplest possible HVX kernels (no rank-1 reductions, no tau setup).
 * ===================================================================== */

static inline float hvx_dot(const float* x, const float* y, int m) {
    HVX_Vector vSum = Q6_V_vzero();
    for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector vx = *(HVX_VectorU*)&x[i];
        HVX_Vector vy = *(HVX_VectorU*)&y[i];
        vSum = Q6_Vqf32_vadd_Vqf32Vqf32(vSum, Q6_Vqf32_vmpy_VsfVsf(vx, vy));
    }
    float temp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
    *(HVX_VectorU*)temp = Q6_Vsf_equals_Vqf32(vSum);
    float s = 0.f;
    for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) s += temp[lane];
    return s;
}

/* y -= scale * x   (in-place, length m, m % HVX_FLOAT_LANES == 0) */
static inline void hvx_axpy_neg(float* y, float scale, const float* x, int m) {
    HVX_Vector vs = Q6_V_vsplat_R(*(uint32_t*)&scale);
    for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector vy = *(HVX_VectorU*)&y[i];
        HVX_Vector vx = *(HVX_VectorU*)&x[i];
        HVX_Vector vy_qf = Q6_Vqf32_vadd_VsfVsf(vy, Q6_V_vzero());
        HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vs, vx);
        *(HVX_VectorU*)&y[i] =
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vy_qf, vp));
    }
}

/* y *= scale */
static inline void hvx_scale(float* y, float scale, int m) {
    HVX_Vector vs = Q6_V_vsplat_R(*(uint32_t*)&scale);
    for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector vy = *(HVX_VectorU*)&y[i];
        HVX_Vector vp = Q6_Vqf32_vmpy_VsfVsf(vy, vs);
        *(HVX_VectorU*)&y[i] = Q6_Vsf_equals_Vqf32(vp);
    }
}

static void qr_mgs2(Matrix* A) {
    int m = A->m, n = A->n;

    /* R accumulator (col-major n×n) */
    float* R = (float*)memalign(128, (size_t)n * n * sizeof(float));
    memset(R, 0, (size_t)n * n * sizeof(float));

    /* Hold the HVX lock for the entire factorization. */
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) {
        printf("ERROR: MGS2 HVX lock failed\n");
        free(R);
        return;
    }

    for (int k = 0; k < n; k++) {
        float* a_k = &MAT(*A, 0, k);

        /* --- First orthogonalization pass --- */
        for (int j = 0; j < k; j++) {
            float* q_j = &MAT(*A, 0, j);
            float r1 = hvx_dot(q_j, a_k, m);
            hvx_axpy_neg(a_k, r1, q_j, m);
            R[(size_t)k * n + j] = r1;
        }

        /* --- Re-orthogonalization pass --- */
        for (int j = 0; j < k; j++) {
            float* q_j = &MAT(*A, 0, j);
            float r2 = hvx_dot(q_j, a_k, m);
            hvx_axpy_neg(a_k, r2, q_j, m);
            R[(size_t)k * n + j] += r2;
        }

        /* --- Normalize a_k → q_k --- */
        float nsq = hvx_dot(a_k, a_k, m);
        float norm = sqrtf(nsq);
        if (norm > 1e-7f) {
            hvx_scale(a_k, 1.f / norm, m);
        }
        R[(size_t)k * n + k] = norm;
    }

    qurt_hvx_unlock();

    /* Copy R back into A's upper triangle, zero below. */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) MAT(*A, i, j) = R[(size_t)j * n + i];
        for (int i = j + 1; i < m; i++) MAT(*A, i, j) = 0.f;
    }
    free(R);
}

/* =====================================================================
 *  [5] Givens rotations QR.
 *
 *  Each rotation zeros A[i,k] against A[i-1,k] with a 2x2 rotation
 *  applied across the row pair (i-1, i) of cols [k, n).
 *
 *  Col-major A is awkward for Givens because the row-pair rotation
 *  touches strided memory. We transpose col-major A into a row-major
 *  scratch, run Givens (HVX-vectorized across cols), then transpose
 *  back. Transpose is O(m·n) << Givens O(m·n²) so the overhead is
 *  negligible.
 *
 *  Cols are zero-padded to a multiple of HVX_FLOAT_LANES so the inner
 *  HVX block always fits without OOB; padding cols stay zero through
 *  any rotation (rotation of two zeros is two zeros) so they don't
 *  affect correctness.
 *
 *  Single-threaded HVX (matches MGS2 for fairness; threading can be
 *  added later if performance justifies it).
 * ===================================================================== */

static void qr_givens(Matrix* A_col) {
    int m = A_col->m, n = A_col->n;
    int n_pad = ((n + HVX_FLOAT_LANES - 1) / HVX_FLOAT_LANES) * HVX_FLOAT_LANES;

    float* S = (float*)memalign(128, (size_t)m * n_pad * sizeof(float));
    if (!S) { printf("ERROR: givens scratch alloc\n"); return; }
    memset(S, 0, (size_t)m * n_pad * sizeof(float));

    /* Transpose col-major A_col → row-major S */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            S[(size_t)i * n_pad + j] = MAT(*A_col, i, j);
        }
    }

    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) {
        printf("ERROR: Givens HVX lock failed\n");
        free(S);
        return;
    }

    for (int k = 0; k < n; k++) {
        for (int i = m - 1; i > k; i--) {
            float a = S[(size_t)(i - 1) * n_pad + k];
            float b = S[(size_t)i       * n_pad + k];
            float r2 = a * a + b * b;
            if (r2 < 1e-20f) continue;
            float r = sqrtf(r2);
            float c = a / r;
            float s = b / r;

            HVX_Vector vc = Q6_V_vsplat_R(*(uint32_t*)&c);
            HVX_Vector vs = Q6_V_vsplat_R(*(uint32_t*)&s);

            float* row_a = &S[(size_t)(i - 1) * n_pad];
            float* row_b = &S[(size_t)i       * n_pad];

            int j = k;
            for (; j + HVX_FLOAT_LANES <= n_pad; j += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&row_a[j];
                HVX_Vector vb = *(HVX_VectorU*)&row_b[j];

                /* new_a = c*va + s*vb   (becomes row i-1) */
                HVX_Vector vca = Q6_Vqf32_vmpy_VsfVsf(vc, va);
                HVX_Vector vsb = Q6_Vqf32_vmpy_VsfVsf(vs, vb);
                HVX_Vector new_a_qf = Q6_Vqf32_vadd_Vqf32Vqf32(vca, vsb);

                /* new_b = c*vb - s*va   (becomes row i; should be 0 at col k) */
                HVX_Vector vcb = Q6_Vqf32_vmpy_VsfVsf(vc, vb);
                HVX_Vector vsa = Q6_Vqf32_vmpy_VsfVsf(vs, va);
                HVX_Vector new_b_qf = Q6_Vqf32_vsub_Vqf32Vqf32(vcb, vsa);

                *(HVX_VectorU*)&row_a[j] = Q6_Vsf_equals_Vqf32(new_a_qf);
                *(HVX_VectorU*)&row_b[j] = Q6_Vsf_equals_Vqf32(new_b_qf);
            }
            /* Scalar tail for any cols not covered by HVX (n_pad is a
             * multiple of 32 by construction so this should be empty,
             * but keep for safety if k is near n_pad). */
            for (; j < n_pad; j++) {
                float a_j = row_a[j], b_j = row_b[j];
                row_a[j] = c * a_j + s * b_j;
                row_b[j] = c * b_j - s * a_j;
            }
        }
    }

    qurt_hvx_unlock();

    /* Transpose row-major S → col-major A_col (upper triangle = R),
     * zero strict lower triangle. */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j && i < m; i++) {
            MAT(*A_col, i, j) = S[(size_t)i * n_pad + j];
        }
        for (int i = j + 1; i < m; i++) {
            MAT(*A_col, i, j) = 0.f;
        }
    }
    free(S);
}

/* =====================================================================
 *  [3] HH WY-blocked  (port of qr_wy_blocked_mt.c, with the T^T fix).
 *
 *  Panel factor uses single-threaded HVX (apply_householder_panel_serial)
 *  to avoid per-Householder thread spawn. Block trailing update is
 *  multi-threaded over trail columns.
 * ===================================================================== */

static void apply_householder_panel_serial(Matrix* A, const float* v,
                                           float tau,
                                           int j_start, int j_end) {
    if (j_end <= j_start) return;
    int m = A->m;
    /* HVX is assumed already locked by the caller. */
    for (int j = j_start; j < j_end; j++) {
        HVX_Vector vSum = Q6_V_vzero();
        for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
            HVX_Vector va = *(HVX_VectorU*)&MAT(*A, i, j);
            HVX_Vector vv = *(HVX_VectorU*)&v[i];
            vSum = Q6_Vqf32_vadd_Vqf32Vqf32(vSum, Q6_Vqf32_vmpy_VsfVsf(va, vv));
        }
        float temp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
        *(HVX_VectorU*)temp = Q6_Vsf_equals_Vqf32(vSum);
        float dot = 0.f;
        for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) dot += temp[lane];
        float w = tau * dot;
        HVX_Vector vw = Q6_V_vsplat_R(*(uint32_t*)&w);
        for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
            HVX_Vector va = *(HVX_VectorU*)&MAT(*A, i, j);
            HVX_Vector vv = *(HVX_VectorU*)&v[i];
            HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, Q6_V_vzero());
            HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
            *(HVX_VectorU*)&MAT(*A, i, j) =
                Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp));
        }
    }
}

static void panel_factor(Matrix* A, int k_start, int nb_actual,
                         float* V_panel, float* tau) {
    int m = A->m;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) return;
    for (int kk = 0; kk < nb_actual; kk++) {
        int k = k_start + kk;
        float* v_kk = &V_panel[(size_t)kk * m];
        float nxsq = 0.f;
        for (int i = k; i < m; i++) { float v = MAT(*A, i, k); nxsq += v*v; }
        float nx = sqrtf(nxsq);
        if (nx < 1e-7f) {
            tau[kk] = 0.f;
            for (int i = 0; i < m; i++) v_kk[i] = 0.f;
            continue;
        }
        float alpha = -(MAT(*A, k, k) >= 0.f ? 1.f : -1.f) * nx;
        for (int i = 0; i < k; i++) v_kk[i] = 0.f;
        for (int i = k; i < m; i++) v_kk[i] = MAT(*A, i, k);
        v_kk[k] -= alpha;
        float nvsq = 0.f;
        for (int i = k; i < m; i++) nvsq += v_kk[i] * v_kk[i];
        if (nvsq < 1e-10f) {
            tau[kk] = 0.f;
            MAT(*A, k, k) = alpha;
            for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
            continue;
        }
        tau[kk] = 2.f / nvsq;
        apply_householder_panel_serial(A, v_kk, tau[kk],
                                       k + 1, k_start + nb_actual);
        MAT(*A, k, k) = alpha;
        for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
    }
    qurt_hvx_unlock();
}

static void build_T_matrix(const float* V_panel, int m, int nb_actual,
                           const float* tau, float* T) {
    memset(T, 0, (size_t)MAX_NB * MAX_NB * sizeof(float));
    for (int k = 0; k < nb_actual; k++) {
        T[(size_t)k * MAX_NB + k] = tau[k];
        if (k == 0) continue;
        float z[MAX_NB];
        const float* vk = &V_panel[(size_t)k * m];
        for (int j = 0; j < k; j++) {
            const float* vj = &V_panel[(size_t)j * m];
            float dot = 0.f;
            for (int i = 0; i < m; i++) dot += vj[i] * vk[i];
            z[j] = dot;
        }
        for (int i = 0; i < k; i++) {
            float s = 0.f;
            for (int j = i; j < k; j++)
                s += T[(size_t)j * MAX_NB + i] * z[j];
            T[(size_t)k * MAX_NB + i] = -tau[k] * s;
        }
    }
}

typedef struct {
    Matrix*       A;
    const float*  V;
    const float*  T;
    int           nb_actual;
    int           trail_start;
    int           trail_end;
    unsigned int* counter;
    int           chunk_cols;
} BlockCtx;

static void block_update_worker(void* arg) {
    BlockCtx* ctx = (BlockCtx*)arg;
    int m  = ctx->A->m;
    int nb = ctx->nb_actual;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) { qurt_thread_exit(1); return; }
    while (1) {
        unsigned int ci = qurt_atomic_inc_return(ctx->counter) - 1;
        int j_base = ctx->trail_start + (int)ci * ctx->chunk_cols;
        if (j_base >= ctx->trail_end) break;
        int j_end_c = j_base + ctx->chunk_cols;
        if (j_end_c > ctx->trail_end) j_end_c = ctx->trail_end;
        for (int j = j_base; j < j_end_c; j++) {
            float* Aj = &MAT(*ctx->A, 0, j);
            HVX_Vector acc[MAX_NB];
            for (int l = 0; l < nb; l++) acc[l] = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&Aj[i];
                for (int l = 0; l < nb; l++) {
                    HVX_Vector vv = *(HVX_VectorU*)&ctx->V[(size_t)l * m + i];
                    HVX_Vector vp = Q6_Vqf32_vmpy_VsfVsf(vv, va);
                    acc[l] = Q6_Vqf32_vadd_Vqf32Vqf32(acc[l], vp);
                }
            }
            float z[MAX_NB], w[MAX_NB];
            float tmp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
            for (int l = 0; l < nb; l++) {
                *(HVX_VectorU*)tmp = Q6_Vsf_equals_Vqf32(acc[l]);
                float s = 0.f;
                for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) s += tmp[lane];
                z[l] = s;
            }
            /* w = T^T * z   (T stored upper-tri col-major; access T^T as lower-tri) */
            for (int i = 0; i < nb; i++) {
                float s = 0.f;
                for (int jj = 0; jj <= i; jj++)
                    s += ctx->T[(size_t)i * MAX_NB + jj] * z[jj];
                w[i] = s;
            }
            HVX_Vector vw_splat[MAX_NB];
            for (int l = 0; l < nb; l++)
                vw_splat[l] = Q6_V_vsplat_R(*(uint32_t*)&w[l]);
            HVX_Vector vzero = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va    = *(HVX_VectorU*)&Aj[i];
                HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, vzero);
                for (int l = 0; l < nb; l++) {
                    HVX_Vector vv = *(HVX_VectorU*)&ctx->V[(size_t)l * m + i];
                    HVX_Vector vp = Q6_Vqf32_vmpy_VsfVsf(vv, vw_splat[l]);
                    va_qf = Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp);
                }
                *(HVX_VectorU*)&Aj[i] = Q6_Vsf_equals_Vqf32(va_qf);
            }
        }
    }
    qurt_hvx_unlock();
    qurt_thread_exit(0);
}

static void block_trailing_update_mt(Matrix* A, const float* V_panel,
                                     const float* T,
                                     int k_start, int nb_actual,
                                     int nt, int chunk_cols) {
    (void)nt;  /* now driven by the persistent thread pool */
    pool_dispatch_block_update(A, V_panel, T,
                               k_start + nb_actual, A->n,
                               nb_actual, chunk_cols);
}

static void qr_hh_wy_blocked(Matrix* A, int nb_req, int nt, int chunk_size) {
    int n = A->n, m = A->m;
    int nb_eff = nb_req;
    if (nb_eff > MAX_NB) nb_eff = MAX_NB;
    if (nb_eff < 1)      nb_eff = 1;

    float* V_panel = (float*)memalign(128, (size_t)m * MAX_NB * sizeof(float));
    float* T       = (float*)memalign(128, (size_t)MAX_NB * MAX_NB * sizeof(float));
    float* tau_buf = (float*)memalign(128, (size_t)MAX_NB * sizeof(float));
    if (!V_panel || !T || !tau_buf) {
        printf("ERROR: blocked QR scratch alloc failed\n");
        return;
    }
    for (int k_start = 0; k_start < n; k_start += nb_eff) {
        int nb_actual = nb_eff;
        if (k_start + nb_actual > n) nb_actual = n - k_start;
        panel_factor(A, k_start, nb_actual, V_panel, tau_buf);
        if (k_start + nb_actual < n) {
            build_T_matrix(V_panel, m, nb_actual, tau_buf, T);
            block_trailing_update_mt(A, V_panel, T,
                                     k_start, nb_actual, nt, chunk_size);
        }
    }
    free(V_panel);
    free(T);
    free(tau_buf);
}

/* =====================================================================
 *  [6] Tile QR — 2-way TSQR-style merge.
 *
 *  1. Split A vertically: top half T_top (m_top × n), bot half T_bot.
 *  2. QR-factor each half independently → R_top, R_bot. (Sequentially
 *     here because we only have one HVX context, but conceptually
 *     parallel — this is the parallelism advantage of TSQR.)
 *  3. Stack [R_top; R_bot] into a 2n×n scratch matrix.
 *  4. QR-factor the stacked matrix → R_final.
 *  5. Copy R_final into A's upper triangle, zero strict lower.
 *
 *  For square A (m = n), this is only modestly better than unblocked HH
 *  even with full parallelism (~0.83× total work).  TSQR's real win is
 *  for tall-skinny (m ≫ n) where leaf-tile QRs dominate and parallelism
 *  scales linearly with tile count.
 * ===================================================================== */

static void qr_tile_2way(Matrix* A_input, int nt) {
    int m = A_input->m, n = A_input->n;

    /* Need each tile to have at least HVX_FLOAT_LANES rows + at least
     * n rows (for full-rank R). Otherwise fall back to unblocked HH. */
    int m_top = (m / 2 / HVX_FLOAT_LANES) * HVX_FLOAT_LANES;
    int m_bot_raw = m - m_top;
    /* m_bot needs to be a multiple of HVX_FLOAT_LANES for create_matrix. */
    int m_bot = m_bot_raw;
    if (m_bot % HVX_FLOAT_LANES != 0) {
        /* Round m_top up to keep m_bot aligned. */
        m_top = (m_top + HVX_FLOAT_LANES - 1) / HVX_FLOAT_LANES * HVX_FLOAT_LANES;
        m_bot = m - m_top;
    }
    if (m_top < n || m_bot < n || m_top % HVX_FLOAT_LANES != 0 ||
        m_bot % HVX_FLOAT_LANES != 0) {
        printf("    (tiles too small; falling back to unblocked HH)\n");
        qr_hh_unblocked(A_input, nt, DEFAULT_CHUNK);
        return;
    }

    Matrix T_top = create_matrix(m_top, n);
    Matrix T_bot = create_matrix(m_bot, n);

    /* Copy A_input top into T_top, bot into T_bot. */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m_top; i++) MAT(T_top, i, j) = MAT(*A_input, i, j);
        for (int i = 0; i < m_bot; i++) MAT(T_bot, i, j) = MAT(*A_input, m_top + i, j);
    }

    /* Leaf tile QRs — conceptually parallel; sequential here. */
    qr_hh_unblocked(&T_top, nt, DEFAULT_CHUNK);
    qr_hh_unblocked(&T_bot, nt, DEFAULT_CHUNK);

    /* Build stacked matrix [R_top; R_bot]. Pad rows to multiple of 32. */
    int m_stack = ((2 * n + HVX_FLOAT_LANES - 1) / HVX_FLOAT_LANES) * HVX_FLOAT_LANES;
    Matrix Stack = create_matrix(m_stack, n);   /* zero-initialized */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) MAT(Stack, i,     j) = MAT(T_top, i, j);
        for (int i = 0; i < n; i++) MAT(Stack, n + i, j) = MAT(T_bot, i, j);
        /* rows 2n..m_stack-1 already zero */
    }

    /* Merge QR: factor the stacked matrix. R_final ends up in rows 0..n-1. */
    qr_hh_unblocked(&Stack, nt, DEFAULT_CHUNK);

    /* Copy R_final into A_input's upper triangle; zero strict lower. */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i <= j; i++) MAT(*A_input, i, j) = MAT(Stack, i, j);
        for (int i = j + 1; i < m; i++) MAT(*A_input, i, j) = 0.f;
    }

    free_matrix(&T_top);
    free_matrix(&T_bot);
    free_matrix(&Stack);
}

/* =====================================================================
 *  main()
 * ===================================================================== */

int main(void) {
    FILE* fp = fopen("./../inparams.txt", "r");
    if (!fp) { printf("Error: cannot open inparams.txt\n"); return 1; }
    int m = 0, n = 0, nt = 1;
    int chunk_size = DEFAULT_CHUNK;
    int use_static = 0;
    int nb         = DEFAULT_NB;
    int nread = fscanf(fp, "%d %d %d %d %d %d",
                       &m, &n, &nt, &chunk_size, &use_static, &nb);
    fclose(fp);
    if (nread < 3) {
        printf("Error: need at least m n nt in inparams.txt\n");
        return 1;
    }
    if (nread < 4) chunk_size = DEFAULT_CHUNK;
    if (nread < 5) use_static = 0;
    if (nread < 6) nb         = DEFAULT_NB;
    (void)use_static;
    if (m % HVX_FLOAT_LANES != 0) {
        printf("Error: m must be a multiple of %d\n", HVX_FLOAT_LANES);
        return 1;
    }

    printf("=== QR algorithm comparison ===\n");
    printf("Matrix: %d x %d  |  Threads: %d  |  Chunk: %d  |  nb: %d\n\n",
           m, n, nt, chunk_size, nb);

    /* Spin up the persistent thread pool used by HH-MT and WY-block. */
    pool_init(nt);

    Matrix A_orig = create_matrix(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            MAT(A_orig, i, j) = (float)((i*3 + j*7) % 11) - 5.f;

    /* CPU reference (oracle) */
    Matrix A_cpu = copy_matrix(&A_orig);
    printf("[1] CPU scalar Householder        ... ");
    uint64_t t0 = get_cycles();
    qr_cpu_householder(&A_cpu);
    uint64_t t1 = get_cycles();
    printf("%llu cycles\n", (unsigned long long)(t1 - t0));

    /* Algorithm 2: HH unblocked HVX-MT */
    Matrix A_hh = copy_matrix(&A_orig);
    printf("[2] HH unblocked HVX-MT           ... ");
    uint64_t t2 = get_cycles();
    qr_hh_unblocked(&A_hh, nt, chunk_size);
    uint64_t t3 = get_cycles();
    float err_hh = verify_R_against(&A_cpu, &A_hh);
    printf("%llu cycles  |  max |R|-err vs CPU: %.3e  %s\n",
           (unsigned long long)(t3 - t2), err_hh,
           err_hh < 1e-3f ? "OK" : "FAIL");

    /* Algorithm 3: HH WY-blocked HVX-MT (panel serial, trail threaded) */
    Matrix A_wy = copy_matrix(&A_orig);
    printf("[3] HH WY-blocked HVX-MT          ... ");
    uint64_t t8 = get_cycles();
    qr_hh_wy_blocked(&A_wy, nb, nt, chunk_size);
    uint64_t t9 = get_cycles();
    float err_wy = verify_R_against(&A_cpu, &A_wy);
    printf("%llu cycles  |  max |R|-err vs CPU: %.3e  %s\n",
           (unsigned long long)(t9 - t8), err_wy,
           err_wy < 1e-3f ? "OK" : "FAIL");

    /* Algorithm 4: MGS2 single-threaded HVX */
    Matrix A_mgs = copy_matrix(&A_orig);
    printf("[4] MGS2 (re-ortho) HVX (1 thr)   ... ");
    uint64_t t4 = get_cycles();
    qr_mgs2(&A_mgs);
    uint64_t t5 = get_cycles();
    float err_mgs = verify_R_against(&A_cpu, &A_mgs);
    printf("%llu cycles  |  max |R|-err vs CPU: %.3e  %s\n",
           (unsigned long long)(t5 - t4), err_mgs,
           err_mgs < 1e-3f ? "OK" : "FAIL");

    /* Algorithm 5: Givens (single-thread HVX, row-major scratch) */
    Matrix A_giv = copy_matrix(&A_orig);
    printf("[5] Givens rotations HVX (1 thr)  ... ");
    uint64_t t6 = get_cycles();
    qr_givens(&A_giv);
    uint64_t t7 = get_cycles();
    float err_giv = verify_R_against(&A_cpu, &A_giv);
    printf("%llu cycles  |  max |R|-err vs CPU: %.3e  %s\n",
           (unsigned long long)(t7 - t6), err_giv,
           err_giv < 1e-3f ? "OK" : "FAIL");

    /* Algorithm 6: Tile QR (2-way TSQR) */
    Matrix A_tile = copy_matrix(&A_orig);
    printf("[6] Tile QR (2-way TSQR) HVX-MT   ... ");
    uint64_t t10 = get_cycles();
    qr_tile_2way(&A_tile, nt);
    uint64_t t11 = get_cycles();
    float err_tile = verify_R_against(&A_cpu, &A_tile);
    printf("%llu cycles  |  max |R|-err vs CPU: %.3e  %s\n",
           (unsigned long long)(t11 - t10), err_tile,
           err_tile < 1e-3f ? "OK" : "FAIL");

    pool_shutdown();

    free_matrix(&A_orig);
    free_matrix(&A_cpu);
    free_matrix(&A_hh);
    free_matrix(&A_wy);
    free_matrix(&A_mgs);
    free_matrix(&A_giv);
    free_matrix(&A_tile);
    return 0;
}
