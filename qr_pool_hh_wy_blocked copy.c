/* =====================================================================
 *  Persistent thread-pool variant of compact-WY *blocked* Householder QR
 *  (column-major, HVX).
 *
 *  Why block?  The unblocked variant (qr_pool_hh_unblocked.c) updates the
 *  trailing matrix one Householder at a time: for every reflector it
 *  streams every trailing column through HVX to do a rank-1 update
 *  (one dot + one axpy per column). That is a BLAS-2 access pattern — each
 *  trailing element is loaded, touched once, and stored, so the kernel is
 *  memory-bound and the wide HVX ALUs sit mostly idle.
 *
 *  The compact-WY block form accumulates `nb` reflectors into a single
 *  block transform
 *
 *        H_0 H_1 ... H_{nb-1} = I - V T V^T,
 *
 *  with V (m x nb, the reflector vectors) and T (nb x nb, upper
 *  triangular). The trailing update for the whole panel then becomes
 *
 *        A_trail := A_trail - V (T (V^T A_trail)),
 *
 *  a BLAS-3 operation. Each trailing column is loaded ONCE and reused
 *  across all `nb` reflectors (nb dot-products + nb axpys per load), so
 *  arithmetic intensity grows ~nb-fold and the HVX vector units finally
 *  have enough work to amortise the load/store traffic. On parts with
 *  large vector units this is exactly where the extra FLOPs/byte pay off.
 *
 *  Structure (mirrors qr_pool_hh_unblocked.c):
 *    - Spawn `nt` workers ONCE at start; each waits on its own
 *      qurt_signal_t (QuRT signals are single-waiter).
 *    - Main does the (cheap, narrow) panel factorisation serially under
 *      one HVX lock, builds T, then dispatches the (wide, expensive)
 *      block trailing update across the pool and waits on done_master.
 *    - Workers acquire HVX per job so other code can use HVX between
 *      dispatches.
 *
 *  Same deterministic input / output format as the other variants so the
 *  experiment notebook can parse it.
 *
 *  inparams.txt format: m n nt [chunk_size] [use_static] [nb]
 *      nb = WY block width (1..32). nb=1 degenerates to the unblocked
 *      algorithm (with extra T overhead), so use nb>=4 to see the win.
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
#define MAX_NB          32      /* T is MAX_NB x MAX_NB; nb clamped to this */

static const char* VARIANT_NAME = "pool_hh_wy_blocked";

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
    POOL_JOB_BLOCK_UPDATE,
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

    /* BLOCK_UPDATE job descriptor:
     *   A_trail := A_trail - V (T (V^T A_trail))  over cols [trail_start,trail_end)
     *   V  : m x nb, column-major, V[l*m + i]
     *   T  : MAX_NB x MAX_NB, row-major, upper-triangular (T[i*MAX_NB+k], i<=k) */
    Matrix*               bu_A;
    const float*          bu_V;
    const float*          bu_T;
    int                   bu_nb;
    int                   bu_trail_start;
    int                   bu_trail_end;
    unsigned int          bu_counter;     /* atomic work-claim */
    int                   bu_chunk_cols;
} ThreadPool;

static ThreadPool    g_pool;
static qurt_thread_t g_pool_tids[MAX_THREADS];
static int           g_pool_running = 0;

/* ---- Worker body: compact-WY block trailing update ----
 * Each claimed trailing column is touched exactly once for the whole
 * panel of `nb` reflectors: nb fused-multiply dot-products to form
 * z = V^T a, a tiny nb x nb triangular multiply y = T z, then nb axpys
 * a -= V y. This is the BLAS-3 hot loop the blocking exists for. */
static void pool_do_block_update(void) {
    int m  = g_pool.bu_A->m;
    int nb = g_pool.bu_nb;
    int chunk = g_pool.bu_chunk_cols;
    const float* V = g_pool.bu_V;
    const float* T = g_pool.bu_T;

    while (1) {
        unsigned int ci = qurt_atomic_inc_return(&g_pool.bu_counter) - 1;
        int j_base = g_pool.bu_trail_start + (int)ci * chunk;
        if (j_base >= g_pool.bu_trail_end) break;
        int j_end_c = j_base + chunk;
        if (j_end_c > g_pool.bu_trail_end) j_end_c = g_pool.bu_trail_end;

        for (int j = j_base; j < j_end_c; j++) {
            float* Aj = &MAT(*g_pool.bu_A, 0, j);

            /* Phase 1: z = V^T a  (nb dot-products, one load of Aj reused) */
            HVX_Vector acc[MAX_NB];
            for (int l = 0; l < nb; l++) acc[l] = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&Aj[i];
                for (int l = 0; l < nb; l++) {
                    HVX_Vector vv = *(HVX_VectorU*)&V[(size_t)l * m + i];
                    acc[l] = Q6_Vqf32_vadd_Vqf32Vqf32(
                                 acc[l], Q6_Vqf32_vmpy_VsfVsf(vv, va));
                }
            }
            float z[MAX_NB];
            float tmp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
            for (int l = 0; l < nb; l++) {
                *(HVX_VectorU*)tmp = Q6_Vsf_equals_Vqf32(acc[l]);
                float s = 0.f;
                for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) s += tmp[lane];
                z[l] = s;
            }

            /* Phase 2: y = T^T z.
             * QR applies the panel reflectors as H_0, then H_1, ... so the
             * transform seen by the trailing matrix is H_{nb-1}...H_0 =
             * (H_0...H_{nb-1})^T = (I - V T V^T)^T = I - V T^T V^T.
             * Hence T^T (not T) multiplies z here.  T is upper-triangular
             * (T[k][i] nonzero for k<=i), so (T^T z)[i] = sum_{k<=i} T[k][i] z[k]. */
            float y[MAX_NB];
            for (int i = 0; i < nb; i++) {
                float s = 0.f;
                for (int k = 0; k <= i; k++)
                    s += T[(size_t)k * MAX_NB + i] * z[k];
                y[i] = s;
            }

            /* Phase 3: a -= V y  (nb axpys, one load+store of Aj) */
            HVX_Vector vy[MAX_NB];
            for (int l = 0; l < nb; l++)
                vy[l] = Q6_V_vsplat_R(*(uint32_t*)&y[l]);
            HVX_Vector vzero = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va    = *(HVX_VectorU*)&Aj[i];
                HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, vzero);
                for (int l = 0; l < nb; l++) {
                    HVX_Vector vv = *(HVX_VectorU*)&V[(size_t)l * m + i];
                    va_qf = Q6_Vqf32_vsub_Vqf32Vqf32(
                                va_qf, Q6_Vqf32_vmpy_VsfVsf(vv, vy[l]));
                }
                *(HVX_VectorU*)&Aj[i] = Q6_Vsf_equals_Vqf32(va_qf);
            }
        }
    }
}

static void pool_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    qurt_signal_t* my_go = &g_pool.go[tid];

    while (1) {
        qurt_signal_wait(my_go, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
        qurt_signal_clear(my_go, 1u);

        if (g_pool.kind == POOL_JOB_SHUTDOWN) break;

        if (qurt_hvx_lock(QURT_HVX_MODE_128B) == 0) {
            switch (g_pool.kind) {
                case POOL_JOB_BLOCK_UPDATE: pool_do_block_update(); break;
                default: break;
            }
            qurt_hvx_unlock();
        }

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

static void pool_dispatch_block_update(Matrix* A, const float* V, const float* T,
                                       int nb, int trail_start, int trail_end,
                                       int chunk_cols) {
    if (trail_end <= trail_start || !g_pool_running) return;
    if (chunk_cols < 1) chunk_cols = 1;
    g_pool.kind           = POOL_JOB_BLOCK_UPDATE;
    g_pool.bu_A           = A;
    g_pool.bu_V           = V;
    g_pool.bu_T           = T;
    g_pool.bu_nb          = nb;
    g_pool.bu_trail_start = trail_start;
    g_pool.bu_trail_end   = trail_end;
    g_pool.bu_counter     = 0;
    g_pool.bu_chunk_cols  = chunk_cols;
    g_pool.done_count     = g_pool.nt;

    qurt_signal_clear(&g_pool.done_master, 1u);
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_set(&g_pool.go[t], 1u);
    qurt_signal_wait(&g_pool.done_master, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
}

/* =====================================================================
 *  Panel factorisation + compact-WY T builder (serial, single HVX lock)
 *
 *  The panel is narrow (nb columns), so factoring it serially avoids
 *  paying dispatch/HVX-lock churn for tiny per-column work; the parallel
 *  win is reserved for the wide trailing update.
 * ===================================================================== */

/* Apply a single reflector H = I - tau v v^T to A columns [j_start,j_end).
 * Caller already holds the HVX lock. Used to bring the rest of the panel
 * up to date as each in-panel reflector is formed. */
static void apply_reflector_serial(Matrix* A, const float* v, float tau,
                                   int j_start, int j_end) {
    if (j_end <= j_start) return;
    int m = A->m;
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
            HVX_Vector va    = *(HVX_VectorU*)&MAT(*A, i, j);
            HVX_Vector vv    = *(HVX_VectorU*)&v[i];
            HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, Q6_V_vzero());
            HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
            *(HVX_VectorU*)&MAT(*A, i, j) =
                Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp));
        }
    }
}

/* Factor columns [k_start, k_start+nb_actual) in place. Fills V_panel
 * (m x nb_actual, column-major) with the reflector vectors and tau[]. */
static void panel_factor(Matrix* A, int k_start, int nb_actual,
                         float* V_panel, float* tau) {
    int m = A->m;
    if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) return;
    for (int kk = 0; kk < nb_actual; kk++) {
        int k = k_start + kk;
        float* v_kk = &V_panel[(size_t)kk * m];

        float nxsq = 0.f;
        for (int i = k; i < m; i++) { float val = MAT(*A, i, k); nxsq += val*val; }
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

        /* Bring the remaining panel columns up to date so the next
         * reflector is formed from the correct trailing data. */
        apply_reflector_serial(A, v_kk, tau[kk], k + 1, k_start + nb_actual);

        MAT(*A, k, k) = alpha;
        for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
    }
    qurt_hvx_unlock();
}

/* Build the compact-WY factor T (nb x nb, upper-triangular) such that
 * H_0 H_1 ... H_{nb-1} = I - V T V^T  (LAPACK LARFT, forward/columnwise).
 * T stored row-major with stride MAX_NB; only i<=k entries are set. */
static void build_T_matrix(const float* V_panel, int m, int nb_actual,
                           const float* tau, float* T) {
    memset(T, 0, (size_t)MAX_NB * MAX_NB * sizeof(float));
    for (int k = 0; k < nb_actual; k++) {
        T[(size_t)k * MAX_NB + k] = tau[k];
        if (k == 0) continue;
        const float* vk = &V_panel[(size_t)k * m];
        /* z[j] = v_j^T v_k  for j < k */
        float z[MAX_NB];
        for (int j = 0; j < k; j++) {
            const float* vj = &V_panel[(size_t)j * m];
            float dot = 0.f;
            for (int i = 0; i < m; i++) dot += vj[i] * vk[i];
            z[j] = dot;
        }
        /* T(0:k, k) = -tau_k * T(0:k,0:k) * z   (T upper-triangular) */
        for (int i = 0; i < k; i++) {
            float s = 0.f;
            for (int j = i; j < k; j++)
                s += T[(size_t)i * MAX_NB + j] * z[j];
            T[(size_t)i * MAX_NB + k] = -tau[k] * s;
        }
    }
}

/* =====================================================================
 *  WY-blocked driver
 * ===================================================================== */

static void qr_hh_wy_blocked_pool(Matrix* A, int nb, int chunk_size) {
    int m = A->m, n = A->n;
    if (nb < 1) nb = 1;
    if (nb > MAX_NB) nb = MAX_NB;

    float* V_panel = (float*)memalign(128, (size_t)nb * m * sizeof(float));
    float* T       = (float*)memalign(128, (size_t)MAX_NB * MAX_NB * sizeof(float));
    float  tau[MAX_NB];

    for (int k_start = 0; k_start < n; k_start += nb) {
        int nb_actual = nb;
        if (k_start + nb_actual > n) nb_actual = n - k_start;

        /* 1) Factor the panel (serial HVX), producing V and tau. */
        panel_factor(A, k_start, nb_actual, V_panel, tau);

        /* 2) Build compact-WY T from V, tau (scalar; small nb x nb). */
        build_T_matrix(V_panel, m, nb_actual, tau, T);

        /* 3) Apply the block reflector to the trailing columns in
         *    parallel: A_trail := A_trail - V (T (V^T A_trail)). */
        pool_dispatch_block_update(A, V_panel, T, nb_actual,
                                   k_start + nb_actual, n, chunk_size);
    }

    free(V_panel);
    free(T);
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
    if (nb < 1)      nb = DEFAULT_NB;
    if (nb > MAX_NB) nb = MAX_NB;
    (void)use_static;

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

    /* Pool-based WY-blocked HH */
    pool_init(nt);
    Matrix A_alg = copy_matrix(&A_orig);
    printf("[alg] %s           ... ", VARIANT_NAME);
    uint64_t t2 = get_cycles();
    qr_hh_wy_blocked_pool(&A_alg, nb, chunk);
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
