/* =====================================================================
 *  GEMM-based compact-WY blocked Householder QR (column-major, HVX,
 *  persistent thread pool).
 *
 *  Sibling of qr_pool_hh_wy_blocked.c. Same algorithm, same pool, same
 *  panel factorisation and T builder. The ONLY difference is how the
 *  trailing-block update
 *
 *        A_trail := A_trail - V (T^T (V^T A_trail))
 *
 *  is realised. qr_pool_hh_wy_blocked.c fuses it into a single
 *  per-trailing-column kernel, which re-streams all of V from memory for
 *  every trailing column (twice). Here we instead express it as two
 *  explicit GEMM passes and implement the dominant one with the tiled
 *  HVX micro-kernel idiom from matmul_vectorised_parallel.c — splat a
 *  scalar from one operand, vector-load 32 contiguous floats from the
 *  other, accumulate in qf32 — plus register-blocking over trailing
 *  columns so each V load feeds CB_GEMM outputs. On large vector units
 *  the higher reuse / arithmetic intensity is where the extra FLOPs/byte
 *  should pay off.
 *
 *  Pass A:  Y := T^T (V^T A_trail)     (nb x t, materialised)
 *           V^T A_trail is a transpose-times-matrix: the contraction runs
 *           down the contiguous m axis, so it is a vector-REDUCTION GEMM
 *           (dot products), not the splat/vector idiom.
 *  Pass B:  A_trail := A_trail - V Y   (m x t, the heavy rank-nb update)
 *           This is the matmul-style GEMM: vectorise over the contiguous
 *           row axis m, splat Y[l][j], vector-load V[:,l], qf32 FMA, with
 *           CB_GEMM columns sharing each V load.
 *
 *  inparams.txt format: m n nt [chunk_size] [use_static] [nb]
 *      nb = WY block width (1..32). chunk_size = trailing columns each
 *      worker claims per atomic grab (the 1-D "grid" granularity).
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
#define CB_GEMM         8       /* Pass B register-block: cols sharing a V load */

static const char* VARIANT_NAME = "pool_hh_wy_gemm";

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
    POOL_JOB_GEMM_VTA,    /* Pass A: Y := T^T (V^T A_trail)        */
    POOL_JOB_GEMM_VY,     /* Pass B: A_trail := A_trail - V Y      */
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

    /* Shared GEMM-job descriptor.
     *   A : the working matrix (m x n, column-major)
     *   V : m x nb reflectors (column-major, V[l*m + i])
     *   T : MAX_NB x MAX_NB row-major upper-tri (T[k*MAX_NB+i], k<=i)
     *   Y : nb x t scratch, column-major over [trail_start,trail_end):
     *       Y[l + (j - trail_start)*nb]
     * Both passes dispatch a dynamic 1-D "grid" over trailing columns;
     * each worker claims `chunk` columns per atomic grab. */
    Matrix*               g_A;
    const float*          g_V;
    const float*          g_T;
    float*                g_Y;
    int                   g_nb;
    int                   g_trail_start;
    int                   g_trail_end;
    unsigned int          g_counter;
    int                   g_chunk;
} ThreadPool;

static ThreadPool    g_pool;
static qurt_thread_t g_pool_tids[MAX_THREADS];
static int           g_pool_running = 0;

/* ---- Pass A: Y := T^T (V^T A_trail) ----
 * V^T A_trail is a transpose-times-matrix; the contraction is over the
 * contiguous m axis, so per trailing column j we vector-reduce nb dot
 * products z = V^T a (A's column loaded once, reused for all nb), then
 * apply the tiny nb x nb T^T and store the column of Y. */
static void pool_do_gemm_vta(void) {
    int m  = g_pool.g_A->m;
    int nb = g_pool.g_nb;
    int chunk = g_pool.g_chunk;
    const float* V = g_pool.g_V;
    const float* T = g_pool.g_T;
    float*       Y = g_pool.g_Y;
    int ts = g_pool.g_trail_start;

    while (1) {
        unsigned int ci = qurt_atomic_inc_return(&g_pool.g_counter) - 1;
        int j_base = ts + (int)ci * chunk;
        if (j_base >= g_pool.g_trail_end) break;
        int j_end_c = j_base + chunk;
        if (j_end_c > g_pool.g_trail_end) j_end_c = g_pool.g_trail_end;

        for (int j = j_base; j < j_end_c; j++) {
            const float* Aj = &MAT(*g_pool.g_A, 0, j);

            /* z = V^T a : nb dot products, reduce down the m axis. */
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
            /* y = T^T z  (T upper-tri, so (T^T z)[i] = sum_{k<=i} T[k][i] z[k]) */
            float* Yj = &Y[(size_t)(j - ts) * nb];
            for (int i = 0; i < nb; i++) {
                float s = 0.f;
                for (int k = 0; k <= i; k++)
                    s += T[(size_t)k * MAX_NB + i] * z[k];
                Yj[i] = s;
            }
        }
    }
}

/* ---- Pass B: A_trail := A_trail - V Y ----
 * The dominant rank-nb GEMM, matmul_vectorised_parallel.c style:
 * vectorise over the contiguous row axis m, splat the Y scalar, vector-
 * load the V column, accumulate in qf32. Register-blocked CB_GEMM columns
 * at a time so each V[:,l] load is reused across CB_GEMM outputs. */
static void pool_do_gemm_vy(void) {
    int m  = g_pool.g_A->m;
    int nb = g_pool.g_nb;
    int chunk = g_pool.g_chunk;
    const float* V = g_pool.g_V;
    const float* Y = g_pool.g_Y;
    int ts = g_pool.g_trail_start;
    HVX_Vector vzero = Q6_V_vzero();

    while (1) {
        unsigned int ci = qurt_atomic_inc_return(&g_pool.g_counter) - 1;
        int j_base = ts + (int)ci * chunk;
        if (j_base >= g_pool.g_trail_end) break;
        int j_end_c = j_base + chunk;
        if (j_end_c > g_pool.g_trail_end) j_end_c = g_pool.g_trail_end;

        for (int jb = j_base; jb < j_end_c; jb += CB_GEMM) {
            int cb = j_end_c - jb;
            if (cb > CB_GEMM) cb = CB_GEMM;

            float* Acol[CB_GEMM];
            const float* Yc[CB_GEMM];
            for (int c = 0; c < cb; c++) {
                Acol[c] = &MAT(*g_pool.g_A, 0, jb + c);
                Yc[c]   = &Y[(size_t)(jb + c - ts) * nb];
            }

            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector acc[CB_GEMM];
                for (int c = 0; c < cb; c++)
                    acc[c] = Q6_Vqf32_vadd_VsfVsf(*(HVX_VectorU*)&Acol[c][i], vzero);

                for (int l = 0; l < nb; l++) {
                    HVX_Vector vv = *(HVX_VectorU*)&V[(size_t)l * m + i]; /* one load */
                    for (int c = 0; c < cb; c++) {
                        float yv = Yc[c][l];
                        HVX_Vector vy = Q6_V_vsplat_R(*(uint32_t*)&yv);
                        acc[c] = Q6_Vqf32_vsub_Vqf32Vqf32(
                                     acc[c], Q6_Vqf32_vmpy_VsfVsf(vv, vy));
                    }
                }
                for (int c = 0; c < cb; c++)
                    *(HVX_VectorU*)&Acol[c][i] = Q6_Vsf_equals_Vqf32(acc[c]);
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
                case POOL_JOB_GEMM_VTA: pool_do_gemm_vta(); break;
                case POOL_JOB_GEMM_VY:  pool_do_gemm_vy();  break;
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
    for (int t = 0; t < g_pool.nt; t++) {
        int s; qurt_thread_join(g_pool_tids[t], &s);
    }
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_destroy(&g_pool.go[t]);
    qurt_signal_destroy(&g_pool.done_master);
    g_pool_running = 0;
}

/* Dispatch one GEMM pass over the trailing columns and block until done. */
static void pool_dispatch(PoolJobKind kind, Matrix* A, const float* V,
                          const float* T, float* Y, int nb,
                          int trail_start, int trail_end, int chunk) {
    if (trail_end <= trail_start || !g_pool_running) return;
    if (chunk < 1) chunk = 1;
    g_pool.kind        = kind;
    g_pool.g_A         = A;
    g_pool.g_V         = V;
    g_pool.g_T         = T;
    g_pool.g_Y         = Y;
    g_pool.g_nb        = nb;
    g_pool.g_trail_start = trail_start;
    g_pool.g_trail_end   = trail_end;
    g_pool.g_counter   = 0;
    g_pool.g_chunk     = chunk;
    g_pool.done_count  = g_pool.nt;

    qurt_signal_clear(&g_pool.done_master, 1u);
    for (int t = 0; t < g_pool.nt; t++) qurt_signal_set(&g_pool.go[t], 1u);
    qurt_signal_wait(&g_pool.done_master, 1u, QURT_SIGNAL_ATTR_WAIT_ANY);
}

/* =====================================================================
 *  Panel factorisation + compact-WY T builder (serial, single HVX lock)
 * ===================================================================== */

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
        apply_reflector_serial(A, v_kk, tau[kk], k + 1, k_start + nb_actual);
        MAT(*A, k, k) = alpha;
        for (int i = k + 1; i < m; i++) MAT(*A, i, k) = 0.f;
    }
    qurt_hvx_unlock();
}

/* Build compact-WY T (nb x nb, upper-triangular, LARFT forward). */
static void build_T_matrix(const float* V_panel, int m, int nb_actual,
                           const float* tau, float* T) {
    memset(T, 0, (size_t)MAX_NB * MAX_NB * sizeof(float));
    for (int k = 0; k < nb_actual; k++) {
        T[(size_t)k * MAX_NB + k] = tau[k];
        if (k == 0) continue;
        const float* vk = &V_panel[(size_t)k * m];
        float z[MAX_NB];
        for (int j = 0; j < k; j++) {
            const float* vj = &V_panel[(size_t)j * m];
            float dot = 0.f;
            for (int i = 0; i < m; i++) dot += vj[i] * vk[i];
            z[j] = dot;
        }
        for (int i = 0; i < k; i++) {
            float s = 0.f;
            for (int j = i; j < k; j++)
                s += T[(size_t)i * MAX_NB + j] * z[j];
            T[(size_t)i * MAX_NB + k] = -tau[k] * s;
        }
    }
}

/* =====================================================================
 *  GEMM-based WY-blocked driver
 * ===================================================================== */

static void qr_hh_wy_gemm_pool(Matrix* A, int nb, int chunk_size) {
    int m = A->m, n = A->n;
    if (nb < 1) nb = 1;
    if (nb > MAX_NB) nb = MAX_NB;

    float* V_panel = (float*)memalign(128, (size_t)nb * m * sizeof(float));
    float* T       = (float*)memalign(128, (size_t)MAX_NB * MAX_NB * sizeof(float));
    float* Y       = (float*)memalign(128, (size_t)nb * n * sizeof(float)); /* nb x t scratch */
    float  tau[MAX_NB];

    for (int k_start = 0; k_start < n; k_start += nb) {
        int nb_actual = nb;
        if (k_start + nb_actual > n) nb_actual = n - k_start;

        /* 1) Factor the panel (serial HVX) -> V, tau. */
        panel_factor(A, k_start, nb_actual, V_panel, tau);

        /* 2) Build compact-WY T. */
        build_T_matrix(V_panel, m, nb_actual, tau, T);

        int trail_start = k_start + nb_actual;
        if (trail_start >= n) continue;

        /* 3a) Pass A GEMM: Y := T^T (V^T A_trail). */
        pool_dispatch(POOL_JOB_GEMM_VTA, A, V_panel, T, Y, nb_actual,
                      trail_start, n, chunk_size);
        /* 3b) Pass B GEMM: A_trail := A_trail - V Y. */
        pool_dispatch(POOL_JOB_GEMM_VY, A, V_panel, T, Y, nb_actual,
                      trail_start, n, chunk_size);
    }

    free(V_panel);
    free(T);
    free(Y);
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

    /* Pool-based GEMM WY-blocked HH */
    pool_init(nt);
    Matrix A_alg = copy_matrix(&A_orig);
    printf("[alg] %s              ... ", VARIANT_NAME);
    uint64_t t2 = get_cycles();
    qr_hh_wy_gemm_pool(&A_alg, nb, chunk);
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
