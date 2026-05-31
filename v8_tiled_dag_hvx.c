/* =====================================================================
 *  v8_tiled_dag_hvx — Phase 2: classic tiled QR (PLASMA/Buttari-style)
 *  on Hexagon, building toward DAG-priority HVX-context scheduling.
 *
 *  STEP 1 (this file, current state): SCALAR correctness skeleton.
 *  A sequential tiled QR using four tile kernels expressed through two
 *  generic Householder primitives.  No HVX, no threads yet — those land in
 *  later steps (see phase2_tiled_dag_plan.md).  Gate: R must match the
 *  unblocked ground truth before any parallelism.
 *
 *  Tiling: A is m x n, split into B x B tiles A[I][J], 0<=I<MT, 0<=J<NT,
 *  with MT=m/B, NT=n/B (B = the `nb` field of inparams; require B | m, B | n).
 *
 *  Tile kernels (LAPACK/PLASMA names):
 *    GEQRT(k)     : QR-factor diagonal tile A[k][k]            (critical path)
 *    ORMQR(k,j)   : apply GEQRT(k)'s Q^T to trailing tile A[k][j]
 *    TSQRT(i,k)   : factor stacked [A[k][k]; A[i][k]] (2B x B)
 *    TSMQR(i,k,j) : apply TSQRT(i,k)'s Q^T to stacked [A[k][j]; A[i][j]]
 *
 *  Both factor/apply reduce to two generic column-major primitives:
 *    house_qr_panel  : dense Householder QR of a rows x cols panel
 *    house_apply_panel: apply that panel's reflectors (Q^T) to a matrix
 *  TSQRT/TSMQR operate the stacked pair through a (2B x B) scratch (correct
 *  but does dense flops; the structure-exploiting form is a later optimization).
 *
 *  Correctness note: the QR R-factor is unique up to row signs, so
 *  |R_tiled| == |R_ref| elementwise — verify_R_against compares fabs, so it
 *  is the right, sign-robust gate.
 *
 *  inparams.txt format: m n nt chunk sched nb   (here only m,n,nb used; nb=B)
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

#include "hvx_utils.h"   /* hvx_dot_col_v, hvx_reduce_qf32, HVX_VectorU_t */

QURT_DECLARE_STATIC_HEAP(16777216 * 2);      /* 32 MB (same as v7; proven OK) */
unsigned int heapSize = 16777216;

#define HVX_FLOAT_LANES 32
#define DEFAULT_NB      32
#define MAX_THREADS     8
#define STACK_SIZE      (32 * 1024)

/* scheduler modes (inparams `sched` field) */
enum { SCHED_ASYNC_PRIO = 0, SCHED_ASYNC = 1, SCHED_BARRIER = 2, SCHED_SEQ = 3 };

static const char* VARIANT_NAME = "tiled_dag_hvx";
static char g_tstacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(128)));

/* #128B HVX contexts the chipset provides (bits 15:8 of get_units; e.g.
 * 0x400 == 4x128B). */
static inline int hvx_units_128b(void) {
    int u = (qurt_hvx_get_units() >> 8) & 0xff;
    return (u > 0) ? u : 4;
}

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
static float subdiag_residual(const Matrix* A) {
    float e = 0.f;
    for (int j = 0; j < A->n; j++)
        for (int i = j + 1; i < A->m; i++) {
            float a = fabsf(MAT(*A, i, j));
            if (a > e) e = a;
        }
    return e;
}

/* ---------------- CPU scalar HH reference (ground truth) ---------------- */
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
 *  Generic Householder primitives (column-major, explicit leading dim)
 * ===================================================================== */

/* Factor a rows x cols panel P (ld=ldp) in place: upper triangle becomes R,
 * P strict-lower is zeroed.  Reflector c is stored in column c of V (ld=ldv)
 * as the (unnormalized) Householder vector v with v[c]=x_c-alpha, v[r>c]=x_r,
 * v[r<c]=0; tau[c]=2/||v||^2 (same convention as qr_cpu_householder). */
static void house_qr_panel(float* P, int ldp, int rows, int cols,
                           float* V, int ldv, float* tau, int use_hvx) {
    for (int c = 0; c < cols; c++) {
        float nxsq = 0.f;
        for (int r = c; r < rows; r++) { float x = P[r + (size_t)c*ldp]; nxsq += x*x; }
        for (int r = 0; r < rows; r++) V[r + (size_t)c*ldv] = 0.f;
        float nx = sqrtf(nxsq);
        if (nx < 1e-7f) { tau[c] = 0.f; continue; }
        float xc = P[c + (size_t)c*ldp];
        float alpha = -(xc >= 0.f ? 1.f : -1.f) * nx;
        V[c + (size_t)c*ldv] = xc - alpha;
        for (int r = c + 1; r < rows; r++) V[r + (size_t)c*ldv] = P[r + (size_t)c*ldp];
        float nvsq = 0.f;
        for (int r = c; r < rows; r++) { float vv = V[r + (size_t)c*ldv]; nvsq += vv*vv; }
        if (nvsq < 1e-10f) { tau[c] = 0.f; continue; }
        float tau_c = 2.f / nvsq; tau[c] = tau_c;
        float* vcol = V + (size_t)c*ldv;
        /* Trailing update is a reflector apply.  v[r<c]=0, so dot/saxpy over the
         * full tile-aligned range [0,rows) match the masked [c,rows) range and
         * vectorize cleanly (rows is a multiple of HVX_FLOAT_LANES).  Only HVX-
         * context holders take the vector path; others stay scalar. */
        for (int jj = c; jj < cols; jj++) {
            float* pj = P + (size_t)jj*ldp;
            if (use_hvx) {
                float dot = hvx_dot_col_v(pj, vcol, rows);
                float w = tau_c * dot;
                HVX_Vector vw = Q6_V_vsplat_R(*(uint32_t*)&w);
                for (int r = 0; r < rows; r += HVX_FLOAT_LANES) {
                    HVX_Vector pa = *(HVX_VectorU_t*)(pj + r);
                    HVX_Vector vv = *(const HVX_VectorU_t*)(vcol + r);
                    HVX_Vector pa_qf = Q6_Vqf32_vadd_VsfVsf(pa, Q6_V_vzero());
                    HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
                    *(HVX_VectorU_t*)(pj + r) =
                        Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(pa_qf, vp));
                }
            } else {
                float dot = 0.f;
                for (int r = c; r < rows; r++) dot += vcol[r] * pj[r];
                float w = tau_c * dot;
                for (int r = c; r < rows; r++) pj[r] -= w * vcol[r];
            }
        }
        for (int r = c + 1; r < rows; r++) P[r + (size_t)c*ldp] = 0.f; /* exact zeros */
    }
}

/* Apply Q^T (= H_{cols_v-1} ... H_0) of the reflectors in V to C (ld=ldc),
 * rows x cols_c.  Applies H_0 first => computes Q^T C.  Scalar reference
 * (kept for correctness comparison / fallback; HVX path is the default). */
__attribute__((unused))
static void house_apply_panel(const float* V, int ldv, const float* tau,
                              int rows, int cols_v,
                              float* C, int ldc, int cols_c) {
    for (int c = 0; c < cols_v; c++) {
        float tau_c = tau[c];
        if (tau_c == 0.f) continue;
        for (int jj = 0; jj < cols_c; jj++) {
            float dot = 0.f;
            for (int r = c; r < rows; r++)
                dot += V[r + (size_t)c*ldv] * C[r + (size_t)jj*ldc];
            float w = tau_c * dot;
            for (int r = c; r < rows; r++)
                C[r + (size_t)jj*ldc] -= w * V[r + (size_t)c*ldv];
        }
    }
}

/* HVX apply: same math, vectorized over the row dimension.
 *
 * Key enabler: house_qr_panel zeroes V[r<c], so summing/updating over the
 * FULL row range [0,rows) gives identical results to the masked [c,rows)
 * range (the extra lanes multiply by V==0).  rows is a tile height (B or 2B,
 * both multiples of HVX_FLOAT_LANES), so each reflector/column is an exact
 * number of HVX vectors.  Requires V columns (ldv) and C columns (ldc) to be
 * HVX-addressable — true here: V scratch ld is rows (B or 2B); C is either a
 * matrix tile (ld=m, m mult of 32) or stacked scratch (ld=2B).  The first row
 * of each column must be 128B-ish aligned for fast loads; we use the unaligned
 * HVX_VectorU_t load (4B align) to stay correct regardless. */
static void house_apply_panel_hvx(const float* V, int ldv, const float* tau,
                                   int rows, int cols_v,
                                   float* C, int ldc, int cols_c) {
    for (int c = 0; c < cols_v; c++) {
        float tau_c = tau[c];
        if (tau_c == 0.f) continue;
        const float* vc = V + (size_t)c*ldv;          /* reflector column */
        for (int jj = 0; jj < cols_c; jj++) {
            float* cj = C + (size_t)jj*ldc;
            float dot = hvx_dot_col_v(cj, vc, rows);   /* qf32 dot + tree reduce */
            float w = tau_c * dot;
            HVX_Vector vw = Q6_V_vsplat_R(*(uint32_t*)&w);
            for (int r = 0; r < rows; r += HVX_FLOAT_LANES) {
                HVX_Vector ca = *(HVX_VectorU_t*)(cj + r);
                HVX_Vector vv = *(const HVX_VectorU_t*)(vc + r);
                HVX_Vector ca_qf = Q6_Vqf32_vadd_VsfVsf(ca, Q6_V_vzero());
                HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
                *(HVX_VectorU_t*)(cj + r) =
                    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(ca_qf, vp));
            }
        }
    }
}

/* =====================================================================
 *  Tile kernels (scalar). B = tile size; ld of A is m.
 *  Scratch is provided by the caller (reused across the sequential run).
 * ===================================================================== */
typedef struct {
    int    B;
    float* Vkk;  float* tau_kk;   /* GEQRT reflectors (B x B)  */
    float* V2;   float* tau2;     /* TSQRT reflectors (2B x B) */
    float* S;                     /* stacked factor scratch (2B x B) */
    float* S2;                    /* stacked apply  scratch (2B x B) */
} TileScratch;

static void k_geqrt(Matrix* A, TileScratch* w, int k) {
    int B = w->B, m = A->m;
    float* P = &MAT(*A, k*B, k*B);
    house_qr_panel(P, m, B, B, w->Vkk, B, w->tau_kk, 1);
}

static void k_ormqr(Matrix* A, TileScratch* w, int k, int j) {
    int B = w->B, m = A->m;
    float* C = &MAT(*A, k*B, j*B);
    house_apply_panel_hvx(w->Vkk, B, w->tau_kk, B, B, C, m, B);
}

static void k_tsqrt(Matrix* A, TileScratch* w, int i, int k) {
    int B = w->B, P2 = 2*B;
    /* build stacked S = [A[k][k]; A[i][k]] (2B x B, ld=2B) */
    for (int c = 0; c < B; c++)
        for (int r = 0; r < B; r++) {
            w->S[r       + (size_t)c*P2] = MAT(*A, k*B + r, k*B + c);
            w->S[B + r   + (size_t)c*P2] = MAT(*A, i*B + r, k*B + c);
        }
    house_qr_panel(w->S, P2, P2, B, w->V2, P2, w->tau2, 1);
    /* new R diagonal tile := top of S */
    for (int c = 0; c < B; c++)
        for (int r = 0; r < B; r++)
            MAT(*A, k*B + r, k*B + c) = w->S[r + (size_t)c*P2];
    /* bottom (A[i][k]) holds consumed reflector data; zeroed at final cleanup */
}

static void k_tsmqr(Matrix* A, TileScratch* w, int i, int k, int j) {
    int B = w->B, P2 = 2*B;
    for (int c = 0; c < B; c++)
        for (int r = 0; r < B; r++) {
            w->S2[r     + (size_t)c*P2] = MAT(*A, k*B + r, j*B + c);
            w->S2[B + r + (size_t)c*P2] = MAT(*A, i*B + r, j*B + c);
        }
    house_apply_panel_hvx(w->V2, P2, w->tau2, P2, B, w->S2, P2, B);
    for (int c = 0; c < B; c++)
        for (int r = 0; r < B; r++) {
            MAT(*A, k*B + r, j*B + c) = w->S2[r     + (size_t)c*P2];
            MAT(*A, i*B + r, j*B + c) = w->S2[B + r + (size_t)c*P2];
        }
}

/* =====================================================================
 *  Sequential tiled QR driver (correctness skeleton)
 * ===================================================================== */
static void tiled_qr_seq(Matrix* A, int B) {
    int m = A->m, n = A->n;
    int MT = m / B, NT = n / B;

    TileScratch w;
    w.B      = B;
    w.Vkk    = (float*)memalign(128, (size_t)B * B * sizeof(float));
    w.tau_kk = (float*)memalign(128, (size_t)B * sizeof(float));
    w.V2     = (float*)memalign(128, (size_t)2 * B * B * sizeof(float));
    w.tau2   = (float*)memalign(128, (size_t)B * sizeof(float));
    w.S      = (float*)memalign(128, (size_t)2 * B * B * sizeof(float));
    w.S2     = (float*)memalign(128, (size_t)2 * B * B * sizeof(float));

    /* The ORMQR/TSMQR apply kernels use HVX; acquire a context for the run.
     * (Sequential driver — the multi-context scheduling lands in step 4.) */
    int hvx_held = (qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK);
    if (!hvx_held) printf("WARNING: qurt_hvx_lock failed; apply kernels need HVX\n");

    for (int k = 0; k < NT; k++) {
        k_geqrt(A, &w, k);
        for (int j = k + 1; j < NT; j++) k_ormqr(A, &w, k, j);
        for (int i = k + 1; i < MT; i++) {
            k_tsqrt(A, &w, i, k);
            for (int j = k + 1; j < NT; j++) k_tsmqr(A, &w, i, k, j);
        }
    }

    if (hvx_held) qurt_hvx_unlock();

    /* Materialize R: zero everything strictly below the diagonal (the lower
     * tiles held reflector / consumed data — Q is not reconstructed here). */
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < m; i++)
            MAT(*A, i, j) = 0.f;

    free(w.Vkk); free(w.tau_kk); free(w.V2); free(w.tau2); free(w.S); free(w.S2);
}

/* =====================================================================
 *  PARALLEL: task-DAG + priority HVX-context scheduler
 *
 *  Tasks (kind): GEQRT(k) ; ORMQR(k,j) ; TSQRT(i,k) ; TSMQR(i,k,j).
 *  The DAG is built by tracing the (already-verified) sequential order and
 *  recording, per matrix tile, its last writer — so every read/write hazard
 *  on a shared tile becomes a dependency edge (correct-by-construction).
 *  Reflector producer edges (GEQRT->ORMQR, TSQRT->TSMQR) are added too.
 *  Because every shared-tile access is serialized by a last-writer edge, no
 *  two concurrently-runnable tasks ever touch the same tile -> no data races.
 *
 *  Priority = bottom-level (longest dependency path to a leaf): the diagonal
 *  GEQRT/TSQRT chain ranks highest.  HVX contexts are budgeted (Phase-1
 *  allocator): the first min(nt,#units) workers hold a context for life and
 *  run the apply kernels on HVX; the rest run the scalar apply.  Pulling the
 *  highest-priority ready task first => HVX cycles flow to critical work.
 * ===================================================================== */
enum { K_GEQRT = 0, K_ORMQR = 1, K_TSQRT = 2, K_TSMQR = 3 };

typedef struct {
    uint8_t          kind;
    int              k, i, j;
    int              prio;          /* bottom-level */
    int              npred;         /* static predecessor count */
    volatile unsigned pred;         /* remaining preds (atomic countdown) */
    int              soff, scnt;    /* CSR successor slice */
} Task;

typedef struct {
    Matrix* A; int m, B, MT, NT;
    /* persistent reflector storage */
    float* Vkk;  float* tau_kk;     /* per k: Vkk[k]=B*B, tau_kk[k]=B */
    float* V2;   float* tau2;       /* per (i,k): V2=2B*B, tau2=B (idx k*MT+i) */
    float* scratch;                 /* per worker: 2B*B */
    /* DAG */
    Task*  task; int ntask;
    int*   succ;                    /* CSR successor ids */
    /* ready queue (mutex + counting semaphore) */
    int*   ready; volatile int rcount;
    qurt_mutex_t rmtx; qurt_sem_t ravail;
    volatile unsigned remaining;    /* tasks not yet completed */
    volatile int done;
    int sched;                      /* SCHED_* */
    /* HVX budget */
    volatile unsigned hvx_claim; int hvx_units; int nt;
} TiledCtx;

static TiledCtx g;

static inline float* VKK(int k)      { return g.Vkk + (size_t)k * (g.B*g.B); }
static inline float* TAUKK(int k)    { return g.tau_kk + (size_t)k * g.B; }
static inline float* V2S(int i,int k){ return g.V2  + (size_t)(k*g.MT+i) * (2*g.B*g.B); }
static inline float* TAU2(int i,int k){return g.tau2 + (size_t)(k*g.MT+i) * g.B; }

/* one task body; use_hvx selects the vector path for every kernel. */
static void exec_task(const Task* t, int tid, int use_hvx) {
    Matrix* A = g.A; int B = g.B, m = g.m, P2 = 2*g.B;
    switch (t->kind) {
    case K_GEQRT:
        house_qr_panel(&MAT(*A, t->k*B, t->k*B), m, B, B, VKK(t->k), B, TAUKK(t->k), use_hvx);
        break;
    case K_ORMQR: {
        float* C = &MAT(*A, t->k*B, t->j*B);
        if (use_hvx) house_apply_panel_hvx(VKK(t->k), B, TAUKK(t->k), B, B, C, m, B);
        else         house_apply_panel    (VKK(t->k), B, TAUKK(t->k), B, B, C, m, B);
        break; }
    case K_TSQRT: {
        float* S = g.scratch + (size_t)tid * (2*B*B);
        for (int c = 0; c < B; c++)
            for (int r = 0; r < B; r++) {
                S[r     + (size_t)c*P2] = MAT(*A, t->k*B + r, t->k*B + c);
                S[B + r + (size_t)c*P2] = MAT(*A, t->i*B + r, t->k*B + c);
            }
        house_qr_panel(S, P2, P2, B, V2S(t->i,t->k), P2, TAU2(t->i,t->k), use_hvx);
        for (int c = 0; c < B; c++)
            for (int r = 0; r < B; r++)
                MAT(*A, t->k*B + r, t->k*B + c) = S[r + (size_t)c*P2];
        break; }
    case K_TSMQR: {
        float* S = g.scratch + (size_t)tid * (2*B*B);
        for (int c = 0; c < B; c++)
            for (int r = 0; r < B; r++) {
                S[r     + (size_t)c*P2] = MAT(*A, t->k*B + r, t->j*B + c);
                S[B + r + (size_t)c*P2] = MAT(*A, t->i*B + r, t->j*B + c);
            }
        if (use_hvx) house_apply_panel_hvx(V2S(t->i,t->k), P2, TAU2(t->i,t->k), P2, B, S, P2, B);
        else         house_apply_panel    (V2S(t->i,t->k), P2, TAU2(t->i,t->k), P2, B, S, P2, B);
        for (int c = 0; c < B; c++)
            for (int r = 0; r < B; r++) {
                MAT(*A, t->k*B + r, t->j*B + c) = S[r     + (size_t)c*P2];
                MAT(*A, t->i*B + r, t->j*B + c) = S[B + r + (size_t)c*P2];
            }
        break; }
    }
}

static void ready_push(int id) {
    qurt_mutex_lock(&g.rmtx);
    g.ready[g.rcount++] = id;
    qurt_mutex_unlock(&g.rmtx);
    qurt_sem_up(&g.ravail);
}

/* pop the highest-priority (or, in non-prio mode, the last) ready task;
 * returns -1 when the run is done. */
static int ready_pop(void) {
    qurt_sem_down(&g.ravail);
    qurt_mutex_lock(&g.rmtx);
    if (g.rcount == 0) {                 /* woken to exit, or (bug) no work */
        qurt_mutex_unlock(&g.rmtx);
        return -1;
    }
    int pick = 0;
    if (g.sched == SCHED_ASYNC) {
        pick = g.rcount - 1;                      /* LIFO, no priority */
    } else {
        int best = 0;                              /* max bottom-level */
        for (int x = 1; x < g.rcount; x++)
            if (g.task[g.ready[x]].prio > g.task[g.ready[best]].prio) best = x;
        pick = best;
    }
    int id = g.ready[pick];
    g.ready[pick] = g.ready[--g.rcount];           /* swap-remove */
    qurt_mutex_unlock(&g.rmtx);
    return id;
}

static void complete_task(int id) {
    const Task* t = &g.task[id];
    for (int e = 0; e < t->scnt; e++) {
        int s = g.succ[t->soff + e];
        unsigned left = qurt_atomic_add_return((unsigned*)&g.task[s].pred, -1u);
        if (left == 0) ready_push(s);
    }
    unsigned rem = qurt_atomic_add_return((unsigned*)&g.remaining, -1u);
    if (rem == 0) {
        g.done = 1;
        /* wake every worker so each sees rcount==0 and exits; extra tokens are
         * harmless (a fresh run re-inits the semaphore). */
        for (int x = 0; x < g.nt; x++) qurt_sem_up(&g.ravail);
    }
}

static qurt_thread_t g_tids[MAX_THREADS];

static void tiled_worker(void* arg) {
    int tid = (int)(intptr_t)arg;
    int use_hvx = 0;
    if ((int)qurt_atomic_add_return(&g.hvx_claim, 1u) <= g.hvx_units)
        use_hvx = (qurt_hvx_lock(QURT_HVX_MODE_128B) == QURT_EOK);
    while (1) {
        int id = ready_pop();
        if (id < 0) break;
        exec_task(&g.task[id], tid, use_hvx);
        complete_task(id);
    }
    if (use_hvx) qurt_hvx_unlock();
    qurt_thread_exit(0);
}

/* ---- DAG construction (sequential-order trace + last-writer) ----
 * Every task is created in the verified sequential order.  add_edge(s,d)
 * records the edge AND increments d's predecessor count in lockstep, so
 * npred == (#incoming edges) by construction.  Duplicate edges (none arise
 * here, but harmless) stay self-consistent: +1 npred and +1 decrement each.
 * All edges run earlier-id -> later-id, so bottom-level priority can be
 * computed in a single reverse pass. */
static int*  s_esrc; static int* s_edst; static int s_ne;
static void add_edge(int s, int d) {
    if (s < 0) return;
    s_esrc[s_ne] = s; s_edst[s_ne] = d; s_ne++;
    g.task[d].npred++;
}
static void build_dag(int B, int MT, int NT) {
    (void)B;
    int tmax = NT + NT*NT + MT*NT + MT*NT*NT;          /* loose upper bound */
    g.task = (Task*)calloc(tmax, sizeof(Task));
    s_esrc = (int*)malloc((size_t)4*tmax * sizeof(int));
    s_edst = (int*)malloc((size_t)4*tmax * sizeof(int));
    int* lw   = (int*)malloc((size_t)MT*NT * sizeof(int)); /* last writer per tile */
    int* gid  = (int*)malloc((size_t)NT * sizeof(int));
    int* tid2 = (int*)malloc((size_t)MT*NT * sizeof(int));
    for (int x = 0; x < MT*NT; x++) lw[x] = -1;

    int nt_ = 0; s_ne = 0;
    #define LWX(ti,tj) lw[(tj)*MT + (ti)]

    for (int k = 0; k < NT; k++) {
        /* GEQRT(k): r/w tile (k,k) */
        { int id = nt_++;
          g.task[id].kind=K_GEQRT; g.task[id].k=k;
          add_edge(LWX(k,k), id);
          LWX(k,k)=id; gid[k]=id; }
        /* ORMQR(k,j): reflector GEQRT(k) + r/w tile (k,j) */
        for (int j = k+1; j < NT; j++) {
          int id = nt_++;
          g.task[id].kind=K_ORMQR; g.task[id].k=k; g.task[id].j=j;
          add_edge(gid[k], id); add_edge(LWX(k,j), id);
          LWX(k,j)=id; }
        /* TSQRT(i,k) then its TSMQR row */
        for (int i = k+1; i < MT; i++) {
          int id = nt_++;
          g.task[id].kind=K_TSQRT; g.task[id].k=k; g.task[id].i=i;
          add_edge(LWX(k,k), id); add_edge(LWX(i,k), id);
          LWX(k,k)=id; LWX(i,k)=id; tid2[k*MT+i]=id;
          for (int j = k+1; j < NT; j++) {
            int sid = nt_++;
            g.task[sid].kind=K_TSMQR; g.task[sid].k=k; g.task[sid].i=i; g.task[sid].j=j;
            add_edge(tid2[k*MT+i], sid); add_edge(LWX(k,j), sid); add_edge(LWX(i,j), sid);
            LWX(k,j)=sid; LWX(i,j)=sid;
          }
        }
    }
    g.ntask = nt_;
    int ne = s_ne;
    int* esrc = s_esrc; int* edst = s_edst;

    /* CSR successor lists from edge set */
    for (int x = 0; x < nt_; x++) g.task[x].scnt = 0;
    for (int e = 0; e < ne; e++) g.task[esrc[e]].scnt++;
    int off = 0;
    for (int x = 0; x < nt_; x++) { g.task[x].soff = off; off += g.task[x].scnt; }
    g.succ = (int*)malloc((size_t)(ne ? ne : 1) * sizeof(int));
    int* cur = (int*)malloc((size_t)nt_ * sizeof(int));
    for (int x = 0; x < nt_; x++) cur[x] = g.task[x].soff;
    for (int e = 0; e < ne; e++) g.succ[cur[esrc[e]]++] = edst[e];

    /* bottom-level priority: reverse creation order (all edges go low->high id) */
    for (int x = nt_ - 1; x >= 0; x--) {
        int best = 0;
        for (int e = 0; e < g.task[x].scnt; e++) {
            int s = g.succ[g.task[x].soff + e];
            if (g.task[s].prio > best) best = g.task[s].prio;
        }
        g.task[x].prio = best + 1;
    }
    /* arm predecessor countdowns */
    for (int x = 0; x < nt_; x++) g.task[x].pred = (unsigned)g.task[x].npred;

    #undef LWX
    free(esrc); free(edst); free(lw); free(gid); free(tid2); free(cur);
}

static void tiled_qr_par(Matrix* A, int B, int nt, int sched) {
    int m = A->m, n = A->n, MT = m/B, NT = n/B;
    if (nt > MAX_THREADS) nt = MAX_THREADS;
    if (nt < 1) nt = 1;

    g.A = A; g.m = m; g.B = B; g.MT = MT; g.NT = NT; g.nt = nt; g.sched = sched;
    g.Vkk     = (float*)memalign(128, (size_t)NT * B*B * sizeof(float));
    g.tau_kk  = (float*)memalign(128, (size_t)NT * B   * sizeof(float));
    g.V2      = (float*)memalign(128, (size_t)NT*MT * 2*B*B * sizeof(float));
    g.tau2    = (float*)memalign(128, (size_t)NT*MT * B     * sizeof(float));
    g.scratch = (float*)memalign(128, (size_t)nt * 2*B*B * sizeof(float));

    build_dag(B, MT, NT);
    g.ready = (int*)malloc((size_t)g.ntask * sizeof(int));
    g.rcount = 0; g.done = 0; g.remaining = (unsigned)g.ntask;
    g.hvx_claim = 0;
    g.hvx_units = hvx_units_128b();
    if (g.hvx_units > nt) g.hvx_units = nt;     /* can't hold more than #workers */
    qurt_mutex_init(&g.rmtx);
    qurt_sem_init_val(&g.ravail, 0);     /* explicit count 0 (plain init is unspecified) */

    for (int x = 0; x < g.ntask; x++) if (g.task[x].pred == 0) ready_push(x);

    qurt_thread_attr_t attr;
    for (int t = 0; t < nt; t++) {
        qurt_thread_attr_init(&attr);
        qurt_thread_attr_set_stack_addr(&attr, g_tstacks[t]);
        qurt_thread_attr_set_stack_size(&attr, STACK_SIZE);
        qurt_thread_attr_set_priority(&attr, 100);
        qurt_thread_create(&g_tids[t], &attr, tiled_worker, (void*)(intptr_t)t);
    }
    for (int t = 0; t < nt; t++) { int s; qurt_thread_join(g_tids[t], &s); }

    /* NB: deliberately not destroying g.rmtx / g.ravail here.  qurt_thread_join
     * returns once a worker calls qurt_thread_exit, but a just-woken worker can
     * still be unwinding its final qurt_sem_down when we'd destroy the sem,
     * tripping "destroying semaphore in use".  The objects are re-initialized at
     * the next tiled_qr_par call, so skipping destroy is safe for our runner. */

    /* materialize R: zero strictly-below-diagonal */
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < m; i++) MAT(*A, i, j) = 0.f;

    free(g.Vkk); free(g.tau_kk); free(g.V2); free(g.tau2); free(g.scratch);
    free(g.task); free(g.succ); free(g.ready);
}

/* =====================================================================
 *  main
 * ===================================================================== */
int main(void) {
    FILE* fp = fopen("./../inparams.txt", "r");
    if (!fp) { printf("Error: cannot open ./../inparams.txt\n"); return 1; }
    int m = 0, n = 0, nt = 1, chunk = 4, sched = 0, nb = DEFAULT_NB;
    int nread = fscanf(fp, "%d %d %d %d %d %d", &m, &n, &nt, &chunk, &sched, &nb);
    fclose(fp);
    if (nread < 3) { printf("Error: inparams.txt needs m n nt\n"); return 1; }
    int B = nb;
    (void)chunk;

    if (m % HVX_FLOAT_LANES != 0) { printf("Error: m must be a multiple of %d\n", HVX_FLOAT_LANES); return 1; }
    if (B < 1)        { printf("Error: nb (tile size B) must be >= 1\n"); return 1; }
    if (B % HVX_FLOAT_LANES != 0) { printf("Error: B=%d must be a multiple of %d\n", B, HVX_FLOAT_LANES); return 1; }
    if (m % B != 0)   { printf("Error: B=%d must divide m=%d\n", B, m); return 1; }
    if (n % B != 0)   { printf("Error: B=%d must divide n=%d\n", B, n); return 1; }

    const char* sname = (sched==SCHED_SEQ)?"seq":(sched==SCHED_BARRIER)?"barrier":
                        (sched==SCHED_ASYNC)?"async":"async_prio";
    printf("=== %s [%s] ===\n", VARIANT_NAME, sname);
    printf("Matrix: %d x %d  |  Tile B: %d  |  Grid: %d x %d  |  nt: %d\n\n",
           m, n, B, m/B, n/B, nt);
    fflush(stdout);

    Matrix A_orig = create_matrix(m, n);
    fill_test_matrix(&A_orig);

    Matrix A_cpu = copy_matrix(&A_orig);
    printf("[ref] CPU scalar Householder        ... "); fflush(stdout);
    uint64_t t0 = get_cycles();
    qr_cpu_householder(&A_cpu);
    uint64_t t1 = get_cycles();
    printf("%llu cycles\n", (unsigned long long)(t1 - t0)); fflush(stdout);

    Matrix A_alg = copy_matrix(&A_orig);
    printf("[alg] %s [%s] ... ", VARIANT_NAME, sname); fflush(stdout);
    uint64_t t2 = get_cycles();
    if (sched == SCHED_SEQ) tiled_qr_seq(&A_alg, B);
    else                    tiled_qr_par(&A_alg, B, nt, sched);
    uint64_t t3 = get_cycles();

    float err = verify_R_against(&A_cpu, &A_alg);
    float sub = subdiag_residual(&A_alg);
    printf("%llu cycles  |  max |R|-err: %.3e %s  |  sub-diag: %.3e %s\n",
           (unsigned long long)(t3 - t2),
           err, err < 1e-3f ? "OK" : "FAIL",
           sub, sub < 1e-5f ? "OK" : "FAIL");

    free_matrix(&A_orig); free_matrix(&A_cpu); free_matrix(&A_alg);
    return 0;
}
