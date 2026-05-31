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
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

QURT_DECLARE_STATIC_HEAP(16777216 * 2);
unsigned int heapSize = 16777216;

#define HVX_FLOAT_LANES 32
#define DEFAULT_NB      32

static const char* VARIANT_NAME = "tiled_dag_hvx [seq-scalar]";

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
                           float* V, int ldv, float* tau) {
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
        for (int jj = c; jj < cols; jj++) {
            float dot = 0.f;
            for (int r = c; r < rows; r++)
                dot += V[r + (size_t)c*ldv] * P[r + (size_t)jj*ldp];
            float w = tau_c * dot;
            for (int r = c; r < rows; r++)
                P[r + (size_t)jj*ldp] -= w * V[r + (size_t)c*ldv];
        }
        for (int r = c + 1; r < rows; r++) P[r + (size_t)c*ldp] = 0.f; /* exact zeros */
    }
}

/* Apply Q^T (= H_{cols_v-1} ... H_0) of the reflectors in V to C (ld=ldc),
 * rows x cols_c.  Applies H_0 first => computes Q^T C. */
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
    house_qr_panel(P, m, B, B, w->Vkk, B, w->tau_kk);
}

static void k_ormqr(Matrix* A, TileScratch* w, int k, int j) {
    int B = w->B, m = A->m;
    float* C = &MAT(*A, k*B, j*B);
    house_apply_panel(w->Vkk, B, w->tau_kk, B, B, C, m, B);
}

static void k_tsqrt(Matrix* A, TileScratch* w, int i, int k) {
    int B = w->B, P2 = 2*B;
    /* build stacked S = [A[k][k]; A[i][k]] (2B x B, ld=2B) */
    for (int c = 0; c < B; c++)
        for (int r = 0; r < B; r++) {
            w->S[r       + (size_t)c*P2] = MAT(*A, k*B + r, k*B + c);
            w->S[B + r   + (size_t)c*P2] = MAT(*A, i*B + r, k*B + c);
        }
    house_qr_panel(w->S, P2, P2, B, w->V2, P2, w->tau2);
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
    house_apply_panel(w->V2, P2, w->tau2, P2, B, w->S2, P2, B);
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

    for (int k = 0; k < NT; k++) {
        k_geqrt(A, &w, k);
        for (int j = k + 1; j < NT; j++) k_ormqr(A, &w, k, j);
        for (int i = k + 1; i < MT; i++) {
            k_tsqrt(A, &w, i, k);
            for (int j = k + 1; j < NT; j++) k_tsmqr(A, &w, i, k, j);
        }
    }

    /* Materialize R: zero everything strictly below the diagonal (the lower
     * tiles held reflector / consumed data — Q is not reconstructed here). */
    for (int j = 0; j < n; j++)
        for (int i = j + 1; i < m; i++)
            MAT(*A, i, j) = 0.f;

    free(w.Vkk); free(w.tau_kk); free(w.V2); free(w.tau2); free(w.S); free(w.S2);
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
    (void)nt; (void)chunk; (void)sched;

    if (m % HVX_FLOAT_LANES != 0) { printf("Error: m must be a multiple of %d\n", HVX_FLOAT_LANES); return 1; }
    if (B < 1)        { printf("Error: nb (tile size B) must be >= 1\n"); return 1; }
    if (m % B != 0)   { printf("Error: B=%d must divide m=%d\n", B, m); return 1; }
    if (n % B != 0)   { printf("Error: B=%d must divide n=%d\n", B, n); return 1; }

    printf("=== %s ===\n", VARIANT_NAME);
    printf("Matrix: %d x %d  |  Tile B: %d  |  Grid: %d x %d tiles\n\n",
           m, n, B, m/B, n/B);

    Matrix A_orig = create_matrix(m, n);
    fill_test_matrix(&A_orig);

    Matrix A_cpu = copy_matrix(&A_orig);
    printf("[ref] CPU scalar Householder        ... ");
    uint64_t t0 = get_cycles();
    qr_cpu_householder(&A_cpu);
    uint64_t t1 = get_cycles();
    printf("%llu cycles\n", (unsigned long long)(t1 - t0));

    Matrix A_alg = copy_matrix(&A_orig);
    printf("[alg] %s ... ", VARIANT_NAME);
    uint64_t t2 = get_cycles();
    tiled_qr_seq(&A_alg, B);
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
