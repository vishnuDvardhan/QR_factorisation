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


//
#define DEFAULT_CHUNK   4

static char thread_stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8)));

typedef HVX_Vector __attribute__((aligned(4))) HVX_VectorU;

/* Column-major: data[j * m + i] */
typedef struct { float* data; int m; int n; } Matrix;
#define MAT(mat, i, j)     ((mat).data[(j) * (mat).m + (i)])
#define MAT_ROW(mat, i, j) ((mat).data[(i) * (mat).n + (j)])

static inline uint64_t get_cycles(void) {
    uint64_t c;
    __asm__ __volatile__("%0 = c15:14" : "=r"(c) : : "memory");
    return c;
}

Matrix create_matrix(int m, int n) {
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
Matrix copy_matrix(Matrix* src) {
    Matrix dst = create_matrix(src->m, src->n);
    memcpy(dst.data, src->data, (size_t)src->m * src->n * sizeof(float));
    return dst;
}
/* Explicit row-major → col-major transpose */
Matrix transpose_row_to_col(Matrix* src) {
    Matrix dst = create_matrix(src->m, src->n);
    for (int i = 0; i < src->m; i++)
        for (int j = 0; j < src->n; j++)
            MAT(dst, i, j) = MAT_ROW(*src, i, j);
    return dst;
}
void free_matrix(Matrix* mat) {
    if (mat->data) { free(mat->data); mat->data = NULL; }
}
void save_matrix(const char* fn, Matrix* mat) {
    FILE* fp = fopen(fn, "w"); if (!fp) return;
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++)
            fprintf(fp, "%9.4f ", (i > j) ? 0.f : MAT(*mat, i, j));
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void householder_qr_cpu(Matrix* A) {
    int m = A->m, n = A->n;
    float* v = (float*)malloc((size_t)m * sizeof(float));
    for (int k = 0; k < n; k++) {
        float nxsq = 0.f;
        for (int i = k; i < m; i++) { float val = MAT(*A,i,k); nxsq += val*val; }
        float nx = sqrtf(nxsq); if (nx < 1e-7f) continue;
        float alpha = -(MAT(*A,k,k) >= 0.f ? 1.f : -1.f) * nx;
        for (int i = k; i < m; i++) v[i] = MAT(*A,i,k); v[k] -= alpha;
        float nvsq = 0.f;
        for (int i = k; i < m; i++) nvsq += v[i]*v[i];
        if (nvsq < 1e-10f) continue;
        float tau = 2.f / nvsq;
        for (int j = k+1; j < n; j++) {
            float dot = 0.f;
            for (int i = k; i < m; i++) dot += v[i] * MAT(*A,i,j);
            float w = tau * dot;
            for (int i = k; i < m; i++) MAT(*A,i,j) -= w * v[i];
        }
        MAT(*A,k,k) = alpha;
        for (int i = k+1; i < m; i++) MAT(*A,i,k) = 0.f;
    }
    free(v);
}


//
typedef struct {
    Matrix*        A;
    const float*   v;
    float          tau;
    int            k;
    int            col_start;    /* SAM: inclusive */
    int            col_end;      /* SAM: exclusive */
    unsigned int*  counter;      /* DAM: shared atomic column-chunk counter */
    int            total_cols;
    int            chunk_size;
    int thread_id;
    int nt;
} ColCtx;


//
void col_worker_dyn(void* arg) {
    ColCtx* ctx = (ColCtx*)arg;
    int m     = ctx->A->m;
    int chunk = ctx->chunk_size;

    while (1) {
        /* Claim chunk_size consecutive columns with one atomic op */
        unsigned int ci  = qurt_atomic_inc_return(ctx->counter) - 1;
        int col_base     = ctx->k + 1 + (int)ci * chunk;
        if (col_base >= ctx->k + 1 + ctx->total_cols) break;
        int col_end_c    = col_base + chunk;
        if (col_end_c > ctx->k + 1 + ctx->total_cols)
            col_end_c = ctx->k + 1 + ctx->total_cols;

        /*
         * ONE HVX lock covers the entire chunk of columns.
         * For each column j in the chunk: dot product then update.
         * Consecutive columns in v[] are likely in the same cache lines.
         */
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) continue;

        for (int j = col_base; j < col_end_c; j++) {
            /* Dot product: v^T * A[:,j] */
            HVX_Vector vSum = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&MAT(*ctx->A, i, j);
                HVX_Vector vv = *(HVX_VectorU*)&ctx->v[i];
                vSum = Q6_Vqf32_vadd_Vqf32Vqf32(vSum, Q6_Vqf32_vmpy_VsfVsf(va, vv));
            }
            /* Horizontal reduction (convert qf32 -> sf first) */
            float temp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
            *(HVX_VectorU*)temp = Q6_Vsf_equals_Vqf32(vSum);
            float dot = 0.f;
            for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) dot += temp[lane];

            /* Column update: A[:,j] -= (tau*dot)*v */
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

/* -----------------------------------------------------------------------
 * SAM worker: pre-assigned column range, chunk_size columns per lock.
 * ----------------------------------------------------------------------- */
void col_worker_sta(void* arg) {
    ColCtx* ctx = (ColCtx*)arg;
    int m     = ctx->A->m;
    int chunk = ctx->chunk_size;
    
    //
    int startIndex = ctx->thread_id * chunk;
    int counter=0;
    

    while (1) {
        /* Claim chunk_size consecutive columns with one atomic op */
        // unsigned int ci  = qurt_atomic_inc_return(ctx->counter) - 1;
        int col_base     = ctx->k + 1 + (startIndex + (ctx->nt * chunk * counter));
        if (col_base >= ctx->k + 1 + ctx->total_cols) break;
        int col_end_c    = col_base + chunk;
        if (col_end_c > ctx->k + 1 + ctx->total_cols)
            col_end_c = ctx->k + 1 + ctx->total_cols;

        /*
         * ONE HVX lock covers the entire chunk of columns.
         * For each column j in the chunk: dot product then update.
         * Consecutive columns in v[] are likely in the same cache lines.
         */
        if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) continue;

        for (int j = col_base; j < col_end_c; j++) {
            /* Dot product: v^T * A[:,j] */
            HVX_Vector vSum = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&MAT(*ctx->A, i, j);
                HVX_Vector vv = *(HVX_VectorU*)&ctx->v[i];
                vSum = Q6_Vqf32_vadd_Vqf32Vqf32(vSum, Q6_Vqf32_vmpy_VsfVsf(va, vv));
            }
            /* Horizontal reduction (convert qf32 -> sf first) */
            float temp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
            *(HVX_VectorU*)temp = Q6_Vsf_equals_Vqf32(vSum);
            float dot = 0.f;
            for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) dot += temp[lane];

            /* Column update: A[:,j] -= (tau*dot)*v */
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

        counter++;
    }
    qurt_thread_exit(0);
}


void col_worker_sta1(void* arg) {
    ColCtx* ctx = (ColCtx*)arg;
    int m     = ctx->A->m;
    int chunk = ctx->chunk_size;

    for (int j = ctx->col_start; j < ctx->col_end; j += chunk) {
        int col_end_c = j + chunk;
        if (col_end_c > ctx->col_end) col_end_c = ctx->col_end;

        if (qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) continue;

        for (int jj = j; jj < col_end_c; jj++) {
            HVX_Vector vSum = Q6_V_vzero();
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&MAT(*ctx->A, i, jj);
                HVX_Vector vv = *(HVX_VectorU*)&ctx->v[i];
                vSum = Q6_Vqf32_vadd_Vqf32Vqf32(vSum, Q6_Vqf32_vmpy_VsfVsf(va, vv));
            }
            float temp[HVX_FLOAT_LANES] __attribute__((aligned(4)));
            *(HVX_VectorU*)temp = Q6_Vsf_equals_Vqf32(vSum);
            float dot = 0.f;
            for (int lane = 0; lane < HVX_FLOAT_LANES; lane++) dot += temp[lane];

            float w = ctx->tau * dot;
            HVX_Vector vw = Q6_V_vsplat_R(*(uint32_t*)&w);
            for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
                HVX_Vector va = *(HVX_VectorU*)&MAT(*ctx->A, i, jj);
                HVX_Vector vv = *(HVX_VectorU*)&ctx->v[i];
                HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, Q6_V_vzero());
                HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vw, vv);
                *(HVX_VectorU*)&MAT(*ctx->A, i, jj) =
                    Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp));
            }
        }

        qurt_hvx_unlock();
    }
    qurt_thread_exit(0);
}

void householder_columns_hvx_mt(Matrix* A, const float* v, float tau,
                                  int k, int nt, int use_static,
                                  int chunk_size) {
    int n = A->n;
    int total_cols = n - k - 1;
    if (total_cols <= 0) return;

    if (nt > MAX_THREADS)  nt = MAX_THREADS;
    if (nt > total_cols)   nt = total_cols;
    if (nt < 1)            nt = 1;
    if (chunk_size < 1)    chunk_size = 1;

    ColCtx         ctx[MAX_THREADS];
    qurt_thread_t  tids[MAX_THREADS];
    qurt_thread_attr_t attrs[MAX_THREADS];

    unsigned int counter = 0;

    /* SAM: ceiling division — chunk_size columns per lock, ranges cover all cols */
    int rpt = (total_cols + nt - 1) / nt;

    for (int t = 0; t < nt; t++) {
        ctx[t].A          = A;
        ctx[t].v          = v;
        ctx[t].tau        = tau;
        ctx[t].k          = k;
        ctx[t].counter    = &counter;
        ctx[t].total_cols = total_cols;
        ctx[t].chunk_size = chunk_size;
        ctx[t].col_start  = k + 1 + t * rpt;
        ctx[t].col_end    = k + 1 + (t + 1) * rpt;
        ctx[t].thread_id = t;
        ctx[t].nt = nt;
        if (ctx[t].col_end > n) ctx[t].col_end = n;

        qurt_thread_attr_init(&attrs[t]);
        qurt_thread_attr_set_stack_addr(&attrs[t], thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attrs[t], STACK_SIZE);
        qurt_thread_attr_set_priority(&attrs[t], 100);

        void (*fn)(void*) = use_static ? col_worker_sta : col_worker_dyn;
        int r = qurt_thread_create(&tids[t], &attrs[t], fn, &ctx[t]);
        if (r != QURT_EOK) {
            printf("thread_create failed: %d\n", r);
            for (int i = 0; i < t; i++) { int s; qurt_thread_join(tids[i], &s); }
            return;
        }
    }
    for (int t = 0; t < nt; t++) { int s; qurt_thread_join(tids[t], &s); }
}

void householder_qr_hvx_mt(Matrix* A, int num_threads, int use_static,
                            int chunk_size) {
    int m = A->m, n = A->n;
    float* v = (float*)memalign(128, (size_t)m * sizeof(float));
    if (!v) { printf("ERROR: alloc v\n"); return; }

    for (int k = 0; k < n; k++) {
        float nxsq = 0.f;
        for (int i = k; i < m; i++) { float val = MAT(*A,i,k); nxsq += val*val; }
        float nx = sqrtf(nxsq);
        if (nx < 1e-7f) continue;

        float alpha = -(MAT(*A,k,k) >= 0.f ? 1.f : -1.f) * nx;
        for (int i = 0; i < k; i++) v[i] = 0.f;
        for (int i = k; i < m; i++) v[i] = MAT(*A,i,k);
        v[k] -= alpha;

        float nvsq = 0.f;
        for (int i = k; i < m; i++) nvsq += v[i]*v[i];
        if (nvsq < 1e-10f) continue;
        float tau = 2.f / nvsq;

        householder_columns_hvx_mt(A, v, tau, k, num_threads, use_static,
                                   chunk_size);
        MAT(*A,k,k) = alpha;
        for (int i = k+1; i < m; i++) MAT(*A,i,k) = 0.f;
    }
    free(v);
}

void verify_results(Matrix* cpu, Matrix* hvx) {
    float max_err = 0.f;
    for (int i = 0; i < cpu->m; i++)
        for (int j = 0; j < cpu->n; j++) {
            float d = fabsf(MAT(*cpu,i,j) - MAT(*hvx,i,j));
            if (d > max_err) max_err = d;
        }
    printf("\nMax absolute error CPU vs HVX-MT: %e\n", max_err);
    puts(max_err < 1e-4f
         ? "VERIFICATION SUCCESSFUL! HVX-MT results match CPU."
         : "VERIFICATION FAILED! Difference too large.");
}

int main(void) {
    FILE* fp = fopen("./../inparams.txt", "r");
    if (!fp) { printf("Error: cannot open inparams.txt\n"); return 1; }

    int m, n, nt, use_static;
    int chunk_size = DEFAULT_CHUNK;

    /* 
    useStatic = 1;
    useDynamic = 0;
    */
    if (fscanf(fp, "%d %d %d %d %d", &m, &n, &nt, &chunk_size, &use_static) != 5) {
        printf("Error: expected: m n threads static(0/1)\n");
        fclose(fp); return 1;
    }
    fclose(fp);

    if (m % HVX_FLOAT_LANES != 0) {
        printf("Error: m must be a multiple of %d\n", HVX_FLOAT_LANES);
        return 1;
    }

    printf("=== Column-Major QR with Multithreading ===\n");
    printf("Matrix: %d x %d  |  Threads: %d  |  Alloc: %s  |  Chunk: %d\n\n",
           m, n, nt, use_static ? "Static" : "Dynamic", chunk_size);

    /* Allocate row-major, then explicitly transpose to column-major */
    Matrix A_row_major = create_matrix(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            MAT_ROW(A_row_major, i, j) = (float)((i*3 + j*7) % 11) - 5.f;

    uint64_t t0 = get_cycles();
    Matrix A_orig = transpose_row_to_col(&A_row_major);
    uint64_t t1 = get_cycles();

    Matrix A_cpu = copy_matrix(&A_orig);
    Matrix A_hvx = copy_matrix(&A_orig);

    // printf("Running CPU reference...\n");
    // householder_qr_cpu(&A_cpu);
    // save_matrix("qr_output_cpu.txt", &A_cpu);

    printf("Running HVX-MT...\n");
    uint64_t t2 = get_cycles();
    householder_qr_hvx_mt(&A_hvx, nt, use_static, chunk_size);
    uint64_t t3 = get_cycles();

    // save_matrix("qr_output_colmajor_mt.txt", &A_hvx);
    // verify_results(&A_cpu, &A_hvx);
    printf("Transpose cycles: %llu\n", (unsigned long long)(t1 - t0));
    printf("QR cycles:        %llu\n", (unsigned long long)(t3 - t2));
    printf("Total cycles:     %llu\n", (unsigned long long)((t1-t0)+(t3-t2)));

    free_matrix(&A_row_major);
    free_matrix(&A_orig);
    free_matrix(&A_cpu);
    free_matrix(&A_hvx);
    return 0;
}