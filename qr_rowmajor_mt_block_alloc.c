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




/*
profile: 
    -> 1) build_qurt qr_rowmajor_mt_block_alloc.c --profile
    -> 2) hexagon-gprof qr_rowmajor_mt_block_alloc gmon.t* > gprof_full_report.txt
*/

QURT_DECLARE_STATIC_HEAP(16777216 * 4);
unsigned int heapSize = 16777216;

#define HVX_FLOAT_LANES 32
#define MAX_THREADS     8
#define STACK_SIZE      (16 * 1024 * 4)

#define DEFAULT_CHUNK   4

static char thread_stacks[MAX_THREADS][STACK_SIZE] __attribute__((aligned(8)));

typedef HVX_Vector __attribute__((aligned(4))) HVX_VectorU;

typedef struct { float* data; int m; int n; } Matrix;
#define MAT(mat, i, j) ((mat).data[(i) * (mat).n + (j)])

static inline uint64_t get_cycles(void) {
    uint64_t c;
    __asm__ __volatile__("%0 = c15:14" : "=r"(c) : : "memory");
    return c;
}

/* =========================================================================
 * MATRIX UTILITIES
 * ========================================================================= */
Matrix create_matrix(int m, int n) {
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

void free_matrix(Matrix* mat) {
    if (mat->data) { free(mat->data); mat->data = NULL; }
}

void save_matrix(const char* fn, Matrix* mat) {
    FILE* fp = fopen(fn, "w"); if (!fp) return;
    for (int i = 0; i < mat->m; i++) {
        for (int j = 0; j < mat->n; j++)
            fprintf(fp, "%9.4f ", (i > j) ? 0.f : MAT(*mat,i,j));
        fprintf(fp, "\n");
    }
    fclose(fp);
}

/* =========================================================================
 * CPU REFERENCE IMPLEMENTATION
 * ========================================================================= */
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

/* =========================================================================
 * OPTIMIZED SHARED CONTEXT
 * ========================================================================= */
typedef struct {
    Matrix* A;
    const float* v;
    float tau;
    int k;
    int j_start;
    int total_rows;
    int n;
    int HVX_LANE_ITERs;
    int vec_iters;
    int chunk_size;
    int num_threads;
    unsigned int dynamic_block_ctr; 
    qurt_mutex_t dam_mutex;
} SharedCtx;

typedef struct {
    SharedCtx* shared;
    int thread_id;
} ThreadCtx;

/* =========================================================================
 * CORE HVX BLOCK PROCESSOR (NO THREAD BARRIERS NEEDED)
 * ========================================================================= */
static inline void process_hvx_block(SharedCtx* ctx, int block) {
    int j = ctx->j_start + block * HVX_FLOAT_LANES;
    int remainingCols = 0;

    // Handle out-of-bounds remainder block
    if (block >= ctx->vec_iters) {
        int remStart = ctx->n - HVX_FLOAT_LANES;
        if (remStart < 0) remStart = 0;
        j = remStart;
        remainingCols = 1;
    }

    float local_acc[HVX_FLOAT_LANES] __attribute__((aligned(4)));
    memset(local_acc, 0, sizeof(local_acc));

    // Request HVX lock for the entire block processing (Phase 1 & 2)
    if(qurt_hvx_lock(QURT_HVX_MODE_128B) != 0) return;

    /* ── PHASE 1: Calculate Dot Product for the Column Block ──
     * Accumulate in qf32; sf*sf→qf32 multiply, qf32+qf32 accumulate.
     * Initial acc bit-pattern is all zeros, valid as qf32 zero. */
    HVX_Vector acc = *(HVX_VectorU*)local_acc;
    for (int row = ctx->k; row < ctx->k + ctx->total_rows; row++) {
        HVX_Vector va = *(HVX_VectorU*)&MAT(*ctx->A, row, j);
        HVX_Vector vv = Q6_V_vsplat_R(*(uint32_t*)&ctx->v[row]);
        acc = Q6_Vqf32_vadd_Vqf32Vqf32(acc, Q6_Vqf32_vmpy_VsfVsf(vv, va));
    }

    // Multiply by tau (convert acc qf32→sf so the masking below sees floats).
    HVX_Vector acc_sf = Q6_Vsf_equals_Vqf32(acc);
    HVX_Vector vt = Q6_V_vsplat_R(*(uint32_t*)&ctx->tau);
    HVX_Vector vSum = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(acc_sf, vt));

    /* ── Robust Out-Of-Bounds Zeroing ── */
    if (remainingCols) {
        float vSum_local[HVX_FLOAT_LANES] __attribute__((aligned(4)));
        *(HVX_VectorU*)vSum_local = vSum;
        int target_start = ctx->j_start + ctx->vec_iters * HVX_FLOAT_LANES;
        
        for (int idx = 0; idx < HVX_FLOAT_LANES; idx++) {
            int col = j + idx;
            if (col < target_start || col >= ctx->n) {
                vSum_local[idx] = 0.f;
            }
        }
        vSum = *(HVX_VectorU*)vSum_local;
    }

    /* ── PHASE 2: Update the Matrix Columns ── */
    for (int row = ctx->k; row < ctx->k + ctx->total_rows; row++) {
        HVX_Vector va    = *(HVX_VectorU*)&MAT(*ctx->A, row, j);
        HVX_Vector vv    = Q6_V_vsplat_R(*(uint32_t*)&ctx->v[row]);
        HVX_Vector va_qf = Q6_Vqf32_vadd_VsfVsf(va, Q6_V_vzero());
        HVX_Vector vp    = Q6_Vqf32_vmpy_VsfVsf(vv, vSum);
        *(HVX_VectorU*)&MAT(*ctx->A, row, j) =
            Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(va_qf, vp));
    }

    qurt_hvx_unlock();
}

/* =========================================================================
 * STATIC WORKER (SAM)
 * ========================================================================= */
void panel_worker_static(void* arg) {
    ThreadCtx* tctx = (ThreadCtx*)arg;
    SharedCtx* ctx = tctx->shared;
    int tid = tctx->thread_id;
    int iter = 0;

    while (1) {
        // Striped chunk allocation
        int start_block = (iter * ctx->num_threads + tid) * ctx->chunk_size;
        if (start_block >= ctx->HVX_LANE_ITERs) break;

        int end_block = start_block + ctx->chunk_size;
        if (end_block > ctx->HVX_LANE_ITERs) end_block = ctx->HVX_LANE_ITERs;

        for (int b = start_block; b < end_block; b++) {
            process_hvx_block(ctx, b);
        }
        iter++;
    }
    qurt_thread_exit(0);
}

/* =========================================================================
 * DYNAMIC WORKER (DAM)
 * ========================================================================= */
void panel_worker_dynamic(void* arg) {
    ThreadCtx* tctx = (ThreadCtx*)arg;
    SharedCtx* ctx = tctx->shared;

    while (1) {
        // Thread-safe dynamic chunk allocation
        qurt_mutex_lock(&ctx->dam_mutex);
        int start_block = ctx->dynamic_block_ctr;
        ctx->dynamic_block_ctr += ctx->chunk_size;
        qurt_mutex_unlock(&ctx->dam_mutex);

        if (start_block >= ctx->HVX_LANE_ITERs) break;

        int end_block = start_block + ctx->chunk_size;
        if (end_block > ctx->HVX_LANE_ITERs) end_block = ctx->HVX_LANE_ITERs;

        for (int b = start_block; b < end_block; b++) {
            process_hvx_block(ctx, b);
        }
    }
    qurt_thread_exit(0);
}

/* =========================================================================
 * THREAD DISPATCHER
 * ========================================================================= */
void hvx_apply_householder_panel_mt(Matrix* A, const float* v, float tau, int k, int nt, int use_static, int chunk_size) {
    int m = A->m, n = A->n;
    int j_start = k + 1;
    if (n - j_start <= 0) return;

    int total_cols = n - j_start;
    int vec_iters = total_cols / HVX_FLOAT_LANES;          
    int has_rem = (total_cols % HVX_FLOAT_LANES) != 0;
    int HVX_LANE_ITERs = vec_iters + has_rem;                  
    int total_rows = m - k;

    if (nt > MAX_THREADS) nt = MAX_THREADS;
    if (nt > HVX_LANE_ITERs) nt = HVX_LANE_ITERs; // No need for more threads than blocks
    if (nt < 1) nt = 1;
    if (chunk_size < 1) chunk_size = 1;

    // Initialize Shared Context
    SharedCtx shared;
    shared.A = A;
    shared.v = v;
    shared.tau = tau;
    shared.k = k;
    shared.j_start = j_start;
    shared.total_rows = total_rows;
    shared.n = n;
    shared.HVX_LANE_ITERs = HVX_LANE_ITERs;
    shared.vec_iters = vec_iters;
    shared.chunk_size = chunk_size;
    shared.num_threads = nt;
    shared.dynamic_block_ctr = 0;
    qurt_mutex_init(&shared.dam_mutex);

    // Optimized Thread Context Array
    ThreadCtx tctx[MAX_THREADS];
    qurt_thread_t tids[MAX_THREADS];
    qurt_thread_attr_t attrs[MAX_THREADS];

    for (int t = 0; t < nt; t++) {
        tctx[t].shared = &shared;
        tctx[t].thread_id = t;

        qurt_thread_attr_init(&attrs[t]);
        qurt_thread_attr_set_stack_addr(&attrs[t], thread_stacks[t]);
        qurt_thread_attr_set_stack_size(&attrs[t], STACK_SIZE);
        qurt_thread_attr_set_priority(&attrs[t], 100);

        void (*fn)(void*) = use_static ? panel_worker_static : panel_worker_dynamic;
        int r = qurt_thread_create(&tids[t], &attrs[t], fn, &tctx[t]);
        if (r != QURT_EOK) {
            printf("thread_create failed: %d\n", r);
            for (int i = 0; i < t; i++) { int s; qurt_thread_join(tids[i], &s); }
            qurt_mutex_destroy(&shared.dam_mutex);
            return;
        }
    }
    
    // Join all threads
    for (int t = 0; t < nt; t++) { 
        int s; 
        qurt_thread_join(tids[t], &s); 
    }
    qurt_mutex_destroy(&shared.dam_mutex);
}

void householder_qr_hvx_mt(Matrix* A, int num_threads, int use_static, int chunk_size) {
    
    
    int m = A->m, n = A->n;
    float v[m] __attribute__((aligned(128)));

    for (int k = 0; k < n; k++) {
        
        
        float nxsq = 0.f;
        for (int i = k; i < m; i++) { 
            float val = MAT(*A, i, k);
            v[i] = val;
            nxsq += val * val; 
        }
        
        float nx = sqrtf(nxsq); 
        if (nx < 1e-7f) {
            continue;
        }

        float x_k = v[k]; 
        float alpha = -(x_k >= 0.f ? 1.f : -1.f) * nx;
        
        v[k] -= alpha;
        float nvsq = nxsq - (x_k * x_k) + (v[k] * v[k]);
        
        if (nvsq < 1e-10f) {
            continue;
        }
        
        float tau = 2.f / nvsq;

        // Dispatch call tracks its own cycles internally
        hvx_apply_householder_panel_mt(A, v, tau, k, num_threads, use_static, chunk_size);
        
        MAT(*A, k, k) = alpha;
        for (int i = k+1; i < m; i++) MAT(*A,i,k) = 0.f;
    }
    
}

void verify_results(Matrix* cpu, Matrix* hvx) {
    float max_err = 0.f;
    for (int i = 0; i < cpu->m; i++)
        for (int j = 0; j < cpu->n; j++) {
            float d = fabsf(MAT(*cpu,i,j) - MAT(*hvx,i,j));
            if (d > max_err) max_err = d;
        }
    printf("\nMax absolute error CPU vs HVX-MT: %e\n", max_err);
    puts(max_err < 1e-3f
         ? "VERIFICATION SUCCESSFUL! HVX-MT results match CPU."
         : "VERIFICATION FAILED! Difference too large.");
}

int main(void) {
    FILE* fp = fopen("./../inparams1.txt", "r");
    if (!fp) { printf("Error: cannot open inparams.txt\n"); return 1; }

    int m, n, nt, use_static;
    int chunk_size = DEFAULT_CHUNK;

    if (fscanf(fp, "%d %d %d %d %d", &m, &n, &nt, &chunk_size, &use_static) != 5) {
        printf("Error: expected: m n threads static(0/1) chunk_size\n");
        fclose(fp); return 1;
    }
    fclose(fp);

    printf("=== Block-Level Row-Major QR with Multithreading ===\n");
    printf("Matrix: %d x %d  |  Threads: %d  |  Alloc: %s  |  Chunk: %d\n\n",
           m, n, nt, use_static ? "Static" : "Dynamic", chunk_size);

    Matrix A_orig = create_matrix(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            MAT(A_orig,i,j) = (float)((i*3 + j*7) % 11) - 5.f;

    Matrix A_cpu = copy_matrix(&A_orig);
    Matrix A_hvx = copy_matrix(&A_orig);

    // printf("Running CPU reference...\n");
    // householder_qr_cpu(&A_cpu);
    // save_matrix("qr_output_cpu.txt", &A_cpu);

    printf("Running HVX-MT...\n");
    uint64_t t0 = get_cycles();
    householder_qr_hvx_mt(&A_hvx, nt, use_static, chunk_size);
    uint64_t t1 = get_cycles();

    // save_matrix("qr_output_rowmajor_mt.txt", &A_hvx);
    // verify_results(&A_cpu, &A_hvx);
    printf("Total cycles:     %llu\n", (unsigned long long)(t1 - t0));

    free_matrix(&A_orig);
    free_matrix(&A_cpu);
    free_matrix(&A_hvx);
    return 0;
}