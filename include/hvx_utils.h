/* =====================================================================
 *  hvx_utils.h — shared HVX helpers for the Householder QR variants.
 *
 *  Build once, reuse across every variant (v1..v6).  All routines assume:
 *    - column-major float matrices, m a multiple of HVX_FLOAT_LANES (32)
 *    - the caller already holds an HVX context (qurt_hvx_lock succeeded)
 *    - qf32 accumulation for numerical parity with the v0 baseline
 *
 *  Intrinsic conventions match qr_pool_hh_unblocked.c (the v0 baseline):
 *    Q6_Vqf32_vmpy_VsfVsf, Q6_Vqf32_vadd_Vqf32Vqf32, Q6_Vsf_equals_Vqf32.
 * ===================================================================== */
#ifndef HVX_UTILS_H
#define HVX_UTILS_H

#include <stdint.h>
#include <string.h>
#include "hexagon_types.h"
#include "hvx_hexagon_protos.h"

#ifndef HVX_FLOAT_LANES
#define HVX_FLOAT_LANES 32
#endif

/* Unaligned HVX vector load/store alias (4-byte alignment), matching v0. */
typedef HVX_Vector __attribute__((aligned(4))) HVX_VectorU_t;

/* ---------------------------------------------------------------------
 * hvx_reduce_qf32 — horizontal sum of a 32-lane qf32 accumulator to scalar.
 *
 * Replaces the store-to-array + 32-iteration scalar loop in v0.  No memory
 * traffic in the reduction: 5 byte-rotate + qf32-add stages collapse the
 * 32 partial sums into lane 0.  Q6_V_vror_VR rotates by BYTES, so halving
 * 32 float lanes (4 bytes each) needs rotates of 64,32,16,8,4 bytes.
 * ------------------------------------------------------------------- */
static inline float hvx_reduce_qf32(HVX_Vector vqsum) {
    HVX_Vector v = Q6_Vsf_equals_Vqf32(vqsum);   /* qf32 -> sf */
    HVX_Vector vhi;

    vhi = Q6_V_vror_VR(v, 64); v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));
    vhi = Q6_V_vror_VR(v, 32); v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));
    vhi = Q6_V_vror_VR(v, 16); v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));
    vhi = Q6_V_vror_VR(v,  8); v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));
    vhi = Q6_V_vror_VR(v,  4); v = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(v, vhi));

    float result;
    memcpy(&result, &v, sizeof(float));   /* lane 0 holds the full sum */
    return result;
}

/* ---------------------------------------------------------------------
 * hvx_col_norm_sq — ||col[k_start:m]||^2 in HVX.
 *
 * Lanes below k_start within the first partial vector contribute zero only
 * if k_start is a multiple of HVX_FLOAT_LANES.  Pivot columns satisfy this
 * (we iterate i from k_start in lane-aligned strides); for a general start
 * the caller must guarantee alignment.  Squaring is exact for the masked
 * region because v0 always zeroes the sub-pivot entries before this call.
 * ------------------------------------------------------------------- */
static inline float hvx_col_norm_sq(const float* col, int m, int k_start) {
    HVX_Vector vacc = Q6_V_vzero();
    for (int i = k_start; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector vc = *(const HVX_VectorU_t*)(col + i);
        vacc = Q6_Vqf32_vadd_Vqf32Vqf32(vacc, Q6_Vqf32_vmpy_VsfVsf(vc, vc));
    }
    return hvx_reduce_qf32(vacc);
}

/* ||v[k_start:m]||^2 — identical operation, named for call-site clarity. */
static inline float hvx_vec_norm_sq(const float* v, int m, int k_start) {
    return hvx_col_norm_sq(v, m, k_start);
}

/* ---------------------------------------------------------------------
 * hvx_dot_col_v — dot(col_j[0:m], v[0:m]) fully in HVX.
 *
 * Replaces the qf32 accumulation + store + 32-iteration scalar reduction in
 * v0's pool_do_col_hh.  Returns the scalar dot product.
 * ------------------------------------------------------------------- */
static inline float hvx_dot_col_v(const float* col_j, const float* v, int m) {
    HVX_Vector vacc = Q6_V_vzero();
    for (int i = 0; i < m; i += HVX_FLOAT_LANES) {
        HVX_Vector va = *(const HVX_VectorU_t*)(col_j + i);
        HVX_Vector vv = *(const HVX_VectorU_t*)(v + i);
        vacc = Q6_Vqf32_vadd_Vqf32Vqf32(vacc, Q6_Vqf32_vmpy_VsfVsf(va, vv));
    }
    return hvx_reduce_qf32(vacc);
}

#endif /* HVX_UTILS_H */
