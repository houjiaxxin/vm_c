#include "vm_c/core/ggml_dequant.hpp"
#include <cstring>
#include <cassert>
#include <cmath>

namespace vm_c {

// ── half → float ──
static float half_to_float(uint16_t h) {
    return fp16_to_float(h); // defined in header
}

// ── Q4_K / Q5_K scale helper (from llama.cpp ggml-quants.c) ──
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q4_0 — 4-bit, symmetric, 32-element blocks
// Imported from llama.cpp ggml-quants.c (dequantize_row_q4_0)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q4_0(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q4_0*)vx;
    const int nb = (int)(k / QK4_0);
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;
            y[i * QK4_0 + j + 0   ] = x0 * d;
            y[i * QK4_0 + j + QK4_0 / 2] = x1 * d;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q4_1 — 4-bit, asymmetric (with min), 32-element blocks
// Imported from llama.cpp ggml-quants.c (dequantize_row_q4_1)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q4_1(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q4_1*)vx;
    const int nb = (int)(k / QK4_1);
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const float m = half_to_float(x[i].m);
        for (int j = 0; j < QK4_1 / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F);
            const int x1 = (x[i].qs[j] >>   4);
            y[i * QK4_1 + j + 0   ] = x0 * d + m;
            y[i * QK4_1 + j + QK4_1 / 2] = x1 * d + m;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q5_0 — 5-bit, symmetric, 32-element blocks
// Imported from llama.cpp ggml-quants.c (dequantize_row_q5_0)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q5_0(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q5_0*)vx;
    const int nb = (int)(k / QK5_0);
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < QK5_0 / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int32_t x0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((x[i].qs[j] >>   4) | xh_1) - 16;
            y[i * QK5_0 + j + 0   ] = x0 * d;
            y[i * QK5_0 + j + QK5_0 / 2] = x1 * d;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q5_1 — 5-bit, asymmetric (with min), 32-element blocks
// Imported from llama.cpp ggml-quants.c (dequantize_row_q5_1)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q5_1(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q5_1*)vx;
    const int nb = (int)(k / QK5_1);
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const float m = half_to_float(x[i].m);
        uint32_t qh;
        memcpy(&qh, x[i].qh, sizeof(qh));
        for (int j = 0; j < QK5_1 / 2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
            const int x0 = (x[i].qs[j] & 0x0F) | xh_0;
            const int x1 = (x[i].qs[j] >>   4) | xh_1;
            y[i * QK5_1 + j + 0   ] = x0 * d + m;
            y[i * QK5_1 + j + QK5_1 / 2] = x1 * d + m;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q8_0 — 8-bit, symmetric, 32-element blocks
// Imported from llama.cpp ggml-quants.c (dequantize_row_q8_0)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q8_0(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q8_0*)vx;
    const int nb = (int)(k / QK8_0);
    for (int i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        for (int j = 0; j < QK8_0; ++j) {
            y[i * QK8_0 + j] = x[i].qs[j] * d;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q2_K — 2-bit K-quant (QK_K=256)
// Imported from llama.cpp ggml-quants.c (dequantize_row_q2_K)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q2_K(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q2_K*)vx;
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const float d   = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        const uint8_t* q = x[i].qs;
        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;
                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;
                shift += 2;
            }
            q += 32;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q3_K — 3-bit K-quant (QK_K=256)
// Imported from llama.cpp ggml-quants.c (dequantize_row_q3_K)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q3_K(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q3_K*)vx;
    const int nb = (int)(k / QK_K);
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {
        const float d_all = half_to_float(x[i].d);
        const uint8_t* q  = x[i].qs;
        const uint8_t* hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q4_K — 4-bit K-quant (QK_K=256)
// Imported from llama.cpp ggml-quants.c (dequantize_row_q4_K)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q4_K(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q4_K*)vx;
    const int nb = (int)(k / QK_K);
    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        const float d   = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q5_K — 5-bit K-quant (QK_K=256)
// Imported from llama.cpp ggml-quants.c (dequantize_row_q5_K)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q5_K(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q5_K*)vx;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d   = half_to_float(x[i].d);
        const float min = half_to_float(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q6_K — 6-bit K-quant (QK_K=256, TYPE 14)
// Imported from llama.cpp ggml-quants.c (dequantize_row_q6_K)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q6_K(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q6_K*)vx;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Q8_K — 8-bit K-quant (QK_K=256)
// Imported from llama.cpp ggml-quants.c (dequantize_row_q8_K)
// ═══════════════════════════════════════════════════════════════════════════
void dequantize_row_q8_K(const void* vx, float* y, int64_t k) {
    const auto* x = (const block_q8_K*)vx;
    const int64_t nb = k / QK_K;
    for (int64_t i = 0; i < nb; i++) {
        const float d = half_to_float(x[i].d);
        for (int j = 0; j < QK_K; ++j) {
            *y++ = d * x[i].qs[j];
        }
    }
}

} // namespace vm_c
