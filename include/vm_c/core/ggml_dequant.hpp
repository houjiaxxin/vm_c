#pragma once
// GGML 量化类型反量化函数（从 llama.cpp 移植）
// 将 block-quantized 数据反量化为 float 数组
// 依赖: gguf_reader.hpp 提供 GgmlType 枚举 + QK_K / QK4_NL 常量

#include "vm_c/core/gguf_reader.hpp"
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace vm_c {

// ── 基础 block 大小常量 ──
constexpr int QK4_0   = 32;   // Q4_0 block size
constexpr int QK4_1   = 32;
constexpr int QK5_0   = 32;
constexpr int QK5_1   = 32;
constexpr int QK8_0   = 32;

// ── Block structures (来自 llama.cpp ggml-common.h) ──

#pragma pack(push, 1)

// fp16 ↔ float 转换
inline float fp16_to_float(uint16_t h) {
    // IEEE 754 half-precision → float
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (0x7F - 15) << 23 | (mant << 13);
    } else if (exp == 0x1F) {
        f = (sign << 31) | 0xFF << 23 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(float));
    return result;
}

struct block_q4_0 {
    uint16_t d;            // fp16 scale
    uint8_t qs[QK4_0 / 2];
};

struct block_q4_1 {
    uint16_t d;            // fp16 scale
    uint16_t m;            // fp16 min
    uint8_t qs[QK4_1 / 2];
};

struct block_q5_0 {
    uint16_t d;            // fp16 scale
    uint8_t qh[4];         // high bits
    uint8_t qs[QK5_0 / 2];
};

struct block_q5_1 {
    uint16_t d;            // fp16 scale
    uint16_t m;            // fp16 min
    uint8_t qh[4];         // high bits
    uint8_t qs[QK5_1 / 2];
};

struct block_q8_0 {
    uint16_t d;            // fp16 scale
    int8_t  qs[QK8_0];
};

struct block_q2_K {
    uint8_t scales[QK_K / 16];
    uint8_t qs[QK_K / 4];
    uint16_t d;            // fp16 scale
    uint16_t dmin;         // fp16 min
};

struct block_q3_K {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    uint16_t d;            // fp16 scale
};

struct block_q4_K {
    uint16_t d;            // fp16 scale
    uint16_t dmin;         // fp16 min
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};

struct block_q5_K {
    uint16_t d;            // fp16 scale
    uint16_t dmin;         // fp16 min
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    uint16_t d;            // fp16 scale
};

struct block_q8_K {
    uint16_t d;            // fp16 scale
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

#pragma pack(pop)

// ── 反量化函数 ──
// 命名: dequantize_row_{type}(block_data, output_float_array, num_elements)

void dequantize_row_q4_0(const void* x, float* y, int64_t k);
void dequantize_row_q4_1(const void* x, float* y, int64_t k);
void dequantize_row_q5_0(const void* x, float* y, int64_t k);
void dequantize_row_q5_1(const void* x, float* y, int64_t k);
void dequantize_row_q8_0(const void* x, float* y, int64_t k);

void dequantize_row_q2_K(const void* x, float* y, int64_t k);
void dequantize_row_q3_K(const void* x, float* y, int64_t k);
void dequantize_row_q4_K(const void* x, float* y, int64_t k);
void dequantize_row_q5_K(const void* x, float* y, int64_t k);
void dequantize_row_q6_K(const void* x, float* y, int64_t k);
void dequantize_row_q8_K(const void* x, float* y, int64_t k);

} // namespace vm_c
