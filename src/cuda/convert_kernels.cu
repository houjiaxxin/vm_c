// ── GPU 端权重格式转换 kernel ──
// 全部模板化，自动适配输出类型：__half (FP16) 或 __nv_bfloat16 (BF16)
// 替代 CPU 上的反量化 + F32→FP16 转换，利用 GPU 大规模并行加速

#include "vm_c/core/ggml_dequant.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace vm_c {

// ============================================================
// 类型转换辅助
// ============================================================
// FP16 uint16_t → float（device 版，不依赖 CPU 头文件的 fp16_to_float）
__device__ inline float fp16_to_float_dev(uint16_t h) {
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
    return __uint_as_float(f);  // CUDA intrinsics — 安全、无别名问题
}

// BF16 uint16_t → float
__device__ inline float bf16_raw_to_float(uint16_t bf16) {
    uint32_t fval = static_cast<uint32_t>(bf16) << 16;
    return __uint_as_float(fval);
}

// ============================================================
// F32 → out_t 转换（out_t = __half 或 __nv_bfloat16）
// ============================================================
template <typename out_t>
__global__ void convert_f32_kernel(
    out_t* __restrict__ out,
    const float* __restrict__ in,
    int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    out[idx] = out_t(in[idx]);
}

// ============================================================
// BF16 raw → out_t 转换
// ============================================================
template <typename out_t>
__global__ void convert_bf16_kernel(
    out_t* __restrict__ out,
    const uint16_t* __restrict__ in,
    int64_t num_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;
    float f = bf16_raw_to_float(in[idx]);
    out[idx] = out_t(f);
}

// ============================================================
// 量化格式 → out_t 反量化 kernel（模板化输出类型）
// 每个 block 处理一个 GGML 量化 block
// ============================================================

// Q4_0
template <typename out_t>
__global__ void dequant_q4_0_kernel(
    out_t* __restrict__ out,
    const block_q4_0* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d = fp16_to_float_dev(b.d);
    #pragma unroll
    for (int i = 0; i < QK4_0 / 2; ++i) {
        const float x0 = (float)((int)(b.qs[i] & 0x0F) - 8) * d;
        const float x1 = (float)((int)(b.qs[i] >> 4) - 8) * d;
        const int64_t base = blk * QK4_0 + i * 2;
        out[base + 0] = out_t(x0);
        out[base + 1] = out_t(x1);
    }
}

// Q4_1
template <typename out_t>
__global__ void dequant_q4_1_kernel(
    out_t* __restrict__ out,
    const block_q4_1* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d = fp16_to_float_dev(b.d);
    const float m = fp16_to_float_dev(b.m);
    #pragma unroll
    for (int i = 0; i < QK4_1 / 2; ++i) {
        const float x0 = (float)(b.qs[i] & 0x0F) * d + m;
        const float x1 = (float)(b.qs[i] >> 4) * d + m;
        const int64_t base = blk * QK4_1 + i * 2;
        out[base + 0] = out_t(x0);
        out[base + 1] = out_t(x1);
    }
}

// Q5_0
template <typename out_t>
__global__ void dequant_q5_0_kernel(
    out_t* __restrict__ out,
    const block_q5_0* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d = fp16_to_float_dev(b.d);
    uint32_t qh;
    memcpy(&qh, b.qh, sizeof(qh));
    #pragma unroll
    for (int i = 0; i < QK5_0 / 2; ++i) {
        const uint32_t xh_0 = ((qh >> (2 * i + 0)) & 1) << 4;
        const uint32_t xh_1 = ((qh >> (2 * i + 1)) & 1) << 4;
        const float x0 = (float)((int)(b.qs[i] & 0x0F) | (int)xh_0 - 16) * d;
        const float x1 = (float)((int)(b.qs[i] >> 4) | (int)xh_1 - 16) * d;
        const int64_t base = blk * QK5_0 + i * 2;
        out[base + 0] = out_t(x0);
        out[base + 1] = out_t(x1);
    }
}

// Q5_1
template <typename out_t>
__global__ void dequant_q5_1_kernel(
    out_t* __restrict__ out,
    const block_q5_1* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d = fp16_to_float_dev(b.d);
    const float m = fp16_to_float_dev(b.m);
    uint32_t qh;
    memcpy(&qh, b.qh, sizeof(qh));
    #pragma unroll
    for (int i = 0; i < QK5_1 / 2; ++i) {
        const uint32_t xh_0 = ((qh >> (2 * i + 0)) & 1) << 4;
        const uint32_t xh_1 = ((qh >> (2 * i + 1)) & 1) << 4;
        const float x0 = (float)((b.qs[i] & 0x0F) | (int)xh_0) * d + m;
        const float x1 = (float)((b.qs[i] >> 4) | (int)xh_1) * d + m;
        const int64_t base = blk * QK5_1 + i * 2;
        out[base + 0] = out_t(x0);
        out[base + 1] = out_t(x1);
    }
}

// Q8_0
template <typename out_t>
__global__ void dequant_q8_0_kernel(
    out_t* __restrict__ out,
    const block_q8_0* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d = fp16_to_float_dev(b.d);
    const int64_t base = blk * QK8_0;
    #pragma unroll
    for (int i = 0; i < QK8_0; ++i) {
        out[base + i] = out_t((float)b.qs[i] * d);
    }
}

// Q2_K
template <typename out_t>
__global__ void dequant_q2_K_kernel(
    out_t* __restrict__ out,
    const block_q2_K* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d  = fp16_to_float_dev(b.d);
    const float dmin = fp16_to_float_dev(b.dmin);
    const uint8_t* q = b.qs;
    const uint8_t* sc = b.scales;
    const int64_t base_out = blk * QK_K;
    for (int i = 0; i < QK_K / 16; ++i) {
        const float dl = d  * (float)(sc[i] & 0xF);
        const float ml = dmin * (float)(sc[i] >> 4);
        for (int j = 0; j < 4; ++j) {
            const int o = (i * 4 + j) * 4;
            const uint8_t v = q[o / 2];
            out[base_out + o + 0] = out_t(((float)((v >> 0) & 3)) * dl - ml);
            out[base_out + o + 1] = out_t(((float)((v >> 2) & 3)) * dl - ml);
            out[base_out + o + 2] = out_t(((float)((v >> 4) & 3)) * dl - ml);
            out[base_out + o + 3] = out_t(((float)((v >> 6) & 3)) * dl - ml);
        }
    }
}

// Q3_K
template <typename out_t>
__global__ void dequant_q3_K_kernel(
    out_t* __restrict__ out,
    const block_q3_K* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d = fp16_to_float_dev(b.d);
    const uint8_t* q = b.qs;
    const uint8_t* hm = b.hmask;
    const uint16_t* sc = (const uint16_t*)b.scales;
    const int64_t base_out = blk * QK_K;
    for (int i = 0; i < QK_K / 16; ++i) {
        int16_t scale = (sc[i / 2] >> ((i % 2) * 8)) & 0xFF;
        if (scale > 63) scale -= 128;
        const float dl = d * (float)scale;
        for (int j = 0; j < 4; ++j) {
            const int offset = (i * 4 + j) * 4;
            const uint8_t v = q[offset / 2];
            int x0 = (v >> 0) & 3;
            int x1 = (v >> 2) & 3;
            int x2 = (v >> 4) & 3;
            int x3 = (v >> 6) & 3;
            if (hm[offset / 2] & (1 << (offset % 2 * 2 + 0))) x0 |= 4;
            if (hm[offset / 2] & (1 << (offset % 2 * 2 + 1))) x1 |= 4;
            if (hm[(offset + 2) / 2] & (1 << ((offset + 2) % 2 * 2 + 0))) x2 |= 4;
            if (hm[(offset + 2) / 2] & (1 << ((offset + 2) % 2 * 2 + 1))) x3 |= 4;
            out[base_out + offset + 0] = out_t(((float)(x0 - 4)) * dl);
            out[base_out + offset + 1] = out_t(((float)(x1 - 4)) * dl);
            out[base_out + offset + 2] = out_t(((float)(x2 - 4)) * dl);
            out[base_out + offset + 3] = out_t(((float)(x3 - 4)) * dl);
        }
    }
}

// Q4_K
template <typename out_t>
__global__ void dequant_q4_K_kernel(
    out_t* __restrict__ out,
    const block_q4_K* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d  = fp16_to_float_dev(b.d);
    const float dmin = fp16_to_float_dev(b.dmin);
    const uint8_t* q = b.qs;
    const uint8_t* sc = b.scales;
    const int64_t base_out = blk * QK_K;
    for (int i = 0; i < QK_K / 32; ++i) {
        const float dl = d  * (float)(sc[i] & 0xF);
        const float ml = dmin * (float)(sc[i] >> 4);
        for (int j = 0; j < 8; ++j) {
            const int offset = (i * 8 + j) * 4;
            const uint8_t v0 = q[offset / 2];
            const uint8_t v1 = q[(offset + 2) / 2];
            out[base_out + offset + 0] = out_t(((float)((v0 >> 0) & 0xF)) * dl - ml);
            out[base_out + offset + 1] = out_t(((float)((v0 >> 4) & 0xF)) * dl - ml);
            out[base_out + offset + 2] = out_t(((float)((v1 >> 0) & 0xF)) * dl - ml);
            out[base_out + offset + 3] = out_t(((float)((v1 >> 4) & 0xF)) * dl - ml);
        }
    }
}

// Q5_K
template <typename out_t>
__global__ void dequant_q5_K_kernel(
    out_t* __restrict__ out,
    const block_q5_K* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d  = fp16_to_float_dev(b.d);
    const float dmin = fp16_to_float_dev(b.dmin);
    const uint8_t* q = b.qs;
    const uint8_t* qh = b.qh;
    const uint8_t* sc = b.scales;
    const int64_t base_out = blk * QK_K;
    for (int i = 0; i < QK_K / 32; ++i) {
        const float dl = d  * (float)(sc[i] & 0xF);
        const float ml = dmin * (float)(sc[i] >> 4);
        for (int j = 0; j < 8; ++j) {
            const int offset = (i * 8 + j) * 4;
            const uint8_t v0 = q[offset / 2];
            const uint8_t v1 = q[(offset + 2) / 2];
            int x0 = (v0 >> 0) & 0xF, x1 = (v0 >> 4) & 0xF;
            int x2 = (v1 >> 0) & 0xF, x3 = (v1 >> 4) & 0xF;
            const int h_off = offset / 4;
            const uint8_t hb = qh[h_off / 2];
            const int h_shift = (h_off % 2) * 4;
            if (hb & (1 << (h_shift + 0))) x0 |= 16;
            if (hb & (1 << (h_shift + 1))) x1 |= 16;
            if (hb & (1 << (h_shift + 2))) x2 |= 16;
            if (hb & (1 << (h_shift + 3))) x3 |= 16;
            out[base_out + offset + 0] = out_t((float)x0 * dl - ml);
            out[base_out + offset + 1] = out_t((float)x1 * dl - ml);
            out[base_out + offset + 2] = out_t((float)x2 * dl - ml);
            out[base_out + offset + 3] = out_t((float)x3 * dl - ml);
        }
    }
}

// Q6_K
template <typename out_t>
__global__ void dequant_q6_K_kernel(
    out_t* __restrict__ out,
    const block_q6_K* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d = fp16_to_float_dev(b.d);
    const uint8_t* ql = b.ql;
    const uint8_t* qh = b.qh;
    const int8_t* sc = b.scales;
    const int64_t base_out = blk * QK_K;
    for (int i = 0; i < QK_K / 16; ++i) {
        const float dl = d * (float)sc[i];
        for (int j = 0; j < 4; ++j) {
            const int offset = (i * 4 + j) * 4;
            const uint8_t q0 = ql[offset / 2];
            int x0 = (q0 >> 0) & 3, x1 = (q0 >> 2) & 3;
            int x2 = (q0 >> 4) & 3, x3 = (q0 >> 6) & 3;
            const uint8_t h = qh[offset / 4];
            const int h_shift = (offset % 4) * 2;
            x0 |= ((h >> (h_shift + 0)) & 1) << 2;
            x1 |= ((h >> (h_shift + 1)) & 1) << 2;
            x2 |= ((h >> (h_shift + 2)) & 1) << 2;
            x3 |= ((h >> (h_shift + 3)) & 1) << 2;
            out[base_out + offset + 0] = out_t(((float)(x0 - 4)) * dl);
            out[base_out + offset + 1] = out_t(((float)(x1 - 4)) * dl);
            out[base_out + offset + 2] = out_t(((float)(x2 - 4)) * dl);
            out[base_out + offset + 3] = out_t(((float)(x3 - 4)) * dl);
        }
    }
}

// Q8_K
template <typename out_t>
__global__ void dequant_q8_K_kernel(
    out_t* __restrict__ out,
    const block_q8_K* __restrict__ blocks,
    int64_t num_blocks) {
    int64_t blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= num_blocks) return;
    const auto& b = blocks[blk];
    const float d = fp16_to_float_dev(b.d);
    const int64_t base = blk * QK_K;
    #pragma unroll
    for (int i = 0; i < QK_K; ++i) {
        out[base + i] = out_t((float)b.qs[i] * d);
    }
}

// ============================================================
// Host 端统一入口：根据目标 dtype 派发到正确的模板实例
// ============================================================

// 内部辅助宏：根据 to_bf16 派发到模板 kernel
#define DISPATCH_TYPE(dst_ptr, src_cast, count, stream, kernel_name)                     \
    do {                                                                                 \
        if (!to_bf16) {                                                                  \
            kernel_name<__half><<<grid, block, 0, stream>>>(                             \
                static_cast<__half*>(dst_ptr),                                           \
                static_cast<src_cast>(src),                                              \
                count);                                                                  \
        } else {                                                                         \
            kernel_name<__nv_bfloat16><<<grid, block, 0, stream>>>(                      \
                static_cast<__nv_bfloat16*>(dst_ptr),                                    \
                static_cast<src_cast>(src),                                              \
                count);                                                                  \
        }                                                                                \
    } while(0)

void launch_gpu_convert(void* dst, const void* src,
                        int src_type, int64_t num_elems,
                        bool to_bf16, cudaStream_t stream) {
    const int block = 256;

    // ── F32 → out_t ──
    if (src_type == GGML_TYPE_F32) {
        int64_t grid = (num_elems + block - 1) / block;
        DISPATCH_TYPE(dst, const float*, num_elems, stream, convert_f32_kernel);
        return;
    }
    // ── BF16 raw → out_t ──
    if (src_type == GGML_TYPE_BF16) {
        int64_t grid = (num_elems + block - 1) / block;
        DISPATCH_TYPE(dst, const uint16_t*, num_elems, stream, convert_bf16_kernel);
        return;
    }

    // ── 量化格式 ──
    #define HANDLE_QUANT(ggml_enum_val, block_cpp_type, blk_elems, kernel)                \
        if (src_type == GGML_TYPE_##ggml_enum_val) {                      \
            int64_t nb = (num_elems + blk_elems - 1) / blk_elems;                         \
            int64_t grid = (nb + block - 1) / block;                                      \
            DISPATCH_TYPE(dst, const block_cpp_type*, nb, stream, kernel);                \
            return;                                                                       \
        }

    HANDLE_QUANT(Q4_0, block_q4_0, QK4_0, dequant_q4_0_kernel);
    HANDLE_QUANT(Q4_1, block_q4_1, QK4_1, dequant_q4_1_kernel);
    HANDLE_QUANT(Q5_0, block_q5_0, QK5_0, dequant_q5_0_kernel);
    HANDLE_QUANT(Q5_1, block_q5_1, QK5_1, dequant_q5_1_kernel);
    HANDLE_QUANT(Q8_0, block_q8_0, QK8_0, dequant_q8_0_kernel);
    HANDLE_QUANT(Q2_K, block_q2_K, QK_K, dequant_q2_K_kernel);
    HANDLE_QUANT(Q3_K, block_q3_K, QK_K, dequant_q3_K_kernel);
    HANDLE_QUANT(Q4_K, block_q4_K, QK_K, dequant_q4_K_kernel);
    HANDLE_QUANT(Q5_K, block_q5_K, QK_K, dequant_q5_K_kernel);
    HANDLE_QUANT(Q6_K, block_q6_K, QK_K, dequant_q6_K_kernel);
    HANDLE_QUANT(Q8_K, block_q8_K, QK_K, dequant_q8_K_kernel);

    #undef HANDLE_QUANT
    #undef DISPATCH_TYPE

    // 未支持的格式：静默跳过（F16 / IQ 类型不应走到这里）
}

// ============================================================
// BF16 → FP16 原地转换（不支持 BF16 计算的 GPU 加载权重时用）
// ============================================================
__global__ void bf16_to_fp16_kernel(uint16_t* __restrict__ data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    __nv_bfloat16 bf = reinterpret_cast<__nv_bfloat16*>(data)[idx];
    reinterpret_cast<__half*>(data)[idx] = __float2half(__bfloat162float(bf));
}

void convert_bf16_to_fp16_gpu(void* data, size_t num_elements, cudaStream_t stream) {
    if (!data || num_elements == 0) {
        return;
    }
    const int block = 256;
    const int grid = static_cast<int>((num_elements + block - 1) / block);
    bf16_to_fp16_kernel<<<grid, block, 0, stream>>>(
        static_cast<uint16_t*>(data), num_elements);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace vm_c
