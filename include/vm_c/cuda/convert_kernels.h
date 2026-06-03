#pragma once
// GPU 端权重格式转换 kernel 声明
// 输出类型根据 to_bf16 自动选择：false→FP16 (__half), true→BF16 (__nv_bfloat16)
// 调用方需保证 dst 缓冲区已用正确类型分配空间（元素数按 num_elems 计）

#include <cstdint>
#include <cuda_runtime.h>

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace vm_c {

// ── 统一入口 ──
// dst:       输出缓冲区（已分配，元素数 = num_elems）
// src:       原始 GGUF 数据（F32 / BF16 raw / 量化 block）
// src_type:  ggml_type 整数值（GGML_TYPE_F32 等）
// num_elems: 输出元素数
// to_bf16:   false → 输出 __half (FP16), true → 输出 __nv_bfloat16 (BF16)
// stream:    CUDA stream
void launch_gpu_convert(void* dst, const void* src,
                        int src_type, int64_t num_elems,
                        bool to_bf16, cudaStream_t stream);

} // namespace vm_c
