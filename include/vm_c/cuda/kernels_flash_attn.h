#pragma once

#include "vm_c_tensor.h"
#include <cstdint>
#include <string>
#include <optional>

namespace vm_c {

// ── 高性能 Flash Attention Prefill Kernel ──
//
// 参考 llama.cpp 的 fattn-tile 实现，使用 warp 级并行 softmax + shared memory tiling。
//
// 与旧版 flash_attention_prefill_kernel 的关键区别：
//   1. 所有线程参与 softmax 计算（而非只有 threadIdx.x==0）
//   2. 使用 shared memory tiling 减少 global memory 访问
//   3. 在线 softmax（online softmax）算法，避免两遍扫描
//   4. 支持 GQA（Grouped Query Attention）
//   5. 支持 causal mask
//
// 算法：
//   - 每个 block 处理一个 (token, head) 对
//   - 将 KV 序列分成 Br 大小的 tile
//   - 每个 tile：加载 K/V 到 shared memory → 计算 Q*K^T → 在线 softmax 更新 → 累加 V
//   - 最终归一化输出
//
// 参数说明：
//   output:          [num_tokens, num_heads, head_dim] 输出
//   query:           [num_tokens, num_heads, head_dim] Q 张量
//   key_cache:       paged KV cache key 缓冲区
//   value_cache:     paged KV cache value 缓冲区
//   num_tokens:      token 数量
//   num_heads:       Q 头数
//   num_kv_heads:    KV 头数
//   head_dim:        头维度
//   scale:           attention 缩放因子 (1/sqrt(head_dim))
//   block_size:      KV cache block 大小
//   max_num_blocks_per_req: 每个请求最大 block 数
//   block_tables:    [num_reqs, max_num_blocks_per_req] block 表
//   seq_lens:        [num_reqs] 序列长度
//   token_to_seq:    [num_tokens] token 到序列的映射
//   token_positions: [num_tokens] token 位置
//   num_reqs:        请求数量
//   kv_cache_dtype:  KV cache 数据类型 ("auto" 或 "fp8_e4m3" 等)

void flash_attention_prefill_v2(
    void* output,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int64_t block_size,
    int64_t max_num_blocks_per_req,
    const int32_t* block_tables,
    const int32_t* seq_lens,
    const int32_t* token_to_seq,
    const int32_t* token_positions,
    int num_reqs,
    const std::string& kv_cache_dtype,
    ScalarType dtype,
    cudaStream_t stream = 0,
    const std::string& layout = "nhd");

// ── 本地（非分页）Flash Attention Prefill ──
// 从局部连续 K/V 数组读取（而非分页 KV cache）。
// 用于 TurboQuant 等不维护独立 value_cache 指针的后端。
void flash_attention_prefill_local(
    void* output,
    const void* query,
    const void* key,
    const void* value,
    int num_tokens, int num_heads, int num_kv_heads, int head_dim,
    float scale,
    const int32_t* token_to_seq,
    const int32_t* token_positions,
    ScalarType dtype,
    cudaStream_t stream = 0);

} // namespace vm_c