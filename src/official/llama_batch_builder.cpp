#include "vm_c/official/llama_runtime.hpp"

#include "llama.h"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace vm_c::official {

// NOTE: llama_batch_init 已经为每个 seq_id[i] 分配了 malloc 数组
// （大小为 n_seq_max * sizeof(llama_seq_id)），我们直接写入值即可，
// 不需要额外的 scratch 缓冲区。之前的实现用 g_seq_scratch 覆盖了
// seq_id[i] 指针，导致：(1) 原 malloc 数组泄漏；(2) free 时对 vector
// 内部缓冲区调用 std::free → "free(): invalid pointer" 崩溃。

static void fill_batch_common(
    llama_batch& batch,
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& positions,
    const std::vector<int32_t>& seq_ids,
    const std::vector<int8_t>& logits_flags) {
    const int32_t n = static_cast<int32_t>(tokens.size());
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]     = tokens[static_cast<size_t>(i)];
        batch.pos[i]       = positions[static_cast<size_t>(i)];
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq_ids[static_cast<size_t>(i)];
        batch.logits[i]    = logits_flags[static_cast<size_t>(i)] != 0;
    }
}

llama_batch LlamaBatchBuilder::build(
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& positions,
    const std::vector<int32_t>& seq_ids,
    const std::vector<int8_t>& logits_flags) {
    const int32_t n_tokens = static_cast<int32_t>(tokens.size());
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    fill_batch_common(batch, tokens, positions, seq_ids, logits_flags);
    if (n_tokens > 0) {
        spdlog::info("[BUILD-BATCH] n_tokens={} tokens[0]={} pos[0]={} seq_id[0]={} logits[0]={}",
            n_tokens, tokens[0], positions[0], seq_ids[0], (int)logits_flags[0]);
    }
    return batch;
}

llama_batch LlamaBatchBuilder::build_single(
    int32_t token, int32_t position, int32_t seq_id, bool want_logits) {
    llama_batch batch = llama_batch_init(1, 0, 1);
    batch.n_tokens    = 1;
    batch.token[0]    = token;
    batch.pos[0]      = position;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = seq_id;
    batch.logits[0]   = want_logits ? 1 : 0;
    return batch;
}

llama_batch LlamaBatchBuilder::build_mtp_draft(
    int32_t token, int32_t position,
    const float* embd_rows, int n_embd) {
    // 关键修复：MTP draft batch 必须同时设置 token 和 embd，不可互斥。
    //
    // MTP 图（qwen35moe.cpp graph_mtp）有两条输入路径：
    //   path[0]: 从 inp->tokens 通过 embedding 表查找 token embedding（正确方式）
    //   path[1]: 直接使用 inp->embd（当 ubatch.token == nullptr 时）
    // 路径选择由 ggml_build_forward_select 的 ubatch.token ? 0 : 1 决定。
    //
    // 旧实现（仅设 embd，token = nullptr）：
    //   - ubatch.token == nullptr → path[1] 被选中
    //   - inp->embd 被设为 ubatch->embd（隐藏状态）
    //   - inp->h 也被设为 ubatch->embd（隐藏状态）
    //   - BOTH 获得相同数据 → MTP 头收到 concat(hnorm(h), hnorm(h)) → 垃圾输出
    //
    // 新实现（同时设 token 和 embd）：
    //   - ubatch.token != nullptr → path[0] 被选中（token embedding 从 embedding 表查找，正确）
    //   - inp->h 从 ubatch->embd 获取隐藏状态（正确）
    //   - ubatch->token 和 ubatch->embd 各自独立工作，互不干扰
    const int32_t n_tokens = 1;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    // llama_batch_init 将 n_tokens 初始化为 0，需要显式设置为 1
    batch.n_tokens = 1;
    // 分配 embd 空间（llama_batch_init 在 embd=0 时不分配 embd）
    batch.embd = static_cast<float*>(std::malloc(
        static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd) * sizeof(float)));
    batch.token[0]     = token;
    batch.pos[0]       = position;
    batch.n_seq_id[0]  = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0]    = 1;

    // 将 host pointer 的 embd_rows（隐藏状态）拷贝到 batch.embd
    if (embd_rows && n_embd > 0 && batch.embd) {
        std::memcpy(batch.embd, embd_rows,
            static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd) * sizeof(float));
    }
    spdlog::info("[BUILD-MTP-DRAFT] token={} pos={} batch.token[0]={} batch.embd[0..7]=[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]",
        token, position, batch.token[0],
        batch.embd[0], batch.embd[1], batch.embd[2], batch.embd[3],
        batch.embd[4], batch.embd[5], batch.embd[6], batch.embd[7]);
    return batch;
}

void LlamaBatchBuilder::free(llama_batch& batch) {
    if (batch.token)    std::free(batch.token);
    if (batch.embd)     std::free(batch.embd);
    if (batch.pos)      std::free(batch.pos);
    if (batch.n_seq_id) std::free(batch.n_seq_id);
    if (batch.seq_id) {
        // 释放 llama_batch_init 分配的每个 seq_id[i] 数组
        for (int32_t i = 0; batch.seq_id[i] != nullptr; ++i) {
            std::free(batch.seq_id[i]);
        }
        std::free(batch.seq_id);
    }
    if (batch.logits)   std::free(batch.logits);
    batch.token    = nullptr;
    batch.embd     = nullptr;
    batch.pos      = nullptr;
    batch.n_seq_id = nullptr;
    batch.seq_id   = nullptr;
    batch.logits   = nullptr;
    batch.n_tokens = 0;
}

}  // namespace vm_c::official
