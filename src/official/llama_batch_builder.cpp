#include "vm_c/official/llama_runtime.hpp"

#include "llama.h"

#include <spdlog/spdlog.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace vm_c::official {

// llama_batch_init 分配了 seq_id 指针数组，但未分配每个 token 的 seq_id 值。
// 我们用这块 scratch 存储实际的 seq_id 值。
// 注意：fill_batch_common 覆盖了 seq_id[i] 使其指向 g_seq_scratch，
// 必须用 LlamaBatchBuilder::free 释放（不要直接用 llama_batch_free）。
static thread_local std::vector<llama_seq_id> g_seq_scratch;

static void fill_batch_common(
    llama_batch& batch,
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& positions,
    const std::vector<int32_t>& seq_ids,
    const std::vector<int8_t>& logits_flags) {
    const int32_t n = static_cast<int32_t>(tokens.size());
    batch.n_tokens = n;
    g_seq_scratch.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i]  = tokens[static_cast<size_t>(i)];
        batch.pos[i]    = positions[static_cast<size_t>(i)];
        g_seq_scratch[static_cast<size_t>(i)] = seq_ids[static_cast<size_t>(i)];
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = &g_seq_scratch[static_cast<size_t>(i)];
        batch.logits[i] = logits_flags[static_cast<size_t>(i)] != 0;
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
    batch.n_tokens = 1;
    batch.token[0]  = token;
    batch.pos[0]    = position;
    g_seq_scratch.resize(1);
    g_seq_scratch[0] = seq_id;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0] = &g_seq_scratch[0];
    batch.logits[0] = want_logits ? 1 : 0;
    return batch;
}

llama_batch LlamaBatchBuilder::build_mtp_draft(
    int32_t token, int32_t position,
    const float* embd_rows, int n_embd) {
    // 构建单 token 的 MTP draft batch。
    //   batch.embd[0] = embd_rows (host pointer to pre-norm hidden state, hidden_size × float32)
    //   batch.token[0] = token
    //   batch.pos[0] = position
    //   batch.logits[0] = 1 (MTP draft 需要从 ctx_mtp 读 pre-norm + logits)
    // llama_batch_init 的 embd 参数与 token 互斥：embd>0 时不分配 token。
    // MTP draft 需要两者都有（token 由 graph token 路径消费，embd 由 graph embd 路径消费），
    // 所以用 embd=0 获取 token 分配，再手动分配 embd。
    const int32_t n_tokens = 1;
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.embd = static_cast<float*>(std::malloc(
        static_cast<size_t>(n_tokens) * static_cast<size_t>(n_embd) * sizeof(float)));
    if (!batch.embd) {
        LlamaBatchBuilder::free(batch);
        throw std::bad_alloc();
    }
    // fill_batch_common 需要 1-token vector
    std::vector<int32_t> tokens_v = {token};
    std::vector<int32_t> positions_v = {position};
    std::vector<int32_t> seq_ids_v = {0};
    std::vector<int8_t> logits_flags_v = {1};
    fill_batch_common(batch, tokens_v, positions_v, seq_ids_v, logits_flags_v);
    if (embd_rows && n_embd > 0) {
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
    // seq_id[i] 被 fill_batch_common 覆盖为 &g_seq_scratch[i]，不是 malloc 出来的，
    // 所以只释放外层的指针数组 batch.seq_id 本身，不释放每个元素。
    if (batch.seq_id)   std::free(batch.seq_id);
    if (batch.logits)   std::free(batch.logits);
    // 避免误用
    batch.token    = nullptr;
    batch.embd     = nullptr;
    batch.pos      = nullptr;
    batch.n_seq_id = nullptr;
    batch.seq_id   = nullptr;
    batch.logits   = nullptr;
    batch.n_tokens = 0;
}

}  // namespace vm_c::official
