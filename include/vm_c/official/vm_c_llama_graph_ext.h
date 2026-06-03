#pragma once

// vm_c 对 llama.cpp 的图扩展 —— 在 llama_build_graph / llama_build_mtp_graph 中注入
// 自定义 ggml 节点（turboquant attention、column-TP allreduce、MTP embed/proj）。
//
// 这些函数由 llama/src/ 中打了 VM_C_LLAMA_INTEGRATION 补丁的文件调用。
// 实现见 src/official/turboquant_kv_bridge.cpp。

#include "llama.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/// 在 llama_graph 构建完成后插入 vm_c 定制算子（turboquant attn, TP allreduce）
void vm_c_llama_graph_ext_build(struct ggml_cgraph* graph, struct llama_context* ctx);

/// 在 MTP graph 构建完成后插入 vm_c 定制算子
void vm_c_llama_graph_ext_build_mtp(struct ggml_cgraph* graph, struct llama_context* ctx);

// ── TP 维度辅助函数（内联） ──────────────────────────────────────────

/// 根据 TP 大小计算本地分片维度
static inline int64_t vm_c_tp_local_dim(int64_t dim, int tp_size) {
    return tp_size > 1 ? dim / tp_size : dim;
}

/// Mamba SSM conv_state 的 TP 分片大小
/// 完整公式: (ssm_d_conv-1) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state)
static inline int vm_c_tp_local_n_embd_r(int ssm_d_conv, int ssm_d_inner, int ssm_n_group, int ssm_d_state, int tp_size) {
    int full = (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state);
    return tp_size > 1 ? full / tp_size : full;
}

/// Mamba SSM ssm_state 的 TP 分片大小
/// 完整公式: ssm_d_state * ssm_d_inner
static inline int vm_c_tp_local_n_embd_s(int ssm_d_state, int ssm_d_inner, int ssm_dt_rank, int tp_size) {
    (void) ssm_dt_rank;
    int full = ssm_d_state * ssm_d_inner;
    return tp_size > 1 ? full / tp_size : full;
}

// ── TurboQuant KV Bridge ───────────────────────────────────────────

/// 获取指定层的 KV cache tensor（用于 turboquant attn 替换）
struct ggml_tensor* vm_c_tq_bridge_kv_tensor(void* bridge, int32_t layer);

/// 查询指定层是否使用 turboquant attention
bool vm_c_tq_bridge_layer_uses_turboquant(void* bridge, int32_t layer);

// ── TP AllReduce ───────────────────────────────────────────────────

/// 在 ggml 图中插入 column-TP allreduce 节点
struct ggml_tensor* vm_c_graph_tp_allreduce(struct ggml_context* ctx, struct ggml_tensor* cur, void* tp, int tp_size);

#ifdef __cplusplus
}
#endif
