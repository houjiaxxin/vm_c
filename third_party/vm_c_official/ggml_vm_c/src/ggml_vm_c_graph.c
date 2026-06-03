#include "ggml.h"

#include <string.h>

// qwen35moe Q 为 ggml_view_3d: [head_dim, n_head, n_tokens]。
// n_tokens=1 时 ggml_n_dims(q)==2，须用 nb[2] 识别 token 轴。
static bool ggml_vm_c_tq_q_is_head_split_3d(const struct ggml_tensor * q) {
    return q->nb[2] != 0 && q->ne[1] > 1;
}

struct ggml_tensor * ggml_vm_c_turboquant_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        struct ggml_tensor  * kv_cache,
        int32_t               layer,
        float                 kq_scale,
        void                * bridge) {
    GGML_ASSERT(q != NULL && k != NULL && v != NULL && kv_cache != NULL);
    GGML_ASSERT(bridge != NULL);

    int64_t n_embd_out;
    int64_t n_tokens_out;
    if (ggml_vm_c_tq_q_is_head_split_3d(q)) {
        n_embd_out = q->ne[0] * q->ne[1];
        n_tokens_out = q->ne[2];
    } else {
        n_embd_out = q->ne[0];
        n_tokens_out = q->ne[1];
    }

    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, q->type, n_embd_out, n_tokens_out);

    result->op = GGML_OP_VM_C_TURBOQUANT_ATTN;
    result->src[0] = q;
    result->src[1] = k;
    result->src[2] = v;
    result->src[3] = kv_cache;

    ((int32_t *) result->op_params)[0] = layer;
    memcpy((char *) result->op_params + sizeof(int32_t), &kq_scale, sizeof(float));
    memcpy((char *) result->op_params + sizeof(int32_t) + sizeof(float), &bridge, sizeof(void *));

    return result;
}

// ggml_vm_c_tp_allreduce 已迁移到 ggml.c（官方 ggml 仓库），
// 此实现移至 ggml.c:ggml_vm_c_tp_allreduce 统一维护。
// remove_if_dup: 此文件中的副本是旧版本，由 ggml.c 中的实现替代。
