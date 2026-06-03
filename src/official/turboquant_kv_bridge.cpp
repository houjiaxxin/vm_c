#include "vm_c/official/turboquant_kv_bridge.hpp"

#include "vmc-turboquant-attn.h"
#include "vm_c/model/llama_gguf_tensor_map.hpp"

#include "llama.h"
#include "llama-model.h"

#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/kernels_turboquant.h"
#include "vm_c/cuda/kernels_flash_attn.h"
#include "vm_c/official/ggml_backend_pool.hpp"
#include "vm_c/official/ggml_weight.hpp"

#include <ggml-cuda.h>

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace vm_c::official {
namespace {

bool tq_q_is_head_split_3d(const ggml_tensor* q) {
    // 对齐 qwen35moe build_layer_attn: Q 为 view_3d [head_dim, n_head, n_tokens]；
    // decode n_tokens=1 时 ggml_n_dims(q)==2，须用 nb[2] 识别 token 轴。
    return q->nb[2] != 0 && q->ne[1] > 1;
}

void tq_attn_q_layout(const ggml_tensor* q, int* n_embd, int* num_tokens) {
    if (tq_q_is_head_split_3d(q)) {
        *n_embd = static_cast<int>(q->ne[0] * q->ne[1]);
        *num_tokens = static_cast<int>(q->ne[2]);
    } else {
        *n_embd = static_cast<int>(q->ne[0]);
        *num_tokens = static_cast<int>(q->ne[1]);
    }
}

void ggml_cuda_vm_c_turboquant_attn_impl(ggml_tensor* dst) {
    const int32_t layer = ((const int32_t*) dst->op_params)[0];
    float kq_scale = 0.0f;
    void* bridge_ptr = nullptr;
    memcpy(&kq_scale, (const char*) dst->op_params + sizeof(int32_t), sizeof(float));
    memcpy(&bridge_ptr, (const char*) dst->op_params + sizeof(int32_t) + sizeof(float), sizeof(void*));

    auto* bridge = static_cast<TurboQuantKvBridge*>(bridge_ptr);
    if (!bridge) {
        throw std::runtime_error("ggml_vm_c_turboquant_attn: null bridge");
    }

    ggml_tensor* q = dst->src[0];
    ggml_tensor* k = dst->src[1];
    ggml_tensor* v = dst->src[2];

    int n_embd = 0;
    int num_tokens = 0;
    tq_attn_q_layout(q, &n_embd, &num_tokens);
    const int num_heads = bridge->num_attention_heads();
    const int num_kv_heads = bridge->num_kv_heads();
    const int head_dim = bridge->head_dim();

    if (n_embd != num_heads * head_dim) {
        throw std::runtime_error(
            "ggml_vm_c_turboquant_attn: q embedding dim mismatch: ne0=" + std::to_string(n_embd) +
            " expected=" + std::to_string(num_heads * head_dim));
    }

    void* kv_cache = bridge->kv_cache_ptr(layer);
    if (!kv_cache) {
        throw std::runtime_error(
            "ggml_vm_c_turboquant_attn: null kv cache for layer " + std::to_string(layer));
    }

    const int64_t* slot_mapping = bridge->slot_mapping();
    if (!slot_mapping) {
        throw std::runtime_error("ggml_vm_c_turboquant_attn: slot_mapping not set");
    }

    cudaStream_t stream = bridge->stream();
    const auto dtype = gpu().compute_dtype();

    turboquant_store(
        k->data, v->data, kv_cache, slot_mapping,
        num_tokens, num_kv_heads, head_dim,
        bridge->block_size(),
        bridge->tq_config(), bridge->tq_buffers(), bridge->tq_workspace(),
        stream);

    // 统一使用分页 KV cache attention：turboquant_decode_attention 通过 block_tables
    // 读取量化缓存，正确处理 prefill / decode / MTP verify 等所有场景。
    // 不使用 flash_attention_prefill_local：该 kernel 假设 local K/V 包含从 position 0
    // 开始的所有历史 KV（stride = HEAD_DIM 模板参数），在多 head_dim 模型和 MTP
    // 验证阶段存在越界风险。
    turboquant_decode_attention(
        dst->data, q->data, kv_cache,
        bridge->block_tables(), bridge->seq_lens(),
        num_tokens, num_heads, num_kv_heads, head_dim,
        kq_scale, bridge->block_size(), bridge->max_num_blocks_per_req(),
        bridge->tq_config(), bridge->tq_buffers(), bridge->tq_workspace(),
        bridge->tq_config().max_num_kv_splits, stream);
}

}  // namespace

void register_ggml_turboquant_attn_kernels() {
    ggml_cuda_register_vm_c_turboquant_attn(ggml_cuda_vm_c_turboquant_attn_impl);
    spdlog::info("TurboQuantKvBridge: registered GGML_OP_VM_C_TURBOQUANT_ATTN CUDA kernel");
}

// allreduce callback 已移至 nccl_comm.cpp（NcclComm::init 中注册）
// 旧 custom_ar 已删除，不再有 register_ggml_tp_allreduce_kernels

TurboQuantKvBridge::~TurboQuantKvBridge() {
    if (tensor_ctx_) {
        // 先释放手动分配的 CUDA backend buffer，再释放 ggml context
        for (auto* t : kv_tensors_) {
            if (t && t->buffer) {
                ggml_backend_buffer_free(t->buffer);
                t->buffer = nullptr;
            }
        }
        ggml_free(tensor_ctx_);
        tensor_ctx_ = nullptr;
    }
    tq_buffers_.free_buf();
    tq_workspace_.free_buf();
}

void TurboQuantKvBridge::initialize(
    const VmCConfig& config,
    KVCacheManager* kv_cache_mgr,
    int gpu_device,
    int num_layers,
    int tp_size) {
    if (!kv_cache_mgr || !kv_cache_mgr->is_turboquant()) {
        throw std::runtime_error("TurboQuantKvBridge::initialize: KVCacheManager is not TurboQuant");
    }
    if (tp_size <= 0) {
        throw std::runtime_error("TurboQuantKvBridge::initialize: tp_size must be positive");
    }

    kv_cache_mgr_ = kv_cache_mgr;
    gpu_device_ = gpu_device;
    num_layers_ = num_layers;
    // column-TP：每 rank 的 Q 投影与 KV cache 均为分片 head 数（对齐 KVCacheManager / graph ne）。
    num_attention_heads_ = config.model.num_attention_heads / tp_size;
    num_kv_heads_ = std::max(1, config.model.num_key_value_heads / tp_size);
    head_dim_ = config.model.head_dim;
    if (num_attention_heads_ <= 0) {
        throw std::runtime_error("TurboQuantKvBridge::initialize: num_attention_heads per rank is zero");
    }

    tq_config_ = TQConfig::from_method(head_dim_, config.cache.kv_cache_quant_method.c_str(), gpu_device_);
    tq_buffers_.init(head_dim_, tq_config_.key_quant_bits, gpu_device_);
    tq_workspace_.ensure_store(num_kv_heads_, head_dim_, gpu_device_);
    tq_workspace_.ensure_decode(
        num_attention_heads_, head_dim_, tq_config_.max_num_kv_splits, gpu_device_);

    kv_ptrs_.resize(static_cast<size_t>(num_layers));
    kv_tensors_.resize(static_cast<size_t>(num_layers), nullptr);

    for (int l = 0; l < num_layers; ++l) {
        if (!kv_cache_mgr_->is_layer_turboquant(l)) {
            kv_ptrs_[static_cast<size_t>(l)] = nullptr;
            continue;
        }
        kv_ptrs_[static_cast<size_t>(l)] = kv_cache_mgr_->key_cache_ptr(l);
    }

    ggml_init_params params{};
    params.mem_size = 32u * 1024u * 1024u;
    params.no_alloc = true;
    tensor_ctx_ = ggml_init(params);
    if (!tensor_ctx_) {
        throw std::runtime_error("TurboQuantKvBridge::initialize: ggml_init failed");
    }

    for (int l = 0; l < num_layers; ++l) {
        if (!kv_ptrs_[static_cast<size_t>(l)]) {
            continue;
        }
        const int64_t cache_bytes = static_cast<int64_t>(
            kv_cache_mgr_->num_gpu_blocks() * kv_cache_mgr_->group(
                kv_cache_mgr_->group_id_for_layer(l)).bytes_per_block);
        ggml_tensor* t = ggml_new_tensor_1d(tensor_ctx_, GGML_TYPE_I8, cache_bytes > 0 ? cache_bytes : 1);
        t->data = kv_ptrs_[static_cast<size_t>(l)];

        // 分配 CUDA backend buffer 使 t->buffer 非空，
        // 图调度器需要此字段判断 tensor 位置（host vs device）。
        // 不覆盖 t->data（指向 KVCacheManager 已分配的 CUDA 内存）。
        ggml_backend_buffer_type_t buft = ggml_backend_cuda_buffer_type(gpu_device_);
        size_t alloc_size = static_cast<size_t>(t->ne[0]) * ggml_type_size(t->type);
        ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(buft, alloc_size);
        if (buf) {
            t->buffer = buf;
        }

        kv_tensors_[static_cast<size_t>(l)] = t;
    }

    spdlog::info(
        "TurboQuantKvBridge: initialized layers={} method={} slot_size={}",
        num_layers, config.cache.kv_cache_quant_method, tq_config_.slot_size);
    // [vm_c] 摘要：bridge 的 head_dim/slot_size/num_kv_heads/num_attn_heads（与 KVCacheManager 对齐）
    spdlog::info("[TQ-BRIDGE-INIT] head_dim={} num_attn_heads={} num_kv_heads={} slot_size={} key_packed={} value_packed={} tq_layers_enabled={}",
                 head_dim_, num_attention_heads_, num_kv_heads_,
                 tq_config_.slot_size, tq_config_.key_packed_size, tq_config_.value_packed_size,
                 tq_layer_enabled_count());
}

void TurboQuantKvBridge::set_batch_metadata(
    const int64_t* d_slot_mapping,
    const int32_t* d_block_tables,
    const int32_t* d_seq_lens,
    int num_tokens,
    int num_reqs,
    int max_num_blocks_per_req,
    int64_t block_size,
    bool is_prefill,
    int num_prefill_tokens,
    int num_decode_tokens,
    const int32_t* d_token_to_seq,
    const int32_t* d_token_positions,
    cudaStream_t stream) {
    d_slot_mapping_ = d_slot_mapping;
    d_block_tables_ = d_block_tables;
    d_seq_lens_ = d_seq_lens;
    num_tokens_ = num_tokens;
    num_reqs_ = num_reqs;
    max_num_blocks_per_req_ = max_num_blocks_per_req;
    block_size_ = block_size;
    is_prefill_ = is_prefill;
    num_prefill_tokens_ = num_prefill_tokens;
    num_decode_tokens_ = num_decode_tokens;
    d_token_to_seq_ = d_token_to_seq;
    d_token_positions_ = d_token_positions;
    stream_ = stream;
}

void TurboQuantKvBridge::mark_decode_mode(int num_tokens) {
    is_prefill_ = false;
    num_prefill_tokens_ = 0;
    num_decode_tokens_ = num_tokens;
    num_tokens_ = num_tokens;
}

void* TurboQuantKvBridge::kv_cache_ptr(int layer) const {
    if (layer < 0 || static_cast<size_t>(layer) >= kv_ptrs_.size()) return nullptr;
    return kv_ptrs_[static_cast<size_t>(layer)];
}

ggml_tensor* TurboQuantKvBridge::kv_cache_tensor(int layer) const {
    if (layer < 0 || static_cast<size_t>(layer) >= kv_tensors_.size()) return nullptr;
    return kv_tensors_[static_cast<size_t>(layer)];
}

int TurboQuantKvBridge::tq_layer_enabled_count() const {
    if (!kv_cache_mgr_) return 0;
    int n = 0;
    const int L = kv_cache_mgr_->num_layers();
    for (int l = 0; l < L; ++l) {
        if (kv_cache_mgr_->is_layer_turboquant(l)) ++n;
    }
    return n;
}

extern "C" ggml_tensor* vm_c_tq_bridge_kv_tensor(void* bridge, int32_t layer) {
    if (!bridge) {
        return nullptr;
    }
    return static_cast<TurboQuantKvBridge*>(bridge)->kv_cache_tensor(layer);
}

extern "C" bool vm_c_tq_bridge_layer_uses_turboquant(void* bridge, int32_t layer) {
    if (!bridge) {
        return false;
    }
    return static_cast<TurboQuantKvBridge*>(bridge)->is_layer_turboquant(layer);
}

extern "C" ggml_tensor* vm_c_graph_tp_allreduce(
    ggml_context* ctx,
    ggml_tensor* tensor,
    void* tp,
    int tp_size) {
    if (!ctx || !tensor || !tp || tp_size <= 1) {
        return tensor;
    }
    return ggml_vm_c_tp_allreduce(ctx, tensor, tp, tp_size);
}

void bind_vm_c_weights_to_llama_model(
    llama_model* model,
    const std::unordered_map<std::string, GpuTensor>& gpu_weights) {
    if (!model) {
        throw std::runtime_error("bind_vm_c_weights_to_llama_model: null model");
    }

    const auto& tensor_map = llama_internal_get_tensor_map(model);
    const int num_layers = llama_model_n_layer(model);

    std::unordered_set<std::string> llama_name_set;
    std::unordered_map<std::string, ggml_tensor*> llama_by_name;
    llama_by_name.reserve(tensor_map.size());
    for (const auto& [llama_name, t] : tensor_map) {
        llama_name_set.insert(llama_name);
        llama_by_name.emplace(llama_name, t);
    }

    int bound = 0;
    int missing = 0;

    for (const auto& [vmc_name, gt] : gpu_weights) {
        const std::string llama_name = vm_c::vmc_tensor_name_to_llama_gguf(
            vmc_name, num_layers, &llama_name_set);
        const auto it = llama_by_name.find(llama_name);
        if (it == llama_by_name.end()) {
            ++missing;
            continue;
        }
        ggml_tensor* lt = it->second;

        official::GgmlWeightRef wref = gt.ggml_weight_ref();
        if (!wref.data) {
            throw std::runtime_error(
                "bind_vm_c_weights_to_llama_model: weight has no device data: " + vmc_name);
        }
        if (lt->type != wref.type) {
            throw std::runtime_error(
                std::string("bind_vm_c_weights_to_llama_model: type mismatch for ") + vmc_name +
                " -> " + llama_name +
                ": llama=" + ggml_type_name(lt->type) +
                " vm_c=" + ggml_type_name(wref.type));
        }
        // column-TP：llama GGUF 元数据为全尺寸，vm_c 持有分片；同步 ne/nb 与分片 buffer。
        official::ggml_sync_tensor_shape_from_weight_ref(lt, wref);

        if (!ggml_rebind_weight_tensor(lt, wref)) {
            throw std::runtime_error(
                "bind_vm_c_weights_to_llama_model: rebind failed for " + vmc_name +
                " -> " + llama_name);
        }
        ++bound;
    }

    spdlog::info(
        "bind_vm_c_weights_to_llama_model: bound={} missing_in_llama={} total_vm_c={}",
        bound, missing, gpu_weights.size());

    if (bound == 0 && !gpu_weights.empty()) {
        throw std::runtime_error(
            "bind_vm_c_weights_to_llama_model: no weights bound — vm_c/llama tensor name map mismatch");
    }

    std::vector<std::string> unbound_llama_weights;
    for (const auto& [name, lt] : tensor_map) {
        if (!lt || lt->data != nullptr) {
            continue;
        }
        if (name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0) {
            unbound_llama_weights.push_back(name);
        }
    }
    if (!unbound_llama_weights.empty()) {
        std::string msg = "bind_vm_c_weights_to_llama_model: llama weights without vm_c bind (first 8): ";
        for (size_t i = 0; i < unbound_llama_weights.size() && i < 8; ++i) {
            if (i > 0) {
                msg += ", ";
            }
            msg += unbound_llama_weights[i];
        }
        if (unbound_llama_weights.size() > 8) {
            msg += " ... (+" + std::to_string(unbound_llama_weights.size() - 8) + " more)";
        }
        throw std::runtime_error(msg);
    }
}

}  // namespace vm_c::official
