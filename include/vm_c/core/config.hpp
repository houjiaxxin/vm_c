#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <set>
#include <unordered_map>

namespace vm_c {

struct ModelConfig {
    std::string model_name;
    std::string model_dir;
    std::string tokenizer_dir;
    int64_t max_model_len = 0;
    bool max_model_len_from_cli = false;
    int hidden_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int intermediate_size = 0;
    int vocab_size = 0;
    float rms_norm_eps = 0.0f;
    float rope_theta = 0.0f;
    float rope_scaling = 1.0f;
    float partial_rotary_factor = 1.0f;
    int head_dim = 0;
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int num_expert_group = 1;
    int topk_group = 1;
    int n_shared_experts = 0;
    int moe_intermediate_size = 0;
    int shared_expert_intermediate_size = 0;
    float routed_scaling_factor = 1.0f;
    bool norm_topk_prob = true;
    int decoder_sparse_step = 1;
    std::vector<int> mlp_only_layers;
    std::string dtype_str = "auto";
    std::string arch;
    std::vector<int> eos_token_ids;
    int64_t block_size = 16;

    std::string quant_method;
    std::unordered_map<std::string, int> per_weight_bits;
    std::string kv_cache_quant_method_hint;

    // Qwen3.5 混合注意力层类型：每层 "full_attention" 或 "linear_attention"
    std::vector<std::string> layer_types;
    // 线性注意力参数（GatedDeltaNet）
    int linear_conv_kernel_dim = 0;
    int linear_key_head_dim = 0;       // = ssm_d_state (head_k_dim / head_v_dim for tensor shapes)
    int linear_value_head_dim = 0;     // = ssm_d_state (matches tensor shape, forward uses ssm_d_inner/n_v_heads)
    int linear_num_key_heads = 0;      // = ssm_n_group
    int linear_num_value_heads = 0;    // = ssm_dt_rank
    int linear_inner_dim = 0;          // = ssm_d_inner (used in forward: head_v_dim = inner_dim / num_v_heads)

    float default_temperature = 0.80f;     // 对齐 llama.cpp common_params_sampling 默认值
    float default_top_p = 0.95f;
    int default_top_k = 40;

    // 推测解码：无/MTPSpecDec/NONE
    std::string spec_method = "none";
    // CLI 显式指定 spec_method 时置 true，防止被 GGUF 元数据覆盖
    bool spec_method_from_cli = false;
    // MTP 预测层数（从 GGUF metadata 中读取）
    int mtp_predict_layers = 0;
    // MTP 每步最多 draft token 数；由 --spec-draft-n-max 设置（对齐 llama.cpp draft.n_max）
    int mtp_spec_width = 5;
    // 最少 draft 数，不足则清空（对齐 llama.cpp draft.n_min，默认 0）
    int mtp_spec_n_min = 0;
    // draft 置信度下限，低于则停止 draft（对齐 llama.cpp draft.p_min，默认 0）
    float mtp_spec_p_min = 0.0f;

    /// 主模型 forward 层数（不含 MTP block），对齐 llama.cpp n_transformer_layers
    int num_transformer_layers() const {
        return num_hidden_layers - mtp_predict_layers;
    }

    /// MTP block 所在层索引（GGUF 中 blk.{n_main}），无 MTP 时返回 -1
    int mtp_block_layer_index() const {
        if (mtp_predict_layers <= 0 || num_hidden_layers <= mtp_predict_layers) {
            return -1;
        }
        return num_hidden_layers - mtp_predict_layers;
    }

    bool is_mtp_block_layer(int layer_idx) const {
        const int mtp_il = mtp_block_layer_index();
        return mtp_il >= 0 && layer_idx == mtp_il;
    }

    /// 该层是否参与主模型 forward（MTP block 仅用于 draft 路径）
    bool is_main_forward_layer(int layer_idx) const {
        return layer_idx >= 0 && layer_idx < num_transformer_layers();
    }

    /// 该层是否为 full attention（含 MTP block）
    bool layer_is_full_attention(int layer_idx) const {
        if (is_mtp_block_layer(layer_idx)) {
            return true;
        }
        if (!layer_types.empty() && layer_idx < static_cast<int>(layer_types.size())) {
            return layer_types[layer_idx] == "full_attention";
        }
        return true;
    }

    struct GdnRuntimeDims {
        int head_k_dim = 0;
        int head_v_dim = 0;
        int num_k_heads = 0;
        int num_v_heads = 0;
        int key_dim = 0;
        int value_dim = 0;
        int qkv_dim = 0;
        int z_dim = 0;
        int ba_dim = 0;
        int conv_kernel_size = 0;
    };

    /// GDN 运行时维度（对齐 llama qwen35moe，禁止 silent fallback）
    GdnRuntimeDims gdn_runtime_dims(int tp_rank, int tp_size) const;

    /// linear_attention 层数（GDN 状态 buffer 按层分配）
    int count_gdn_layers() const;

    /// 每个序列槽位的 GDN 状态字节数（单层：ssm + conv）
    size_t gdn_state_bytes_per_entry(int tp_size) const;

    /// 全部 GDN 层总字节数 = bytes_per_entry × num_gdn_layers × num_slots
    size_t gdn_state_total_bytes(int num_slots, int tp_size) const;

    /// 该层是否需要分配 paged / TurboQuant KV cache
    bool layer_needs_kv_cache(int layer_idx) const {
        if (is_mtp_block_layer(layer_idx)) {
            return spec_method == "mtp" && mtp_predict_layers > 0;
        }
        if (!is_main_forward_layer(layer_idx)) {
            return false;
        }
        return layer_is_full_attention(layer_idx);
    }
};

struct CacheConfig {
    int64_t block_size = 16;
    float gpu_memory_utilization = 0.90f;
    int64_t cpu_offload_gb = 0;
    std::set<std::string> cpu_offload_params;
    int64_t kv_offloading_size = 0;
    std::string kv_cache_dtype = "auto";
    std::string kv_cache_quant_method;
    // KV cache 存储布局: "nhd" (默认, 连续token) 或 "hnd" (连续head)
    // HND 布局在 decode attention 中有更好的 L2 cache 局部性，
    // 因为同一个 head 的所有 token 的 K/V 数据连续存储。
    // 仅对非 TurboQuant 的标准 KV cache 模式生效。
    std::string kv_cache_layout = "nhd";
    int64_t num_gpu_blocks = 0;
    int64_t num_cpu_blocks = 0;
    bool enable_prefix_caching = true;
    int expert_gpu_cache_capacity = 0;
    int expert_cache_update_period = 0;
    int expert_cache_history_window = 0;
    int64_t gpu_safety_margin_mb = 256;
};

struct ParallelConfig {
    int tensor_parallel_size = 1;
    int tp_rank = 0;
    std::vector<int> gpu_devices;
};

struct OffloadConfig {
    int offload_group_size = 0;
    int offload_num_in_group = 1;
    int offload_prefetch_step = 1;
    std::string kv_offloading_backend = "native";
};

struct SchedulerConfig {
    int long_prefill_token_threshold = 0;
    bool enable_chunked_prefill = true;

    // ── 混合 batch 调度配置 ──
    // decode_first: 每轮优先调度所有 decode 请求，剩余 budget 给 prefill（降低 decode 延迟）
    bool decode_first = true;
    // decode_scheduling_policy: "decode_first" (优先 decode), "fair" (公平)
    std::string scheduling_policy = "decode_first";

    // max_prefill_per_batch: 每轮 batch 中 prefill token 数上限（0 表示不限制）
    // 防止 prefill 挤占 decode 导致延迟飙升
    int max_prefill_per_batch = 0;

    // decode_latency_target_ms: decode 延迟目标（毫秒）
    // 当 decode 请求等待时间超过此值时，触发 preempt_prefill
    int decode_latency_target_ms = 100;

    // preempt_prefill_on_latency: 当 decode 延迟可能超标时，抢占正在执行的 prefill chunk
    bool preempt_prefill_on_latency = true;

    // max_decode_batch_tokens: decode 每轮最大 token 数（用于控制 decode 资源预留）
    int decode_reserved_tokens = 0;
};

struct ServerConfig {
    static constexpr int DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048;
    static constexpr int DEFAULT_MAX_NUM_SEQS = 128;
    static constexpr int DEFAULT_UBATCH_SIZE = 512;
    static constexpr int MIN_MODEL_LEN = 256;

    std::string host = "0.0.0.0";
    int port = 8000;
    std::string served_model_name;
    std::string system_fingerprint;
    int max_num_seqs = 0;
    int max_num_batched_tokens = 0;
    int poll_interval_ms = 500;
    int step_interval_ms = 10;
    int thread_pool_size = 8;
    int default_max_tokens = 4096;
    int default_completion_max_tokens = 16;
};

/// GDN recurrent 槽位数：llama.cpp 按 n_seqs；vLLM mamba_cache_mode=none 为每 seq (1+spec) 页
int compute_gdn_state_slots(const ModelConfig& model, const ServerConfig& server);

struct VmCConfig {
    ModelConfig model;
    CacheConfig cache;
    ParallelConfig parallel;
    SchedulerConfig scheduler;
    OffloadConfig offload;
    ServerConfig server;
};

/// rank0 EngineBuffers 峰值显存（与 EngineBuffers::ensure 对齐，不含 d_pre_norm_hidden / SpeculativeEngine）
size_t estimate_engine_bufs_peak_bytes(const VmCConfig& config, int64_t kv_cache_blocks);

/// 每个 TP rank 的 ctx.d_input_ids +（rank>0 时）d_hidden_states / d_position_ids 峰值
size_t estimate_tp_rank_io_buffer_bytes(const VmCConfig& config, bool rank_has_local_hidden);

ModelConfig load_model_config_from_json(const std::string& model_dir);

}
