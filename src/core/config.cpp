#include "vm_c/core/config.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/core/tensor.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <set>
#include <numeric>
#include <spdlog/spdlog.h>

namespace vm_c {

using json = nlohmann::json;

static DataType parse_dtype(const std::string& s) {
    if (s == "auto") return gpu().supports_bf16() ? DataType::BFLOAT16 : DataType::FLOAT16;
    if (s == "float32" || s == "fp32") return DataType::FLOAT32;
    if (s == "float16" || s == "fp16") return DataType::FLOAT16;
    if (s == "bfloat16" || s == "bf16") return DataType::BFLOAT16;
    if (s == "fp8_e4m3" || s == "float8_e4m3fn") return DataType::FLOAT8_E4M3;
    if (s == "fp8_e5m2" || s == "float8_e5m2") return DataType::FLOAT8_E5M2;
    spdlog::warn("Unknown dtype '{}', defaulting to {} based on GPU arch", s,
                 gpu().supports_bf16() ? "BF16" : "FP16");
    return gpu().supports_bf16() ? DataType::BFLOAT16 : DataType::FLOAT16;
}

ModelConfig load_model_config_from_json(const std::string& model_dir) {
    std::string config_path = model_dir + "/config.json";
    std::ifstream f(config_path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open config.json: " + config_path);
    }
    json j;
    f >> j;

    ModelConfig cfg;
    cfg.model_dir = model_dir;

    if (j.contains("architectures") && j["architectures"].is_array() && !j["architectures"].empty()) {
        cfg.arch = j["architectures"][0].get<std::string>();
    }

    json tc = j;
    if (j.contains("text_config") && j["text_config"].is_object()) {
        tc = j["text_config"];
        spdlog::info("Detected nested text_config in config.json (multimodal/moe model)");
    }

    cfg.hidden_size = tc.value("hidden_size", 0);
    cfg.num_hidden_layers = tc.value("num_hidden_layers", 0);
    cfg.num_attention_heads = tc.value("num_attention_heads", 0);
    cfg.intermediate_size = tc.value("intermediate_size", 0);
    cfg.vocab_size = tc.value("vocab_size", 0);
    cfg.rms_norm_eps = tc.value("rms_norm_eps", 0.0f);
    cfg.rope_theta = tc.value("rope_theta", 0.0f);

    if (tc.contains("num_key_value_heads")) {
        cfg.num_key_value_heads = tc["num_key_value_heads"].get<int>();
    } else {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }

    if (tc.contains("head_dim")) {
        cfg.head_dim = tc["head_dim"].get<int>();
    } else {
        cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    }

    if (tc.contains("n_routed_experts")) {
        cfg.num_experts = tc["n_routed_experts"].get<int>();
        cfg.num_experts_per_tok = tc.value("num_experts_per_tok",
            tc.value("top_k", 0));
    } else if (tc.contains("num_local_experts")) {
        cfg.num_experts = tc["num_local_experts"].get<int>();
        cfg.num_experts_per_tok = tc.value("num_experts_per_tok", 0);
    } else if (tc.contains("num_experts")) {
        cfg.num_experts = tc["num_experts"].get<int>();
        cfg.num_experts_per_tok = tc.value("num_experts_per_tok", 0);
    }

    if (tc.contains("n_group")) {
        cfg.num_expert_group = tc["n_group"].get<int>();
    } else if (tc.contains("num_expert_group")) {
        cfg.num_expert_group = tc["num_expert_group"].get<int>();
    }

    if (tc.contains("n_shared_experts")) {
        cfg.n_shared_experts = tc["n_shared_experts"].get<int>();
    }

    if (tc.contains("moe_intermediate_size")) {
        cfg.moe_intermediate_size = tc["moe_intermediate_size"].get<int>();
    }
    if (tc.contains("shared_expert_intermediate_size")) {
        cfg.shared_expert_intermediate_size = tc["shared_expert_intermediate_size"].get<int>();
        if (cfg.n_shared_experts == 0 && cfg.shared_expert_intermediate_size > 0) {
            cfg.n_shared_experts = 1;
        }
    }
    cfg.decoder_sparse_step = tc.value("decoder_sparse_step", 1);
    if (cfg.decoder_sparse_step < 1) {
        spdlog::warn("decoder_sparse_step {} invalid, using 1", cfg.decoder_sparse_step);
        cfg.decoder_sparse_step = 1;
    }
    if (tc.contains("mlp_only_layers") && tc["mlp_only_layers"].is_array()) {
        for (const auto& item : tc["mlp_only_layers"]) {
            if (item.is_number_integer()) {
                cfg.mlp_only_layers.push_back(item.get<int>());
            }
        }
    }

    // 解析 Qwen3.5 混合注意力层类型
    if (tc.contains("layer_types") && tc["layer_types"].is_array()) {
        for (const auto& item : tc["layer_types"]) {
            if (item.is_string()) {
                cfg.layer_types.push_back(item.get<std::string>());
            }
        }
        spdlog::info("Parsed layer_types from config: {} entries", cfg.layer_types.size());
    } else if (tc.contains("full_attention_interval")) {
        // 根据 full_attention_interval 自动生成 layer_types
        int interval = tc.value("full_attention_interval", 4);
        for (int i = 0; i < cfg.num_hidden_layers; ++i) {
            if ((i + 1) % interval == 0) {
                cfg.layer_types.push_back("full_attention");
            } else {
                cfg.layer_types.push_back("linear_attention");
            }
        }
        spdlog::info("Generated layer_types from full_attention_interval={}: {} entries",
                     interval, cfg.layer_types.size());
    }

    // 线性注意力参数（GatedDeltaNet）
    cfg.linear_conv_kernel_dim = tc.value("linear_conv_kernel_dim", 0);
    cfg.linear_key_head_dim = tc.value("linear_key_head_dim", 0);
    cfg.linear_value_head_dim = tc.value("linear_value_head_dim", 0);
    cfg.linear_num_key_heads = tc.value("linear_num_key_heads", 0);
    cfg.linear_num_value_heads = tc.value("linear_num_value_heads", 0);
    cfg.linear_inner_dim = tc.value("linear_inner_dim", tc.value("ssm_d_inner", 0));
    if (tc.contains("norm_topk_prob")) {
        cfg.norm_topk_prob = tc["norm_topk_prob"].get<bool>();
    }
    if (tc.contains("routed_scaling_factor")) {
        cfg.routed_scaling_factor = tc["routed_scaling_factor"].get<float>();
    }
    if (tc.contains("topk_group")) {
        cfg.topk_group = tc["topk_group"].get<int>();
    }

    if (j.contains("torch_dtype")) {
        cfg.dtype_str = j["torch_dtype"].get<std::string>();
    } else if (tc.contains("torch_dtype")) {
        cfg.dtype_str = tc["torch_dtype"].get<std::string>();
    }

    if (tc.contains("rope_scaling")) {
        auto& rs = tc["rope_scaling"];
        if (rs.contains("factor")) {
            cfg.rope_scaling = rs["factor"].get<float>();
        }
        std::string rope_type = rs.value("type", "");
        if (rope_type == "yarn" || rope_type == "longrope") {
            if (rs.contains("original_max_position_embeddings")) {
                cfg.max_model_len = rs["original_max_position_embeddings"].get<int64_t>();
                spdlog::info("RoPE scaling: type={}, using original_max_position_embeddings={}", rope_type, cfg.max_model_len);
            }
        } else if (cfg.rope_scaling > 1.0f) {
            if (tc.contains("max_position_embeddings")) {
                int64_t base_max = tc["max_position_embeddings"].get<int64_t>();
                int64_t scaled_max = static_cast<int64_t>(base_max * cfg.rope_scaling);
                cfg.max_model_len = scaled_max;
                spdlog::info("RoPE scaling: factor={}, scaled max_model_len={} (base={} * factor)",
                             cfg.rope_scaling, scaled_max, base_max);
            }
        }
    }
    if (tc.contains("rope_parameters") && tc["rope_parameters"].is_object()) {
        auto& rp = tc["rope_parameters"];
        if (rp.contains("partial_rotary_factor")) {
            cfg.partial_rotary_factor = rp["partial_rotary_factor"].get<float>();
        }
        if (rp.contains("rope_theta")) {
            cfg.rope_theta = rp["rope_theta"].get<float>();
        }
        spdlog::info("RoPE parameters: partial_rotary_factor={}, rope_theta={}",
                     cfg.partial_rotary_factor, cfg.rope_theta);
    }

    if (cfg.max_model_len <= 0 && tc.contains("max_position_embeddings")) {
        cfg.max_model_len = tc["max_position_embeddings"].get<int64_t>();
    }

    std::string kv_cache_quant_method_from_config;
    if (j.contains("quantization_config") && j["quantization_config"].is_object()) {
        auto& qc = j["quantization_config"];
        kv_cache_quant_method_from_config = qc.value("kv_cache_quant_method", "");

        spdlog::info("Quantization config: kv_cache_quant_method={}",
                     kv_cache_quant_method_from_config.empty() ? "(none)" : kv_cache_quant_method_from_config);
    } else {
        kv_cache_quant_method_from_config = "";
    }

    cfg.kv_cache_quant_method_hint = kv_cache_quant_method_from_config;

    if (cfg.hidden_size <= 0) throw std::runtime_error("config.json missing required field: hidden_size");
    if (cfg.num_hidden_layers <= 0) throw std::runtime_error("config.json missing required field: num_hidden_layers");
    if (cfg.num_attention_heads <= 0) throw std::runtime_error("config.json missing required field: num_attention_heads");
    if (cfg.vocab_size <= 0) throw std::runtime_error("config.json missing required field: vocab_size");
    if (cfg.rms_norm_eps <= 0.0f) throw std::runtime_error("config.json missing required field: rms_norm_eps");
    if (cfg.rope_theta <= 0.0f) throw std::runtime_error("config.json missing required field: rope_theta");

    std::string mlp_layers_repr = "[";
    for (size_t i = 0; i < cfg.mlp_only_layers.size(); ++i) {
        if (i > 0) mlp_layers_repr += ',';
        mlp_layers_repr += std::to_string(cfg.mlp_only_layers[i]);
    }
    mlp_layers_repr += ']';

    spdlog::info("Model config: arch={}, hidden={}, layers={}, heads={}, kv_heads={}, "
                 "intermediate={}, vocab={}, experts={}/{}, moe_intermediate={}, shared_expert_intermediate={}, "
                 "decoder_sparse_step={}, mlp_only_layers={} ({} entries), layer_types={} ({} entries)",
                 cfg.arch, cfg.hidden_size, cfg.num_hidden_layers,
                 cfg.num_attention_heads, cfg.num_key_value_heads,
                 cfg.intermediate_size, cfg.vocab_size,
                 cfg.num_experts, cfg.num_experts_per_tok,
                 cfg.moe_intermediate_size, cfg.shared_expert_intermediate_size,
                 cfg.decoder_sparse_step,
                 mlp_layers_repr,
                 cfg.mlp_only_layers.size(),
                 cfg.layer_types.empty() ? "(all full_attention)" :
                     (cfg.layer_types.size() > 3 ?
                         cfg.layer_types[0] + "," + cfg.layer_types[1] + ",..." + cfg.layer_types.back() :
                         std::accumulate(cfg.layer_types.begin(), cfg.layer_types.end(), std::string(),
                             [](const std::string& a, const std::string& b) { return a.empty() ? b : a + "," + b; })),
                 cfg.layer_types.size());

    auto collect_eos_ids = [](const json& doc, const std::string& field) -> std::vector<int> {
        std::vector<int> ids;
        if (!doc.contains(field)) return ids;
        if (doc[field].is_array()) {
            for (auto& v : doc[field]) {
                if (v.is_number_integer()) ids.push_back(v.get<int>());
            }
        } else if (doc[field].is_number_integer()) {
            ids.push_back(doc[field].get<int>());
        }
        return ids;
    };

    std::set<int> eos_set;

    auto from_text_config = collect_eos_ids(tc, "eos_token_id");
    for (auto id : from_text_config) eos_set.insert(id);

    {
        std::string tok_config_path = model_dir + "/tokenizer_config.json";
        std::ifstream tf(tok_config_path);
        if (tf.is_open()) {
            json tj;
            tf >> tj;
            auto from_tok = collect_eos_ids(tj, "eos_token_id");
            for (auto id : from_tok) eos_set.insert(id);
        }
    }

    {
        std::string gen_config_path = model_dir + "/generation_config.json";
        std::ifstream gf(gen_config_path);
        if (gf.is_open()) {
            json gj;
            gf >> gj;
            auto from_gen = collect_eos_ids(gj, "eos_token_id");
            for (auto id : from_gen) eos_set.insert(id);
            if (gj.contains("temperature") && gj["temperature"].is_number()) {
                cfg.default_temperature = gj["temperature"].get<float>();
            }
            if (gj.contains("top_p") && gj["top_p"].is_number()) {
                cfg.default_top_p = gj["top_p"].get<float>();
            }
            if (gj.contains("top_k") && gj["top_k"].is_number_integer()) {
                cfg.default_top_k = gj["top_k"].get<int>();
            }
        }
    }

    for (auto id : eos_set) {
        cfg.eos_token_ids.push_back(id);
    }

    if (!cfg.eos_token_ids.empty()) {
        std::string eos_repr = std::to_string(cfg.eos_token_ids[0]);
        for (size_t i = 1; i < cfg.eos_token_ids.size(); ++i) {
            eos_repr += "," + std::to_string(cfg.eos_token_ids[i]);
        }
        spdlog::info("EOS token ids: [{}] (from config.json/generation_config/tokenizer_config)", eos_repr);
    } else {
        spdlog::warn("No eos_token_id found in any config file");
    }

    return cfg;
}

ModelConfig::GdnRuntimeDims ModelConfig::gdn_runtime_dims(int tp_rank, int tp_size) const {
    (void)tp_rank;
    if (linear_key_head_dim <= 0 || linear_num_key_heads <= 0 || linear_num_value_heads <= 0) {
        throw std::runtime_error(
            "GDN config missing linear_key_head_dim / linear_num_key_heads / linear_num_value_heads");
    }
    if (linear_inner_dim <= 0) {
        throw std::runtime_error("GDN config missing linear_inner_dim (ssm_d_inner)");
    }
    if (linear_inner_dim % linear_num_value_heads != 0) {
        throw std::runtime_error(
            "GDN config invalid: linear_inner_dim must be divisible by linear_num_value_heads");
    }

    GdnRuntimeDims d;
    d.head_k_dim = linear_key_head_dim;
    d.head_v_dim = linear_inner_dim / linear_num_value_heads;
    d.num_k_heads = tp_size > 1 ? linear_num_key_heads / tp_size : linear_num_key_heads;
    d.num_v_heads = tp_size > 1 ? linear_num_value_heads / tp_size : linear_num_value_heads;
    if (d.num_k_heads <= 0 || d.num_v_heads <= 0 ||
        linear_num_key_heads % tp_size != 0 || linear_num_value_heads % tp_size != 0) {
        throw std::runtime_error("GDN TP shard invalid for configured head counts");
    }
    d.key_dim = d.num_k_heads * d.head_k_dim;
    d.value_dim = d.num_v_heads * d.head_v_dim;
    d.qkv_dim = 2 * d.key_dim + d.value_dim;
    d.z_dim = d.value_dim;
    d.ba_dim = 2 * d.num_v_heads;
    d.conv_kernel_size = linear_conv_kernel_dim > 0 ? linear_conv_kernel_dim : 4;
    return d;
}

int ModelConfig::count_gdn_layers() const {
    int count = 0;
    for (int i = 0; i < num_hidden_layers; ++i) {
        if (!layer_types.empty() && i < static_cast<int>(layer_types.size())) {
            if (layer_types[i] == "linear_attention") {
                ++count;
            }
        }
    }
    return count;
}

size_t ModelConfig::gdn_state_bytes_per_entry(int tp_size) const {
    if (count_gdn_layers() <= 0) {
        return 0;
    }
    const GdnRuntimeDims gdn = gdn_runtime_dims(0, tp_size);
    const size_t per_ssm = static_cast<size_t>(gdn.num_v_heads) *
        static_cast<size_t>(gdn.head_v_dim) * static_cast<size_t>(gdn.head_k_dim) *
        sizeof(float);
    const size_t per_conv = static_cast<size_t>(gdn.qkv_dim) *
        static_cast<size_t>(gdn.conv_kernel_size - 1) * gpu().compute_dtype_size();
    return per_ssm + per_conv;
}

size_t ModelConfig::gdn_state_total_bytes(int num_slots, int tp_size) const {
    if (num_slots <= 0) {
        return 0;
    }
    const int num_gdn_layers = count_gdn_layers();
    if (num_gdn_layers <= 0) {
        return 0;
    }
    return gdn_state_bytes_per_entry(tp_size) * static_cast<size_t>(num_gdn_layers) *
        static_cast<size_t>(num_slots);
}

int compute_gdn_state_slots(const ModelConfig& model, const ServerConfig& server) {
    int slots_per_seq = 1;
    if (model.spec_method == "mtp" && model.mtp_predict_layers > 0) {
        const int spec_w = model.mtp_spec_width;
        slots_per_seq = 1 + spec_w;
    }
    const int max_seqs = server.max_num_seqs > 0 ? server.max_num_seqs : 1;
    return max_seqs * slots_per_seq;
}

size_t estimate_engine_bufs_peak_bytes(const VmCConfig& config, int64_t kv_cache_blocks) {
    const int max_num_batched_tokens = config.server.max_num_batched_tokens;
    const int max_num_seqs = config.server.max_num_seqs > 0 ? config.server.max_num_seqs : 1;
    const int hidden_size = config.model.hidden_size;
    const int vocab_size = config.model.vocab_size;
    const int spec_w = config.model.mtp_spec_width;
    const int mtp_verify_tokens =
        (config.model.spec_method == "mtp" && config.model.mtp_predict_layers > 0)
        ? (spec_w + 1) : 1;
    const int engine_token_cap = std::max(max_num_batched_tokens, mtp_verify_tokens);
    const size_t ds = gpu().compute_dtype_size();

    size_t bytes = 0;
    bytes += static_cast<size_t>(engine_token_cap) * sizeof(int32_t);
    bytes += static_cast<size_t>(engine_token_cap) * sizeof(int64_t);
    bytes += static_cast<size_t>(engine_token_cap) * static_cast<size_t>(hidden_size) * ds;
    bytes += static_cast<size_t>(max_num_seqs) * sizeof(int32_t);
    bytes += static_cast<size_t>(engine_token_cap) * sizeof(int64_t);
    bytes += static_cast<size_t>(engine_token_cap) * sizeof(int32_t);
    bytes += static_cast<size_t>(engine_token_cap) * sizeof(int32_t);
    bytes += static_cast<size_t>(max_num_seqs) * static_cast<size_t>(kv_cache_blocks) * sizeof(int32_t);
    bytes += static_cast<size_t>(std::max(max_num_seqs, mtp_verify_tokens))
        * static_cast<size_t>(vocab_size) * sizeof(float);
    bytes += static_cast<size_t>(max_num_seqs) * static_cast<size_t>(hidden_size) * ds;
    bytes += static_cast<size_t>(max_num_seqs) * (sizeof(int32_t) + sizeof(float));
    bytes += static_cast<size_t>(max_num_seqs) * (sizeof(float) * 3 + sizeof(int32_t));
    bytes += static_cast<size_t>(max_num_seqs) * static_cast<size_t>(vocab_size)
        * (sizeof(uint8_t) + sizeof(int32_t));
    bytes += static_cast<size_t>(max_num_seqs) * sizeof(float) * 3;
    return bytes;
}

size_t estimate_tp_rank_io_buffer_bytes(const VmCConfig& config, bool rank_has_local_hidden) {
    const int T = config.server.max_num_batched_tokens;
    const size_t ds = gpu().compute_dtype_size();
    size_t bytes = static_cast<size_t>(T) * sizeof(int32_t);
    if (rank_has_local_hidden) {
        bytes += static_cast<size_t>(T) * static_cast<size_t>(config.model.hidden_size) * ds;
        bytes += static_cast<size_t>(T) * sizeof(int64_t);
    }
    return bytes;
}

}
