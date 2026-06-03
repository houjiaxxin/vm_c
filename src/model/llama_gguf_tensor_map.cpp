#include "vm_c/model/llama_gguf_tensor_map.hpp"

#include <unordered_map>

namespace vm_c {
namespace {

const std::unordered_map<std::string, std::string> kQwenGgufToVmc = {
    {"token_embd.weight",           "model.embed_tokens.weight"},
    {"output.weight",               "lm_head.weight"},
    {"output_norm.weight",          "model.norm.weight"},

    {"blk.{i}.attn_norm.weight",    "model.layers.{i}.input_layernorm.weight"},
    {"blk.{i}.attn_q.weight",       "model.layers.{i}.self_attn.q_proj.weight"},
    {"blk.{i}.attn_k.weight",       "model.layers.{i}.self_attn.k_proj.weight"},
    {"blk.{i}.attn_v.weight",       "model.layers.{i}.self_attn.v_proj.weight"},
    {"blk.{i}.attn_output.weight",  "model.layers.{i}.self_attn.o_proj.weight"},
    {"blk.{i}.attn_q_norm.weight",  "model.layers.{i}.self_attn.q_norm.weight"},
    {"blk.{i}.attn_k_norm.weight",  "model.layers.{i}.self_attn.k_norm.weight"},

    {"blk.{i}.ffn_norm.weight",            "model.layers.{i}.post_attention_layernorm.weight"},
    {"blk.{i}.post_attention_norm.weight", "model.layers.{i}.post_attention_layernorm.weight"},

    {"blk.{i}.attn_qkv.weight",    "model.layers.{i}.linear_attn.in_proj_qkv.weight"},
    {"blk.{i}.attn_gate.weight",   "model.layers.{i}.linear_attn.in_proj_z.weight"},
    {"blk.{i}.ssm_conv1d.weight",  "model.layers.{i}.linear_attn.conv1d.weight"},
    {"blk.{i}.ssm_dt.bias",         "model.layers.{i}.linear_attn.dt_bias"},
    {"blk.{i}.ssm_dt",              "model.layers.{i}.linear_attn.dt_bias"},
    {"blk.{i}.ssm_a",               "model.layers.{i}.linear_attn.A_log"},
    {"blk.{i}.ssm_a.weight",        "model.layers.{i}.linear_attn.A_log"},
    {"blk.{i}.ssm_beta.weight",     "model.layers.{i}.linear_attn.in_proj_b.weight"},
    {"blk.{i}.ssm_alpha.weight",    "model.layers.{i}.linear_attn.in_proj_a.weight"},
    {"blk.{i}.ssm_norm.weight",     "model.layers.{i}.linear_attn.norm.weight"},
    {"blk.{i}.ssm_out.weight",      "model.layers.{i}.linear_attn.out_proj.weight"},

    {"blk.{i}.ffn_gate_inp.weight", "model.layers.{i}.mlp.gate.weight"},
    {"blk.{i}.ffn_gate.weight",     "model.layers.{i}.mlp.gate_proj.weight"},
    {"blk.{i}.ffn_up.weight",       "model.layers.{i}.mlp.up_proj.weight"},
    {"blk.{i}.ffn_down.weight",     "model.layers.{i}.mlp.down_proj.weight"},
    {"blk.{i}.ffn_gate_up.weight",  "model.layers.{i}.mlp.gate_up_proj.weight"},

    {"blk.{i}.ffn_gate_exps.weight", "model.layers.{i}.mlp.experts.gate_proj.weight"},
    {"blk.{i}.ffn_up_exps.weight",   "model.layers.{i}.mlp.experts.up_proj.weight"},
    {"blk.{i}.ffn_down_exps.weight", "model.layers.{i}.mlp.experts.down_proj.weight"},

    {"blk.{i}.ffn_gate_shexp.weight",      "model.layers.{i}.mlp.shared_expert.gate_proj.weight"},
    {"blk.{i}.ffn_up_shexp.weight",        "model.layers.{i}.mlp.shared_expert.up_proj.weight"},
    {"blk.{i}.ffn_down_shexp.weight",      "model.layers.{i}.mlp.shared_expert.down_proj.weight"},
    {"blk.{i}.ffn_gate_inp_shexp.weight",  "model.layers.{i}.mlp.shared_expert_gate.weight"},
    {"blk.{i}.ffn_gate_shexp_gate.weight", "model.layers.{i}.mlp.shared_expert_gate.weight"},

    {"blk.{i}.nextn.eh_proj.weight",          "model.layers.{i}.nextn.eh_proj.weight"},
    {"blk.{i}.nextn.enorm.weight",            "model.layers.{i}.nextn.en_norm.weight"},
    {"blk.{i}.nextn.hnorm.weight",            "model.layers.{i}.nextn.h_norm.weight"},
    {"blk.{i}.nextn.embed_tokens.weight",     "model.layers.{i}.nextn.embed_tokens.weight"},
    {"blk.{i}.nextn.shared_head_head.weight", "model.layers.{i}.nextn.shared_head_head.weight"},
    {"blk.{i}.nextn.shared_head_norm.weight", "model.layers.{i}.nextn.shared_head_norm.weight"},
};

bool extract_layer_index(
    const std::string& concrete_name,
    const std::string& template_key,
    int num_layers,
    std::string* layer_num) {
    const auto pos = template_key.find("{i}");
    if (pos == std::string::npos) {
        return false;
    }
    const std::string prefix = template_key.substr(0, pos);
    const std::string suffix = template_key.substr(pos + 3);
    if (concrete_name.size() < prefix.size() + suffix.size()) {
        return false;
    }
    if (concrete_name.substr(0, prefix.size()) != prefix ||
        concrete_name.substr(concrete_name.size() - suffix.size()) != suffix) {
        return false;
    }
    const std::string num_str = concrete_name.substr(
        prefix.size(), concrete_name.size() - prefix.size() - suffix.size());
    try {
        const int layer = std::stoi(num_str);
        if (layer < 0 || layer >= num_layers) {
            return false;
        }
        if (layer_num) {
            *layer_num = num_str;
        }
        return true;
    } catch (...) {
        return false;
    }
}

std::string apply_layer_index(const std::string& template_key, const std::string& layer_num) {
    std::string out = template_key;
    const auto pos = out.find("{i}");
    if (pos != std::string::npos) {
        out.replace(pos, 3, layer_num);
    }
    return out;
}

void collect_gguf_candidates_from_vmc(
    const std::string& vmc_name,
    int num_layers,
    std::vector<std::string>& out) {
    for (const auto& [gguf_key, vmc_key] : kQwenGgufToVmc) {
        std::string layer_num;
        if (extract_layer_index(vmc_name, vmc_key, num_layers, &layer_num)) {
            out.push_back(apply_layer_index(gguf_key, layer_num));
        } else if (vmc_name == vmc_key) {
            out.push_back(gguf_key);
        }
    }
}

}  // namespace

std::string gguf_tensor_name_to_vmc(const std::string& gguf_name, int num_layers) {
    for (const auto& [gguf_key, vmc_key] : kQwenGgufToVmc) {
        std::string layer_num;
        if (extract_layer_index(gguf_name, gguf_key, num_layers, &layer_num)) {
            return apply_layer_index(vmc_key, layer_num);
        }
        if (gguf_name == gguf_key) {
            return vmc_key;
        }
    }
    return gguf_name;
}

std::string vmc_tensor_name_to_llama_gguf(
    const std::string& vmc_name,
    int num_layers,
    const std::unordered_set<std::string>* llama_names) {
    std::vector<std::string> candidates;
    collect_gguf_candidates_from_vmc(vmc_name, num_layers, candidates);
    if (llama_names) {
        for (const auto& candidate : candidates) {
            if (llama_names->count(candidate) > 0) {
                return candidate;
            }
        }
    }
    if (!candidates.empty()) {
        return candidates.front();
    }
    return vmc_name;
}

}  // namespace vm_c
