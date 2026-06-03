#include "vm_c/model/forward_context.hpp"
#include "vm_c/cuda/gpu_arch.hpp"
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace vm_c {

void ForwardContext::allocate(const BufferSizeInfo& sizes, int max_num_tokens_in,
                              int hidden_size_in, int gpu_dev) {
    if (gpu_device >= 0) {
        free();
    }

    gpu_device = gpu_dev;
    max_num_tokens = max_num_tokens_in;
    hidden_size = hidden_size_in;
    activation_elem_size = gpu().compute_dtype_size();
    residual_capacity_bytes = sizes.residual_bytes;
    CUDA_CHECK(cudaSetDevice(gpu_device));

    const size_t total_peak = sizes.peak_bytes();

    // ── 从 GPU 分配单一 arena ──
    if (total_peak > 0) {
        CUDA_CHECK(cudaMalloc(&arena_, total_peak));
    }
    char* base = static_cast<char*>(arena_);

    // ── 辅助：在 arena 内子分配（256 字节对齐）──
    size_t cur = 0;
    auto sub_alloc = [&](size_t bytes) -> void* {
        if (bytes == 0) return nullptr;
        size_t aligned = (bytes + 255) & ~255;
        void* ptr = base + cur;
        cur += aligned;
        return ptr;
    };

    // ════════════════════════════════════════════════════════════
    // 持久区域（整个 forward pass 生命周期内始终存活）
    // ════════════════════════════════════════════════════════════
    buf_residual = sub_alloc(sizes.residual_bytes);
    const size_t shared_region_start = cur;  // 对齐后的实际偏移
    const size_t shared_size = total_peak - shared_region_start;

    // ── 辅助：重置 bump 指针到共享区域起点 ──
    auto reset_to_shared = [&]() { cur = shared_region_start; };

    // ════════════════════════════════════════════════════════════
    // 共享区域：Attention / MoE / Dense / GDN 时间复用
    // 各阶段从 shared_region_start 开始，同一物理显存由不同阶段顺序使用
    // ════════════════════════════════════════════════════════════

    // ── Attention 阶段 ──
    reset_to_shared();
    buf_q    = sub_alloc(sizes.q_bytes);
    buf_k    = sub_alloc(sizes.k_bytes);
    buf_v    = sub_alloc(sizes.v_bytes);
    buf_attn_out = sub_alloc(sizes.attn_out_bytes);
    // deq buffer 顺序使用（Q→K→V→O 投影），取最大值别名到同一地址
    {
        size_t deq_max = std::max({sizes.deq_q_bytes, sizes.deq_k_bytes,
                                   sizes.deq_v_bytes, sizes.deq_o_bytes});
        void* deq_base = sub_alloc(deq_max);
        deq_q = deq_base;
        deq_k = deq_base;
        deq_v = deq_base;
        deq_o = deq_base;
    }
    const size_t attn_end = cur;  // 仅调试用

    // ── MoE 阶段 ──
    reset_to_shared();
    moe_router_logits       = sub_alloc(sizes.moe_router_logits_bytes);
    moe_softmax_workspace   = sub_alloc(sizes.moe_softmax_workspace_bytes);
    moe_topk_weights        = sub_alloc(sizes.moe_topk_weights_bytes);
    moe_topk_indices        = sub_alloc(sizes.moe_topk_indices_bytes);
    moe_token_expert_indices = sub_alloc(sizes.moe_token_expert_indices_bytes);
    moe_expert_offsets      = static_cast<int32_t*>(sub_alloc(sizes.moe_expert_offsets_bytes));
    moe_weight_offsets      = static_cast<int32_t*>(sub_alloc(sizes.moe_weight_offsets_bytes));
    moe_temp_counters       = static_cast<int32_t*>(sub_alloc(sizes.moe_temp_counters_bytes));
    moe_expert_input        = sub_alloc(sizes.moe_expert_input_bytes);
    moe_sorted_weights      = sub_alloc(sizes.moe_sorted_weights_bytes);
    moe_output_fp32         = static_cast<float*>(sub_alloc(sizes.moe_output_fp32_bytes));

    // Per-expert buffer：gate_up / mid / output 三者 stride 不同，不可别名。
    // gate_up stride = moe_inter*2，output stride = hidden_size（通常 hidden >> moe_inter*2），
    // 若强制 alias 则 prefill 176 expert 时 output 写入越界 → cudaErrorInvalidResourceHandle。
    moe_expert_gate_up      = sub_alloc(sizes.moe_expert_gate_up_bytes);
    moe_expert_mid          = sub_alloc(sizes.moe_expert_mid_bytes);
    moe_expert_output       = sub_alloc(sizes.moe_expert_output_bytes);

    moe_output              = sub_alloc(sizes.moe_output_bytes);
    moe_weighted_output     = sub_alloc(sizes.moe_weighted_output_bytes);
    moe_shared_gate_buf     = sub_alloc(sizes.moe_shared_gate_buf_bytes);
    moe_shared_up_buf       = sub_alloc(sizes.moe_shared_up_buf_bytes);
    moe_shared_gate_up      = sub_alloc(sizes.moe_shared_gate_up_bytes);
    moe_shared_out          = sub_alloc(sizes.moe_shared_out_bytes);
    moe_shared_gate_logits  = sub_alloc(sizes.moe_shared_gate_logits_bytes);
    const size_t moe_end = cur;  // 仅调试用

    // ── Dense FFN 阶段 ──
    reset_to_shared();
    buf_mlp_gate    = sub_alloc(sizes.mlp_gate_bytes);
    buf_mlp_up      = sub_alloc(sizes.mlp_up_bytes);
    buf_mlp_gate_up = sub_alloc(sizes.mlp_gate_up_bytes);
    buf_mlp_down    = sub_alloc(sizes.mlp_down_bytes);
    const size_t dense_end = cur;  // 仅调试用

    // ── GDN 临时 buffer 阶段 ──
    reset_to_shared();
    gdn_qkvz_buf    = sub_alloc(sizes.gdn_qkvz_buf_bytes);
    gdn_ba_buf      = sub_alloc(sizes.gdn_ba_buf_bytes);
    gdn_q_buf       = sub_alloc(sizes.gdn_q_buf_bytes);
    gdn_k_buf       = sub_alloc(sizes.gdn_k_buf_bytes);
    gdn_v_buf       = sub_alloc(sizes.gdn_v_buf_bytes);
    gdn_z_buf       = sub_alloc(sizes.gdn_z_buf_bytes);
    gdn_gate_buf    = sub_alloc(sizes.gdn_gate_buf_bytes);
    gdn_beta_buf    = sub_alloc(sizes.gdn_beta_buf_bytes);
    gdn_alpha_buf   = sub_alloc(sizes.gdn_alpha_buf_bytes);
    gdn_conv_out_buf = sub_alloc(sizes.gdn_conv_out_buf_bytes);
    gdn_core_out_buf = sub_alloc(sizes.gdn_core_out_buf_bytes);
    gdn_normed_out_buf = sub_alloc(sizes.gdn_normed_out_buf_bytes);
    gdn_prefill_f32_scratch = sub_alloc(sizes.gdn_prefill_f32_scratch_bytes);
    const size_t gdn_end = cur;  // 仅调试用

    // 验证：各阶段都不应超出 shared_region 边界
    auto phase_total = [&](size_t phase_end) {
        return phase_end - shared_region_start;
    };
    auto check_phase = [&](const char* name, size_t phase_end) {
        size_t total = phase_total(phase_end);
        if (total > shared_size) {
            throw std::runtime_error(
                std::string("ForwardContext arena: phase ") + name +
                " requires " + std::to_string(total / (1024 * 1024)) + " MB but shared region is " +
                std::to_string(shared_size / (1024 * 1024)) + " MB");
        }
    };
    check_phase("attention", attn_end);
    check_phase("moe", moe_end);
    check_phase("dense", dense_end);
    check_phase("gdn", gdn_end);

    if (cur > total_peak) {
        throw std::runtime_error(
            "ForwardContext arena: allocated " + std::to_string(cur / (1024 * 1024)) +
            " MB exceeds arena " + std::to_string(total_peak / (1024 * 1024)) + " MB");
    }

    spdlog::info("ForwardContext: arena {:.1f} MB (persistent {:.1f} MB + shared {:.1f} MB)",
                 total_peak / (1024.0 * 1024.0),
                 shared_region_start / (1024.0 * 1024.0),
                 shared_size / (1024.0 * 1024.0));
    spdlog::info("  phase sizes: attn={:.1f} MB, moe={:.1f} MB, dense={:.1f} MB, gdn={:.1f} MB",
                 phase_total(attn_end) / (1024.0 * 1024.0),
                 phase_total(moe_end) / (1024.0 * 1024.0),
                 phase_total(dense_end) / (1024.0 * 1024.0),
                 phase_total(gdn_end) / (1024.0 * 1024.0));
    spdlog::info("  (equivalent naive sum would be {:.1f} MB)",
                 sizes.total_bytes() / (1024.0 * 1024.0));
}

void ForwardContext::free() {
    if (gpu_device < 0) return;  // 尚未分配，无需释放
    CUDA_CHECK(cudaSetDevice(gpu_device));

    // GDN 状态 buffer 由外部单独管理（engine.cpp / tp_runtime.cpp），在此释放指针
    for (auto* ptr : gdn_ssm_states) {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
    }
    gdn_ssm_states.clear();
    for (auto* ptr : gdn_conv_states) {
        if (ptr) CUDA_CHECK(cudaFree(ptr));
    }
    gdn_conv_states.clear();

    // 单一 arena 释放（替代逐个 buffer 的 cudaFree）
    if (arena_) {
        CUDA_CHECK(cudaFree(arena_));
        arena_ = nullptr;
    }

    // 所有指针置空（它们指向 arena 内偏移，arena 释放后全部失效）
    buf_residual = nullptr;
    buf_q = nullptr; buf_k = nullptr; buf_v = nullptr; buf_attn_out = nullptr;
    buf_mlp_gate = nullptr; buf_mlp_up = nullptr; buf_mlp_gate_up = nullptr; buf_mlp_down = nullptr;
    deq_q = nullptr; deq_k = nullptr; deq_v = nullptr; deq_o = nullptr;
    moe_router_logits = nullptr; moe_topk_weights = nullptr;
    moe_sorted_weights = nullptr; moe_topk_indices = nullptr;
    moe_token_expert_indices = nullptr; moe_output = nullptr;
    moe_expert_input = nullptr; moe_expert_gate_up = nullptr;
    moe_expert_mid = nullptr; moe_expert_output = nullptr;
    moe_weighted_output = nullptr; moe_shared_gate_buf = nullptr;
    moe_shared_up_buf = nullptr; moe_shared_gate_up = nullptr;
    moe_shared_out = nullptr; moe_shared_gate_logits = nullptr;
    moe_softmax_workspace = nullptr; moe_output_fp32 = nullptr;
    moe_expert_offsets = nullptr; moe_weight_offsets = nullptr;
    moe_temp_counters = nullptr;
    gdn_qkvz_buf = nullptr; gdn_ba_buf = nullptr;
    gdn_q_buf = nullptr; gdn_k_buf = nullptr; gdn_v_buf = nullptr; gdn_z_buf = nullptr;
    gdn_gate_buf = nullptr; gdn_beta_buf = nullptr; gdn_alpha_buf = nullptr;
    gdn_conv_out_buf = nullptr; gdn_core_out_buf = nullptr; gdn_normed_out_buf = nullptr;
    gdn_prefill_f32_scratch = nullptr;
    max_num_tokens = 0;
    hidden_size = 0;
    activation_elem_size = 0;
    residual_capacity_bytes = 0;
}

size_t ForwardContext::activation_bytes(int num_tokens) const {
    if (num_tokens <= 0 || hidden_size <= 0 || activation_elem_size == 0) {
        return 0;
    }
    return static_cast<size_t>(num_tokens) * static_cast<size_t>(hidden_size) *
           activation_elem_size;
}

void ForwardContext::clear_residual(int num_tokens, cudaStream_t stream) const {
    if (num_tokens <= 0) {
        return;
    }
    if (!buf_residual) {
        throw std::runtime_error(
            "ForwardContext::clear_residual: buf_residual is null "
            "(max_num_tokens=" + std::to_string(max_num_tokens) + ")");
    }
    if (max_num_tokens > 0 && num_tokens > max_num_tokens) {
        throw std::runtime_error(
            "ForwardContext::clear_residual: num_tokens=" + std::to_string(num_tokens) +
            " exceeds allocated max_num_tokens=" + std::to_string(max_num_tokens));
    }
    const size_t bytes = activation_bytes(num_tokens);
    if (bytes == 0 || bytes > residual_capacity_bytes) {
        throw std::runtime_error(
            "ForwardContext::clear_residual: bytes=" + std::to_string(bytes) +
            " exceeds residual_capacity_bytes=" +
            std::to_string(residual_capacity_bytes));
    }
    CUDA_CHECK(cudaMemsetAsync(buf_residual, 0, bytes, stream));
}

void ForwardContext::copy_pre_norm_hidden(void* dst, const void* src, int num_tokens,
                                        cudaStream_t stream) const {
    if (!dst || !src || num_tokens <= 0) {
        return;
    }
    if (max_num_tokens > 0 && num_tokens > max_num_tokens) {
        throw std::runtime_error(
            "ForwardContext::copy_pre_norm_hidden: num_tokens=" +
            std::to_string(num_tokens) + " exceeds max_num_tokens=" +
            std::to_string(max_num_tokens));
    }
    const size_t bytes = activation_bytes(num_tokens);
    if (bytes == 0) {
        return;
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
}

} // namespace vm_c