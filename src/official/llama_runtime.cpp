#include "vm_c/official/llama_runtime.hpp"
#include "vm_c/official/turboquant_kv_bridge.hpp"
#include "vm_c/cuda/vm_c_tensor.h"
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <spdlog/spdlog.h>

#include <stdexcept>
#include <algorithm>

namespace vm_c::official {

bool LlamaRuntime::backend_initialized_ = false;

namespace {

void vm_c_llama_log_callback(ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_NONE) {
        return;
    }
    fputs(text, stderr);
    fflush(stderr);
}

}  // namespace

void LlamaRuntime::ensure_backend() {
    if (backend_initialized_) {
        return;
    }
    llama_backend_init();
    llama_log_set(vm_c_llama_log_callback, nullptr);
    register_ggml_turboquant_attn_kernels();
    // allreduce 回调由 NcclComm::init 注册（nccl_comm.cpp），不再在这里注册
    backend_initialized_ = true;
    spdlog::info("LlamaRuntime: llama_backend_init done");
}

LlamaRuntime::~LlamaRuntime() {
    free_all();
}

void LlamaRuntime::free_all() {
    if (ctx_mtp_) {
        llama_free(ctx_mtp_);
        ctx_mtp_ = nullptr;
    }
    if (ctx_main_) {
        llama_free(ctx_main_);
        ctx_main_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    bound_weights_.clear();
}

llama_context* LlamaRuntime::select_ctx(bool mtp) const {
    if (mtp) {
        if (!ctx_mtp_) {
            throw std::runtime_error("LlamaRuntime: MTP context not loaded");
        }
        return ctx_mtp_;
    }
    if (!ctx_main_) {
        throw std::runtime_error("LlamaRuntime: main context not loaded");
    }
    return ctx_main_;
}

void LlamaRuntime::decode(const llama_batch& batch, bool mtp) {
    llama_context* ctx = select_ctx(mtp);
    const int32_t rc = llama_decode(ctx, batch);
    if (rc != 0) {
        throw std::runtime_error(
            "LlamaRuntime::decode failed: rc=" + std::to_string(rc) +
            (mtp ? " (MTP context)" : " (main context)"));
    }
}

float* LlamaRuntime::logits_ith(int index, bool mtp) const {
    llama_context* ctx = select_ctx(mtp);
    float* logits = llama_get_logits_ith(ctx, index);
    if (!logits) {
        throw std::runtime_error(
            "LlamaRuntime::logits_ith: null logits at index " + std::to_string(index));
    }
    return logits;
}

float* LlamaRuntime::embeddings_pre_norm_ith(int index, bool mtp) const {
    llama_context* ctx = select_ctx(mtp);
    float* emb = llama_get_embeddings_pre_norm_ith(ctx, index);
    if (!emb) {
        throw std::runtime_error(
            "LlamaRuntime::embeddings_pre_norm_ith: null at index " + std::to_string(index));
    }
    return emb;
}

void LlamaRuntime::memory_clear(bool data) {
    llama_memory_t mem = llama_get_memory(ctx_main_);
    if (!mem) {
        throw std::runtime_error("LlamaRuntime::memory_clear: llama_get_memory returned null");
    }
    llama_memory_clear(mem, data);
}

void LlamaRuntime::memory_seq_rm(int32_t seq_id, int32_t p0, int32_t p1) {
    llama_memory_t mem = llama_get_memory(ctx_main_);
    if (!mem) {
        throw std::runtime_error("LlamaRuntime::memory_seq_rm: llama_get_memory returned null");
    }
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        throw std::runtime_error(
            "LlamaRuntime::memory_seq_rm failed: seq=" + std::to_string(seq_id) +
            " p0=" + std::to_string(p0) + " p1=" + std::to_string(p1));
    }
}

void LlamaRuntime::memory_seq_add(int32_t seq_id, int32_t p0, int32_t p1, int32_t shift) {
    llama_memory_t mem = llama_get_memory(ctx_main_);
    if (!mem) {
        throw std::runtime_error("LlamaRuntime::memory_seq_add: llama_get_memory returned null");
    }
    llama_memory_seq_add(mem, seq_id, p0, p1, shift);
}

void LlamaRuntime::memory_seq_rm_mtp(int32_t seq_id, int32_t p0, int32_t p1) {
    if (!ctx_mtp_) {
        throw std::runtime_error("LlamaRuntime::memory_seq_rm_mtp: MTP context not loaded");
    }
    llama_memory_t mem = llama_get_memory(ctx_mtp_);
    if (!mem) {
        throw std::runtime_error("LlamaRuntime::memory_seq_rm_mtp: llama_get_memory returned null");
    }
    if (!llama_memory_seq_rm(mem, seq_id, p0, p1)) {
        throw std::runtime_error(
            "LlamaRuntime::memory_seq_rm_mtp failed: seq=" + std::to_string(seq_id) +
            " p0=" + std::to_string(p0) + " p1=" + std::to_string(p1));
    }
}

int LlamaRuntime::n_embd() const {
    if (!model_) {
        throw std::runtime_error("LlamaRuntime::n_embd: model not loaded");
    }
    return llama_model_n_embd(model_);
}

int LlamaRuntime::n_vocab() const {
    if (!model_) {
        throw std::runtime_error("LlamaRuntime::n_vocab: model not loaded");
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model_);
    if (!vocab) {
        throw std::runtime_error("LlamaRuntime::n_vocab: vocab not available");
    }
    return llama_vocab_n_tokens(vocab);
}

void LlamaRuntime::set_turboquant_bridge(TurboQuantKvBridge* bridge) {
    tq_bridge_ = bridge;
}

void LlamaRuntime::set_tensor_parallel(void* tp, int tp_size) {
    vm_c_tp_ = tp;
    vm_c_tp_size_ = tp_size > 0 ? tp_size : 1;
}

namespace {

}  // namespace

void LlamaRuntime::load(const VmCConfig& config, int gpu_device,
                        std::unordered_map<std::string, GpuTensor> vm_c_weights) {
    ensure_backend();
    free_all();

    if (config.model.model_dir.empty()) {
        throw std::runtime_error("LlamaRuntime::load: model_dir (GGUF path) is empty");
    }
    if (vm_c_weights.empty()) {
        throw std::runtime_error("LlamaRuntime::load: vm_c weight snapshot is empty");
    }

    bound_weights_ = std::move(vm_c_weights);

    llama_model_params mparams = llama_model_default_params();
    // vm_c ModelLoader 已在 GPU 上加载 TP 分片权重；llama 仅解析 GGUF 元数据建图，
    // 随后 rebind 到 vm_c buffer。no_alloc 使 load_tensors 只建 dummy buffer、不 cudaMalloc 权重。
    mparams.no_alloc = true;
    // 对齐 llama.cpp 默认（model_default_params.n_gpu_layers=-1）：仅用于 dev_layer 映射，
    // 使 hybrid GDN 的 llama_memory_recurrent RS buffer 与 offload_kqv 落到 GPU。
    // 不可设为 0：会把全部 dev_layer 标为 CPU，RS 错误地绑 CPU（与 no_alloc 无关）。
    mparams.n_gpu_layers = -1;
    // no_alloc 与 llama-model.cpp mmap host_ptr buffer 路径互斥（GGML_ASSERT(!ml.no_alloc)）
    mparams.use_mmap = false;
    // SPLIT_MODE_NONE 下 main_gpu 是 llama 设备列表下标；vm_c TP rank 的 cuda ordinal 与之对应
    mparams.main_gpu = gpu_device;
    mparams.use_mlock = false;
    // vm_c TP 为 column-parallel lockstep（每 rank 单 GPU + 分片权重），
    // 非 llama LLAMA_SPLIT_MODE_ROW 层切分。
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;

    model_ = llama_model_load_from_file(config.model.model_dir.c_str(), mparams);
    if (!model_) {
        throw std::runtime_error(
            "LlamaRuntime::load: llama_model_load_from_file failed: " + config.model.model_dir);
    }

    bind_vm_c_weights_to_llama_model(model_, bound_weights_);
    spdlog::info(
        "LlamaRuntime: bound {} vm_c weight tensors (no_alloc load, gpu={})",
        bound_weights_.size(), gpu_device);

    const bool has_mtp = config.model.mtp_predict_layers > 0;
    const uint32_t spec_width = has_mtp ? static_cast<uint32_t>(config.model.mtp_spec_width) : 0;

    // ============================================================
    // 动态参数计算系统（基于模型元数据 + GPU 显存）
    // ============================================================

    // 1. 从模型元数据获取基础参数
    const int32_t n_ctx_train = llama_model_n_ctx_train(model_);
    const int32_t n_embd      = llama_model_n_embd(model_);
    const int32_t n_layer     = llama_model_n_layer(model_);
    const int32_t n_head      = llama_model_n_head(model_);
    const int32_t n_head_kv   = llama_model_n_head_kv(model_);

    spdlog::info("LlamaRuntime: Model metadata - n_ctx_train={}, n_embd={}, n_layer={}, n_head={}, n_head_kv={}",
            n_ctx_train, n_embd, n_layer, n_head, n_head_kv);

    // 2. 检测 GPU 显存（确保在正确的 GPU 上查询）
    size_t gpu_free_bytes = 0;
    size_t gpu_total_bytes = 0;
    {
        CudaDeviceGuard guard(gpu_device);
        cudaError_t err = cudaMemGetInfo(&gpu_free_bytes, &gpu_total_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                "LlamaRuntime: Failed to query GPU memory info on device " +
                std::to_string(gpu_device) + ": " + cudaGetErrorString(err));
        }
        spdlog::info("LlamaRuntime: GPU {} - free={:.1f} MiB, total={:.1f} MiB",
                gpu_device, gpu_free_bytes / (1024.0*1024.0), gpu_total_bytes / (1024.0*1024.0));
    }

    // 3. 计算可用于 llama context 的显存预算
    //    使用实际空闲显存（已扣除权重、CUDA context、NCCL 等所有已分配内存），
    //    再预留 (1 - mem_util) 比例的总显存作为安全边界。
    if (config.cache.gpu_memory_utilization <= 0.0f || config.cache.gpu_memory_utilization > 1.0f) {
        throw std::runtime_error(
            "LlamaRuntime: gpu_memory_utilization must be in (0, 1], got " +
            std::to_string(config.cache.gpu_memory_utilization));
    }
    const float mem_util = config.cache.gpu_memory_utilization;
    const size_t current_usage = gpu_total_bytes - gpu_free_bytes;
    const size_t max_total_usage = static_cast<size_t>(gpu_total_bytes * mem_util);

    if (current_usage >= max_total_usage) {
        throw std::runtime_error(
            "LlamaRuntime: current GPU memory usage (" +
            std::to_string(current_usage / (1024*1024)) +
            " MiB) already exceeds utilization limit (" +
            std::to_string(max_total_usage / (1024*1024)) +
            " MiB, utilization=" + std::to_string(static_cast<int>(mem_util * 100)) +
            "%). Reduce model size or increase --gpu-memory-utilization.");
    }
    const size_t context_budget_bytes = max_total_usage - current_usage;

    size_t model_bytes_for_log = 0;
    for (const auto& [name, tensor] : bound_weights_) {
        model_bytes_for_log += tensor.nbytes();
    }

    spdlog::info("LlamaRuntime: GPU memory budget - total={:.1f} MiB, current_usage={:.1f} MiB "
                 "(weights={:.1f} MiB + overhead={:.1f} MiB), max_usage={:.1f} MiB (utilization={:.0f}%), "
                 "context_budget={:.1f} MiB",
            gpu_total_bytes / (1024.0*1024.0),
            current_usage / (1024.0*1024.0),
            model_bytes_for_log / (1024.0*1024.0),
            (current_usage - model_bytes_for_log) / (1024.0*1024.0),
            max_total_usage / (1024.0*1024.0),
            mem_util * 100,
            context_budget_bytes / (1024.0*1024.0));

    // 4. 计算 KV cache 每 token 每层的显存
    // KV cache: 2 * n_layer * n_head_kv * n_embd_head_k * sizeof(type) * n_ctx
    // [BUG-FIX] 之前用 n_embd / n_head 反推 head_dim 是错的（Qwen3.5-A3B: 2048/16=128，
    // 实际 attention.key_length=256）。改用 llama_model_n_embd_head_k()（来自 GGUF
    // attention.key_length），fallback 才用 n_embd/n_head。这导致 KV cache 显存预算
    // 少算一半，进而把动态 n_ctx 也算小。
    int32_t kv_head_dim_i = llama_model_n_embd_head_k(model_);
    if (kv_head_dim_i <= 0 && n_head > 0) {
        kv_head_dim_i = n_embd / n_head;
    }
    if (kv_head_dim_i <= 0) {
        kv_head_dim_i = 128;  // 兜底，与旧行为一致
    }
    const size_t kv_head_dim = static_cast<size_t>(kv_head_dim_i);

    // 根据 --kv-cache-dtype 动态计算 KV cache 每元素字节数
    const auto& kv_dtype_str = config.cache.kv_cache_quant_method.empty()
        ? config.cache.kv_cache_dtype : config.cache.kv_cache_quant_method;

    double kv_element_bytes = sizeof(ggml_fp16_t);
    if (kv_dtype_str.find("turboquant_4bit") != std::string::npos) {
        kv_element_bytes = 4.0 / 8.0;
    } else if (kv_dtype_str.find("turboquant_3bit") != std::string::npos) {
        kv_element_bytes = 3.0 / 8.0;
    } else if (kv_dtype_str.find("turboquant_k8v4") != std::string::npos) {
        kv_element_bytes = 4.0 / 8.0;
    } else if (kv_dtype_str.find("turboquant_k3v4_nc") != std::string::npos) {
        kv_element_bytes = 4.0 / 8.0;
    } else if (kv_dtype_str == "fp8_e4m3" || kv_dtype_str == "fp8_e5m2") {
        kv_element_bytes = 1.0;
    } else if (kv_dtype_str == "bf16" || kv_dtype_str == "fp16") {
        kv_element_bytes = 2.0;
    } else if (kv_dtype_str == "auto") {
        kv_element_bytes = 2.0;
    }

    const double kv_cache_per_token_per_layer = 2.0 * n_head_kv * kv_head_dim * kv_element_bytes;
    const double kv_cache_per_token = kv_cache_per_token_per_layer * n_layer;

    // MTP context 的 KV cache 使用 BF16/FP16（与主 context 的 type_k/type_v 相同），
    // 不使用 TurboQuant（vm_c_kv_quant_method = nullptr），且仅有 mtp_predict_layers 层。
    // MTP KV cache 每层字节数 = 主 KV cache 每层字节数 * (mtp_element / main_element)，
    // 例如 TurboQuant 4bit 主 KV 每层 0.5B/elem，MTP 每层 2B/elem，MTP 是主 KV 的 4 倍。
    // 注意：MTP context 继承主 context 的 type_k/type_v，而 TurboQuant 场景下 type_k 为
    // BF16/FP16（compute dtype），因此 MTP KV element 始终为 2 bytes。
    double mtp_kv_cache_per_token = 0.0;
    if (has_mtp) {
        const double mtp_kv_element_bytes = sizeof(ggml_fp16_t);
        const double mtp_kv_cache_per_token_per_layer = 2.0 * n_head_kv * kv_head_dim * mtp_kv_element_bytes;
        mtp_kv_cache_per_token = mtp_kv_cache_per_token_per_layer * config.model.mtp_predict_layers;
    }

    const double total_kv_cache_per_token = kv_cache_per_token + mtp_kv_cache_per_token;

    spdlog::info("LlamaRuntime: KV cache per token = {:.2f} KiB (main={:.2f} KiB, mtp={:.2f} KiB, "
                 "dtype={}, element_bytes={:.3f}, head_dim={} (n_embd/n_head fallback={}))",
            total_kv_cache_per_token / 1024.0,
            kv_cache_per_token / 1024.0,
            mtp_kv_cache_per_token / 1024.0,
            kv_dtype_str, kv_element_bytes,
            (int)kv_head_dim, (n_head > 0 ? n_embd / n_head : -1));

    // 5. 动态计算 n_ctx
    int32_t n_ctx = (config.model.max_model_len > 0)
        ? static_cast<int32_t>(config.model.max_model_len)
        : n_ctx_train;

    // compute buffer 预留：每个 context（main / MTP）各需一份 compute buffer，
    // 每份大小取决于模型结构和 n_ubatch，通常为模型权重的 5-10%。
    // gpu_safety_margin_mb 为单个 context 的 compute buffer 预留量，
    // 有 MTP 时需要双份。
    const size_t compute_reserve_per_ctx = static_cast<size_t>(config.cache.gpu_safety_margin_mb) * 1024 * 1024;
    const size_t total_compute_reserve = has_mtp ? compute_reserve_per_ctx * 2 : compute_reserve_per_ctx;

    // 如果用户指定的 n_ctx 超出预算，自动减小
    const double total_kv_cache_bytes = total_kv_cache_per_token * n_ctx;
    const size_t kv_budget = context_budget_bytes > total_compute_reserve
        ? context_budget_bytes - total_compute_reserve : 0;
    if (total_kv_cache_bytes > static_cast<double>(kv_budget)) {
        const int32_t max_n_ctx = static_cast<int32_t>(static_cast<double>(kv_budget) / total_kv_cache_per_token);
        const int32_t original_n_ctx = n_ctx;
        n_ctx = std::max(max_n_ctx, ServerConfig::MIN_MODEL_LEN);
        spdlog::warn("LlamaRuntime: n_ctx reduced from {} to {} due to memory budget "
                     "(requested_kv={:.1f} MiB, kv_budget={:.1f} MiB, compute_reserve={:.1f} MiB)",
                     original_n_ctx, n_ctx,
                     total_kv_cache_bytes / (1024.0*1024.0),
                     kv_budget / (1024.0*1024.0),
                     total_compute_reserve / (1024.0*1024.0));
    }

    // 6. 动态计算 n_batch 和 n_ubatch
    // n_batch 影响预填充阶段的并行度，n_ubatch 影响 decode 阶段的并行度
    int32_t n_batch = (config.server.max_num_batched_tokens > 0)
        ? static_cast<int32_t>(config.server.max_num_batched_tokens)
        : std::min(n_ctx, ServerConfig::DEFAULT_MAX_NUM_BATCHED_TOKENS);

    int32_t n_ubatch = (config.server.max_num_batched_tokens > 0)
        ? std::min(static_cast<int32_t>(config.server.max_num_batched_tokens), ServerConfig::DEFAULT_UBATCH_SIZE)
        : ServerConfig::DEFAULT_UBATCH_SIZE;

    // 确保 n_ubatch <= n_batch
    n_ubatch = std::min(n_ubatch, n_batch);

    // 7. n_seq_max
    int32_t n_seq_max = static_cast<int32_t>(std::max(1, config.server.max_num_seqs));

    // ============================================================
    // 创建 context 参数（官方逻辑：llama-context.cpp + common.cpp）
    // ============================================================

    llama_context_params cparams = llama_context_default_params();
    cparams.samplers    = nullptr;
    cparams.n_samplers  = 0;
    cparams.n_threads = 1;
    cparams.n_threads_batch = 1;

    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = n_batch;
    cparams.n_ubatch  = n_ubatch;
    cparams.n_seq_max = n_seq_max;

    // KV cache 类型（根据 --kv-cache-dtype 动态设置）
    const auto& kv_dtype_for_llama = config.cache.kv_cache_quant_method.empty()
        ? config.cache.kv_cache_dtype : config.cache.kv_cache_quant_method;

    if (kv_dtype_for_llama.find("turboquant") != std::string::npos) {
        // TurboQuant 量化：llama context 使用 compute dtype（BF16/FP16）作为内部表示，
        // 实际量化通过 vm_c_tq_bridge 处理
        cparams.type_k = gpu().supports_bf16() ? GGML_TYPE_BF16 : GGML_TYPE_F16;
        cparams.type_v = gpu().supports_bf16() ? GGML_TYPE_BF16 : GGML_TYPE_F16;
    } else if (kv_dtype_for_llama == "fp8_e4m3" || kv_dtype_for_llama == "fp8_e5m2") {
        throw std::runtime_error(
            "LlamaRuntime: FP8 KV cache dtype ('" + kv_dtype_for_llama +
            "') is not supported by the current ggml build. "
            "Supported types: fp16, bf16, turboquant_*");
    } else if (kv_dtype_for_llama == "bf16") {
        cparams.type_k = GGML_TYPE_BF16;
        cparams.type_v = GGML_TYPE_BF16;
    } else if (kv_dtype_for_llama == "fp16" || kv_dtype_for_llama == "auto") {
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
    } else {
        // 默认 F16
        cparams.type_k = GGML_TYPE_F16;
        cparams.type_v = GGML_TYPE_F16;
    }

    spdlog::info("LlamaRuntime: KV cache type set to k={}, v={} (from --kv-cache-dtype={})",
            ggml_type_name(cparams.type_k), ggml_type_name(cparams.type_v), kv_dtype_for_llama);

    cparams.offload_kqv = true;
    cparams.n_rs_seq = spec_width;

    spdlog::info("LlamaRuntime: Context params - n_ctx={}, n_batch={}, n_ubatch={}, n_seq_max={}, n_rs_seq={}",
            cparams.n_ctx, cparams.n_batch, cparams.n_ubatch, cparams.n_seq_max, cparams.n_rs_seq);

#ifdef VM_C_LLAMA_INTEGRATION
    // TurboQuant KV bridge：始终启用（当指定了量化方法时）
    if (!config.cache.kv_cache_quant_method.empty()) {
        cparams.vm_c_kv_quant_method = config.cache.kv_cache_quant_method.c_str();
        cparams.vm_c_tq_bridge = tq_bridge_;
        spdlog::info("LlamaRuntime: TurboQuant KV bridge enabled (method={})",
                     config.cache.kv_cache_quant_method);
    }
    // Tensor Parallel：始终启用（当 tp_size > 1 时）
    if (vm_c_tp_ && vm_c_tp_size_ > 1) {
        cparams.vm_c_tp = vm_c_tp_;
        cparams.vm_c_tp_size = vm_c_tp_size_;
        spdlog::info("LlamaRuntime: Tensor Parallel enabled (size={})", vm_c_tp_size_);
    }
#endif

    // ============================================================
    // 创建主 context（官方顺序：先主 context，再 MTP context）
    // ============================================================

    ctx_main_ = llama_init_from_model(model_, cparams);
    if (!ctx_main_) {
        throw std::runtime_error(
            "LlamaRuntime::load: llama_init_from_model (main) failed. "
            "Insufficient GPU memory for n_ctx=" + std::to_string(cparams.n_ctx) +
            ", n_ubatch=" + std::to_string(cparams.n_ubatch) +
            ". Try reducing --max-model-len or increasing --gpu-memory-utilization.");
    }

    if (config.model.mtp_predict_layers > 0) {
        llama_set_embeddings_pre_norm(ctx_main_, true, false);
    }

    {
        auto bd = llama_get_memory_breakdown(ctx_main_);
        size_t total_context = 0, total_compute = 0;
        for (auto & [buft, data] : bd) {
            total_context += data.context;
            total_compute += data.compute;
        }
        spdlog::info("LlamaRuntime: main context memory: context={:.1f} MiB, compute={:.1f} MiB",
                total_context / (1024.0*1024.0), total_compute / (1024.0*1024.0));
    }

    // ============================================================
    // 创建 MTP context（官方逻辑：server-context.cpp:943-950）
    // ============================================================

    if (has_mtp) {
        llama_context_params mtp_cparams = cparams;
        mtp_cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        mtp_cparams.n_rs_seq = 0;
        mtp_cparams.vm_c_kv_quant_method = cparams.vm_c_kv_quant_method;
        mtp_cparams.vm_c_tq_bridge = cparams.vm_c_tq_bridge;

#ifdef VM_C_LLAMA_INTEGRATION
        if (vm_c_tp_ && vm_c_tp_size_ > 1) {
            mtp_cparams.vm_c_tp = vm_c_tp_;
            mtp_cparams.vm_c_tp_size = vm_c_tp_size_;
        }
#endif

        spdlog::info("LlamaRuntime: Creating MTP context - n_batch={}, n_ubatch={}, n_ctx={}, n_rs_seq={}",
                mtp_cparams.n_batch, mtp_cparams.n_ubatch, mtp_cparams.n_ctx, mtp_cparams.n_rs_seq);

        ctx_mtp_ = llama_init_from_model(model_, mtp_cparams);
        if (!ctx_mtp_) {
            throw std::runtime_error("LlamaRuntime::load: llama_init_from_model (MTP) failed");
        }
        llama_set_embeddings_pre_norm(ctx_mtp_, true, true);

        size_t mtp_compute_bytes = 0;
        size_t mtp_context_bytes = 0;
        {
            auto bd = llama_get_memory_breakdown(ctx_mtp_);
            for (auto & [buft, data] : bd) {
                mtp_compute_bytes += data.compute;
                mtp_context_bytes += data.context;
            }
            spdlog::info("LlamaRuntime: MTP context memory (actual) - context={:.1f} MiB, compute={:.1f} MiB",
                    mtp_context_bytes / (1024.0*1024.0), mtp_compute_bytes / (1024.0*1024.0));
        }

    }

    spdlog::info(
        "LlamaRuntime: loaded {} gpu={} n_gpu_layers={} no_alloc=1 tp={} ctx={} n_batch={} n_ubatch={} mtp_ctx={}",
        config.model.model_dir, gpu_device, mparams.n_gpu_layers,
        config.parallel.tensor_parallel_size, config.model.max_model_len,
        llama_n_batch(ctx_main_), llama_n_ubatch(ctx_main_),
        config.model.mtp_predict_layers > 0 ? "yes" : "no");
}

}  // namespace vm_c::official
