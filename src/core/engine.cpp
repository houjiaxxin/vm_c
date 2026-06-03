#include "vm_c/core/engine.hpp"
#include "vm_c/core/logits_sample_plan.hpp"
#include "llama.h"
#include <spdlog/spdlog.h>
#include "vm_c/model/attention_metadata.hpp"
#include "vm_c/model/models.hpp"
#include "vm_c/cuda/kernels.hpp"
#include "vm_c/cuda/kernels_sampler.h"
#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/official/llama_runtime.hpp"
#include "vm_c/cuda/convert_kernels.h"
#include "vm_c/core/ggml_dequant.hpp"

#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
#include "vm_c/official/ggml_backend_pool.hpp"
#endif

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>
#include <thread>
#include <future>
#include <functional>

namespace vm_c {

namespace {

int32_t decode_input_token(const Request& req) {
    if (!req.output_token_ids.empty()) {
        return req.output_token_ids.back();
    }
    if (!req.prompt_token_ids.empty()) {
        return req.prompt_token_ids.back();
    }
    throw std::runtime_error("request " + req.request_id + " has no input token for decode");
}

void fill_prefill_input_ids(const Request& req, int num_sched, int start_pos,
                            std::vector<int32_t>& host_input_ids, int token_offset) {
    if (req.prompt_token_ids.empty()) {
        throw std::runtime_error("request " + req.request_id + " has empty prompt for prefill");
    }
    if (start_pos < 0 || start_pos >= static_cast<int>(req.prompt_token_ids.size())) {
        throw std::runtime_error(
            "request " + req.request_id + " invalid prefill start_pos=" + std::to_string(start_pos));
    }
    if (start_pos + num_sched > static_cast<int>(req.prompt_token_ids.size())) {
        throw std::runtime_error(
            "request " + req.request_id + " prefill over-scheduled: start_pos=" +
            std::to_string(start_pos) + " num_sched=" + std::to_string(num_sched));
    }
    for (int i = 0; i < num_sched; ++i) {
        host_input_ids[static_cast<size_t>(token_offset + i)] =
            req.prompt_token_ids[static_cast<size_t>(start_pos + i)];
    }
}

std::vector<int32_t> build_batch_seq_ids(int total_tokens, const int32_t* cu_seqlens,
                                          const std::vector<RequestPtr>& all_reqs) {
    std::vector<int32_t> seq_ids(static_cast<std::size_t>(total_tokens), 0);
    int tok = 0;
    for (int ri = 0; ri < (int)all_reqs.size(); ++ri) {
        const int ns = cu_seqlens[ri + 1] - cu_seqlens[ri];
        const int32_t sid = all_reqs[ri]->seq_id_;
        for (int j = 0; j < ns; ++j) {
            seq_ids[static_cast<std::size_t>(tok++)] = sid;
        }
    }
    return seq_ids;
}

void copy_llama_logits_rows(
    official::LlamaRuntime& llama_rt,
    const LogitsSamplePlan& plan,
    void* d_logits,
    int vocab_size,
    int gpu_device,
    cudaStream_t stream) {
    for (int si = 0; si < plan.num_logits(); ++si) {
        const auto& slot = plan.slots[static_cast<std::size_t>(si)];
        float* src = llama_rt.logits_ith(slot.hidden_batch_index);
        float* dst = static_cast<float*>(d_logits)
            + static_cast<std::size_t>(slot.logits_row) * static_cast<std::size_t>(vocab_size);
        CudaDeviceGuard guard(gpu_device);
        CUDA_CHECK(cudaMemcpyAsync(
            dst, src,
            static_cast<std::size_t>(vocab_size) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }
}

void copy_llama_pre_norm_rows(
    official::LlamaRuntime& llama_rt,
    int num_tokens,
    int hidden_size,
    void* d_pre_norm_hidden,
    int gpu_device,
    cudaStream_t stream) {
    if (!d_pre_norm_hidden || num_tokens <= 0) {
        return;
    }
    CudaDeviceGuard guard(gpu_device);
    // [BUG-FIX] d_pre_norm_hidden 已改为 f32 buffer (见 Engine::initialize), 直接 D2D 拷贝即可
    // 旧实现 launch_gpu_convert 写入 compute_dtype 元素 (bf16/fp16), 但下游 D2H 按 sizeof(float) 读
    // 出现 2 倍字节错位, 越界读到 buffer 后段未初始化内存, 解析出 garbage f32
    // 修复: src 本身就是 f32 (embeddings_pre_norm_ith 返回 f32 device ptr), 走 D2D memcpy
    float* dst = static_cast<float*>(d_pre_norm_hidden);
    for (int i = 0; i < num_tokens; ++i) {
        float* src = llama_rt.embeddings_pre_norm_ith(i);
        CUDA_CHECK(cudaMemcpyAsync(
            dst + static_cast<int64_t>(i) * hidden_size,
            src,
            static_cast<size_t>(hidden_size) * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
    }
}

}  // namespace

Engine::Engine(const VmCConfig& config) : config_(config), tp_rank_(config.parallel.tp_rank), tp_size_(config.parallel.tensor_parallel_size) {}

void Engine::EngineBuffers::ensure(int num_tokens, int max_num_seqs, int hidden_size,
                                    int vocab_size, int max_blocks_per_req, int num_reqs,
                                    int gpu_device, int min_lm_gather_rows) {
    const int lm_rows = std::max(max_num_seqs, std::max(1, min_lm_gather_rows));
    const int logits_rows = lm_rows;

    if (num_tokens > capacity) {
        if (d_input_ids) CUDA_CHECK(cudaFree(d_input_ids));
        if (d_position_ids) CUDA_CHECK(cudaFree(d_position_ids));
        if (d_hidden_states) CUDA_CHECK(cudaFree(d_hidden_states));
        if (d_seq_lens) CUDA_CHECK(cudaFree(d_seq_lens));
        if (d_slot_mapping) CUDA_CHECK(cudaFree(d_slot_mapping));
        if (d_token_to_seq) CUDA_CHECK(cudaFree(d_token_to_seq));
        if (d_token_positions) CUDA_CHECK(cudaFree(d_token_positions));

        set_device_safe(gpu_device);
        d_input_ids = cuda_alloc(num_tokens * sizeof(int32_t));
        d_position_ids = cuda_alloc(num_tokens * sizeof(int64_t));
        d_hidden_states = cuda_alloc(
            static_cast<size_t>(num_tokens) * static_cast<size_t>(hidden_size) * gpu().compute_dtype_size());
        d_seq_lens = cuda_alloc(max_num_seqs * sizeof(int32_t));
        d_slot_mapping = cuda_alloc(num_tokens * sizeof(int64_t));
        d_token_to_seq = cuda_alloc(num_tokens * sizeof(int32_t));
        d_token_positions = cuda_alloc(num_tokens * sizeof(int32_t));
        capacity = num_tokens;
    }

    int block_tables_needed = num_reqs * max_blocks_per_req;
    if (block_tables_needed > block_tables_size) {
        if (d_block_tables) CUDA_CHECK(cudaFree(d_block_tables));
        set_device_safe(gpu_device);
        d_block_tables = cuda_alloc(block_tables_needed * sizeof(int32_t));
        block_tables_size = block_tables_needed;
    }

    int logits_needed = logits_rows * vocab_size;
    if (logits_needed > logits_size) {
        if (d_logits) CUDA_CHECK(cudaFree(d_logits));
        set_device_safe(gpu_device);
        d_logits = cuda_alloc(static_cast<size_t>(logits_needed) * sizeof(float));
        logits_size = logits_needed;
    }

    int lm_gather_needed = lm_rows * hidden_size;
    if (lm_gather_needed > lm_gather_size) {
        if (d_lm_gather) CUDA_CHECK(cudaFree(d_lm_gather));
        set_device_safe(gpu_device);
        d_lm_gather = cuda_alloc(
            static_cast<size_t>(lm_gather_needed) * gpu().compute_dtype_size());
        lm_gather_size = lm_gather_needed;
    }

    if (max_num_seqs > sample_buf_size) {
        if (d_output_ids) CUDA_CHECK(cudaFree(d_output_ids));
        if (d_output_logprobs) CUDA_CHECK(cudaFree(d_output_logprobs));
        if (d_temperatures) CUDA_CHECK(cudaFree(d_temperatures));
        if (d_top_k) CUDA_CHECK(cudaFree(d_top_k));
        if (d_top_p) CUDA_CHECK(cudaFree(d_top_p));
        if (d_min_p) CUDA_CHECK(cudaFree(d_min_p));
        set_device_safe(gpu_device);
        d_output_ids = cuda_alloc(max_num_seqs * sizeof(int32_t));
        d_output_logprobs = cuda_alloc(max_num_seqs * sizeof(float));
        d_temperatures = cuda_alloc(max_num_seqs * sizeof(float));
        d_top_k = cuda_alloc(max_num_seqs * sizeof(int32_t));
        d_top_p = cuda_alloc(max_num_seqs * sizeof(float));
        d_min_p = cuda_alloc(max_num_seqs * sizeof(float));
        sample_buf_size = max_num_seqs;
    }

    int penalty_needed = max_num_seqs * vocab_size;
    if (penalty_needed > penalty_buf_size) {
        if (d_prompt_mask) CUDA_CHECK(cudaFree(d_prompt_mask));
        if (d_output_bin_counts) CUDA_CHECK(cudaFree(d_output_bin_counts));
        if (d_rep_penalties) CUDA_CHECK(cudaFree(d_rep_penalties));
        if (d_freq_penalties) CUDA_CHECK(cudaFree(d_freq_penalties));
        if (d_pres_penalties) CUDA_CHECK(cudaFree(d_pres_penalties));
        set_device_safe(gpu_device);
        d_prompt_mask = static_cast<uint8_t*>(cuda_alloc(penalty_needed * sizeof(uint8_t)));
        d_output_bin_counts = static_cast<int32_t*>(cuda_alloc(penalty_needed * sizeof(int32_t)));
        d_rep_penalties = static_cast<float*>(cuda_alloc(max_num_seqs * sizeof(float)));
        d_freq_penalties = static_cast<float*>(cuda_alloc(max_num_seqs * sizeof(float)));
        d_pres_penalties = static_cast<float*>(cuda_alloc(max_num_seqs * sizeof(float)));
        penalty_buf_size = penalty_needed;
    }

    // Staging buffer for MTP verify sampling（vocab_size floats，用于 penalties 和采样）
    const int staging_needed = vocab_size;
    if (staging_needed > verify_staging_size) {
        if (d_verify_staging) CUDA_CHECK(cudaFree(d_verify_staging));
        set_device_safe(gpu_device);
        d_verify_staging = cuda_alloc(static_cast<size_t>(staging_needed) * sizeof(float));
        verify_staging_size = staging_needed;
    }
}

void Engine::EngineBuffers::free() {
    if (d_input_ids) { CUDA_CHECK(cudaFree(d_input_ids)); d_input_ids = nullptr; }
    if (d_position_ids) { CUDA_CHECK(cudaFree(d_position_ids)); d_position_ids = nullptr; }
    if (d_hidden_states) { CUDA_CHECK(cudaFree(d_hidden_states)); d_hidden_states = nullptr; }
    if (d_logits) { CUDA_CHECK(cudaFree(d_logits)); d_logits = nullptr; }
    if (d_lm_gather) { CUDA_CHECK(cudaFree(d_lm_gather)); d_lm_gather = nullptr; }
    if (d_block_tables) { CUDA_CHECK(cudaFree(d_block_tables)); d_block_tables = nullptr; }
    if (d_seq_lens) { CUDA_CHECK(cudaFree(d_seq_lens)); d_seq_lens = nullptr; }
    if (d_slot_mapping) { CUDA_CHECK(cudaFree(d_slot_mapping)); d_slot_mapping = nullptr; }
    if (d_token_to_seq) { CUDA_CHECK(cudaFree(d_token_to_seq)); d_token_to_seq = nullptr; }
    if (d_token_positions) { CUDA_CHECK(cudaFree(d_token_positions)); d_token_positions = nullptr; }
    if (d_output_ids) { CUDA_CHECK(cudaFree(d_output_ids)); d_output_ids = nullptr; }
    if (d_output_logprobs) { CUDA_CHECK(cudaFree(d_output_logprobs)); d_output_logprobs = nullptr; }
    if (d_temperatures) { CUDA_CHECK(cudaFree(d_temperatures)); d_temperatures = nullptr; }
    if (d_top_k) { CUDA_CHECK(cudaFree(d_top_k)); d_top_k = nullptr; }
    if (d_top_p) { CUDA_CHECK(cudaFree(d_top_p)); d_top_p = nullptr; }
    if (d_min_p) { CUDA_CHECK(cudaFree(d_min_p)); d_min_p = nullptr; }
    if (d_pre_norm_hidden) { CUDA_CHECK(cudaFree(d_pre_norm_hidden)); d_pre_norm_hidden = nullptr; }
    if (d_prompt_mask) { CUDA_CHECK(cudaFree(d_prompt_mask)); d_prompt_mask = nullptr; }
    if (d_output_bin_counts) { CUDA_CHECK(cudaFree(d_output_bin_counts)); d_output_bin_counts = nullptr; }
    if (d_rep_penalties) { CUDA_CHECK(cudaFree(d_rep_penalties)); d_rep_penalties = nullptr; }
    if (d_freq_penalties) { CUDA_CHECK(cudaFree(d_freq_penalties)); d_freq_penalties = nullptr; }
    if (d_pres_penalties) { CUDA_CHECK(cudaFree(d_pres_penalties)); d_pres_penalties = nullptr; }
    if (d_verify_staging) { CUDA_CHECK(cudaFree(d_verify_staging)); d_verify_staging = nullptr; }
    capacity = 0;
    block_tables_size = 0;
    logits_size = 0;
    lm_gather_size = 0;
    sample_buf_size = 0;
    penalty_buf_size = 0;
}

Engine::~Engine() {
    shutdown();
}

void Engine::initialize() {
    if (initialized_) return;

    spdlog::info("Initializing vm_c engine...");
    spdlog::info("[INIT] config.gpu_device={} tp_size={}",
                 gpu_device_, config_.parallel.tensor_parallel_size);

    int tp_size = config_.parallel.tensor_parallel_size;
    if (tp_size < 1) {
        tp_size = 1;
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    spdlog::info("[INIT] cudaGetDeviceCount returned {} (tp_size={})", device_count, tp_size);
    if (device_count < tp_size) {
        spdlog::error("Tensor parallel size {} exceeds available GPUs {}", tp_size, device_count);
        throw std::runtime_error("Not enough GPUs for tensor parallel");
    }

    spdlog::info("[INIT] Loading model config...");
    resolve_model_config();

    // 确定 GPU 设备列表：优先使用 --gpu-devices，否则从 0 开始连续分配
    int base_gpu_device = 0;
    if (!config_.parallel.gpu_devices.empty()) {
        if (static_cast<int>(config_.parallel.gpu_devices.size()) < tp_size) {
            spdlog::error("--gpu-devices specified {} devices, but tensor-parallel-size is {}",
                         config_.parallel.gpu_devices.size(), tp_size);
            throw std::runtime_error("Not enough GPU devices specified for tensor parallel");
        }
        spdlog::info("[INIT] Using GPU devices from --gpu-devices:");
        for (size_t i = 0; i < config_.parallel.gpu_devices.size(); ++i) {
            spdlog::info("  rank {} -> GPU {}", i, config_.parallel.gpu_devices[i]);
        }
        base_gpu_device = config_.parallel.gpu_devices[0];
    } else {
        spdlog::info("[INIT] Using default GPU device allocation (rank 0 -> GPU 0, rank N -> GPU N)");
    }

    tp_runtime_ = std::make_unique<TPRuntime>(tp_size, base_gpu_device);
    tp_runtime_->initialize(config_);

    spdlog::info("[INIT] Loading weights for all ranks (parallel)...");
    std::vector<std::future<void>> load_futures;
    load_futures.reserve(tp_size);
    for (int i = 0; i < tp_size; ++i) {
        load_futures.push_back(std::async(std::launch::async, [this, i, tp_size]() {
            auto& ctx = tp_runtime_->rank_ctx(i);
            CudaDeviceGuard guard(ctx.gpu_device);
            auto rank_loader = std::make_unique<ModelLoader>(config_.model, config_.cache, ctx.gpu_device, i, tp_size);
            auto weights = rank_loader->load_weights();
            ctx.llama_weight_snapshot = std::move(weights.gpu_weights);
            prepare_llama_weight_snapshot(
                ctx.llama_weight_snapshot, config_.model, ctx.gpu_device);
            ctx.uva_buffers = rank_loader->release_uva_buffers();
        }));
    }
    for (auto& f : load_futures) {
        f.get();
    }

    if (config_.model.spec_method == "mtp" && config_.model.mtp_predict_layers > 0) {
        auto& ctx0 = tp_runtime_->rank_ctx(0);
        CudaDeviceGuard guard0(ctx0.gpu_device);
        speculative_engine_ = std::make_unique<SpeculativeEngine>(
            config_.model, ctx0.gpu_device, 0, tp_size);
        speculative_engine_->allocate_buffers(config_.server.max_num_batched_tokens);
        // [BUG-FIX] d_pre_norm_hidden 改为 f32 buffer, 配合下游 D2H 按 sizeof(float) 读取
        // 旧实现: alloc 按 compute_dtype_size (bf16=2) + launch_gpu_convert 写 compute_dtype
        //         + D2H 按 sizeof(float) 读 4 字节/元素, 出现 2 倍字节错位, 越界读到 garbage
        // 修复: 改用 f32 buffer, copy_llama_pre_norm_rows 直接 D2D 拷贝 f32 数据
        // (embeddings_pre_norm_ith 返回 f32 device ptr, 已是正确格式)
        size_t pre_norm_bytes = static_cast<size_t>(config_.model.hidden_size)
            * static_cast<size_t>(config_.server.max_num_batched_tokens)
            * sizeof(float);
        CUDA_CHECK(cudaMalloc(&engine_bufs_.d_pre_norm_hidden, pre_norm_bytes));
        spdlog::info("[INIT] MTP speculative engine ready (libllama path)");
    }

    spdlog::info("Tensor parallel initialized via TPRuntime: world_size={}", tp_size);

    gpu_device_ = tp_runtime_->rank_ctx(0).gpu_device;
    set_device_safe(gpu_device_);
#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
    compute_stream_ = official::ggml_bind_compute_stream(gpu_device_);
    compute_stream_owned_ = false;
#else
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    compute_stream_owned_ = true;
#endif

    spdlog::info("[INIT] Planning KV cache...");
    tp_runtime_->profile_gpu_memory(config_);

    spdlog::info("[INIT] Initializing KV cache...");
    tp_runtime_->initialize_kv_cache(config_);

    spdlog::info("[INIT] Initializing libllama official forward graph...");
    tp_runtime_->initialize_llama_graph(config_);

    spdlog::info("[INIT] Verifying GPU memory after libllama load...");
    tp_runtime_->verify_gpu_memory_after_llama(config_);

    kv_cache_mgr_ = tp_runtime_->rank_ctx(0).kv_cache_mgr;

    {
        const int max_num_seqs = config_.server.max_num_seqs > 0 ? config_.server.max_num_seqs : 1;
        const int max_blocks = static_cast<int>(tp_runtime_->kv_cache_blocks());
        const int mtp_min_gather = (speculative_engine_ && speculative_engine_->enabled())
            ? speculative_engine_->spec_width() + 1 : 0;
        CudaDeviceGuard guard(gpu_device_);
        engine_bufs_.ensure(config_.server.max_num_batched_tokens, max_num_seqs,
                            config_.model.hidden_size, config_.model.vocab_size,
                            max_blocks, max_num_seqs, gpu_device_, mtp_min_gather);
        spdlog::info("[INIT] TP rank0 engine buffers pre-allocated (capacity={} tokens)",
                     engine_bufs_.capacity);
    }

    // 初始化 seq_id pool（与 llama_context 的 n_seq_max 一致）
    {
        const int pool_size = config_.server.max_num_seqs > 0 ? config_.server.max_num_seqs : 1;
        seq_id_in_use_.assign(static_cast<std::size_t>(pool_size), false);
    }

    scheduler_ = std::make_unique<Scheduler>(config_, kv_cache_mgr_);

    running_ = true;
    initialized_ = true;
    spdlog::info("vm_c engine initialized successfully (libllama TP, {} GPU(s))", tp_size);
}

void Engine::shutdown() {
    if (!running_) return;
    running_ = false;

    set_device_safe(gpu_device_);

    if (compute_stream_) {
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    }

    speculative_engine_.reset();
    model_loader_.reset();
    tp_runtime_.reset();

    engine_bufs_.free();

    kv_cache_mgr_.reset();
    scheduler_.reset();

    if (compute_stream_ && compute_stream_owned_) {
        CUDA_CHECK(cudaStreamDestroy(compute_stream_));
        compute_stream_ = nullptr;
    }

    spdlog::info("vm_c engine shut down");
}

void Engine::resolve_model_config() {
    const bool cli_max_model_len = config_.model.max_model_len_from_cli;
    const int64_t cli_max_model_len_value = config_.model.max_model_len;

    model_loader_ = std::make_unique<ModelLoader>(config_.model, config_.cache, gpu_device_);
    auto model_config = model_loader_->load_config();
    int preserved_vocab = config_.model.vocab_size;
    config_.model = model_config;
    if (preserved_vocab < config_.model.vocab_size) {
        config_.model.vocab_size = preserved_vocab;
    }
    config_.model.block_size = config_.cache.block_size;
    config_.model.max_model_len_from_cli = cli_max_model_len;
    const int64_t hf_max_ctx = model_config.max_model_len;
    if (cli_max_model_len) {
        config_.model.max_model_len = std::min(hf_max_ctx, cli_max_model_len_value);
        if (cli_max_model_len_value > hf_max_ctx) {
            spdlog::warn("--max-model-len {} exceeds HF max_position_embeddings {}; using {}",
                         cli_max_model_len_value, hf_max_ctx, config_.model.max_model_len);
        }
    }

    if (config_.cache.kv_cache_dtype == "auto") {
        const auto& hint = config_.model.kv_cache_quant_method_hint;
        if (!hint.empty()) {
            config_.cache.kv_cache_dtype = hint;
            config_.cache.kv_cache_quant_method = hint;
            spdlog::info("kv_cache_dtype=auto resolved to '{}' from model quantization_config", hint);
        } else {
            config_.cache.kv_cache_dtype = config_.model.dtype_str;
            spdlog::info("kv_cache_dtype=auto resolved to '{}' from model dtype", config_.model.dtype_str);
        }
    }

    const std::string cli_cap_str =
        cli_max_model_len ? std::to_string(cli_max_model_len_value) : std::string("unset");
    spdlog::info("Sequence length cap max_model_len={} (HF max_position_embeddings={}, CLI={})",
                 config_.model.max_model_len, hf_max_ctx, cli_cap_str);

    if (config_.server.max_num_seqs <= 0) {
        config_.server.max_num_seqs = ServerConfig::DEFAULT_MAX_NUM_SEQS;
    }
    if (config_.server.max_num_batched_tokens <= 0) {
        config_.server.max_num_batched_tokens = std::max(
            ServerConfig::DEFAULT_MAX_NUM_BATCHED_TOKENS,
            static_cast<int>(config_.model.max_model_len));
    }
}

std::string Engine::submit_request(const std::vector<int32_t>& prompt_token_ids,
                                    const SamplingParams& params,
                                    const std::string& model_name,
                                    std::function<void(const std::vector<int32_t>&, bool, FinishReason)> stream_cb) {
    static std::atomic<int64_t> request_counter{0};
    std::string req_id = "req-" + std::to_string(request_counter.fetch_add(1));

    auto req = std::make_shared<Request>();
    req->request_id = req_id;
    req->prompt_token_ids = prompt_token_ids;
    req->sampling_params = params;
    req->model_name = model_name.empty() ? config_.server.served_model_name : model_name;
    req->stream_callback = std::move(stream_cb);
    req->arrival_time = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        pending_requests_.push(req);
    }
    queue_cv_.notify_one();

    return req_id;
}

void Engine::abort_request(const std::string& request_id) {
    scheduler_->abort_request(request_id);
}

std::vector<RequestOutput> Engine::step() {
    static int idle_steps = 0;
    static bool was_idle = true;

    while (!pending_requests_.empty()) {
        RequestPtr req;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (pending_requests_.empty()) break;
            req = pending_requests_.front();
            pending_requests_.pop();
        }
        scheduler_->add_request(req);
    }

    auto sched_output = scheduler_->schedule();

    if (sched_output.total_num_scheduled_tokens == 0) {
        return {};
    }

    idle_steps = 0;
    was_idle = false;

    std::vector<RequestOutput> outputs;
    outputs = run_forward(sched_output);

    // 清理已完成请求的 llama memory module + 归还 seq_id
    for (auto& out : outputs) {
        if (out.finished && tp_runtime_ && out.seq_id_ >= 0) {
            spdlog::info("[MTP-DEBUG] request finished: seq_id={} mtp_enabled={}",
                         out.seq_id_, speculative_engine_ && speculative_engine_->enabled());
            tp_runtime_->llama_main_seq_trim(out.seq_id_, 0);
            if (speculative_engine_ && speculative_engine_->enabled()) {
                tp_runtime_->llama_mtp_seq_clear(out.seq_id_);
            }
            release_seq_id(out.seq_id_);
            break;
        }
    }

    scheduler_->update_from_output(outputs);
    return outputs;
}

int32_t Engine::alloc_seq_id() {
    std::lock_guard<std::mutex> lock(seq_id_mtx_);
    for (int32_t i = 0; i < (int32_t)seq_id_in_use_.size(); ++i) {
        if (!seq_id_in_use_[static_cast<std::size_t>(i)]) {
            seq_id_in_use_[static_cast<std::size_t>(i)] = true;
            return i;
        }
    }
    throw std::runtime_error("no available seq_id (pool exhausted, max="
                             + std::to_string(seq_id_in_use_.size()) + ")");
}

void Engine::release_seq_id(int32_t sid) {
    if (sid < 0 || sid >= (int32_t)seq_id_in_use_.size()) return;
    std::lock_guard<std::mutex> lock(seq_id_mtx_);
    seq_id_in_use_[static_cast<std::size_t>(sid)] = false;
}

std::vector<RequestOutput> Engine::run_forward(const SchedulerOutput& sched_output) {
    CudaDeviceGuard guard(gpu_device_);
    auto t_fwd_start = std::chrono::steady_clock::now();

    int total_tokens = sched_output.total_num_scheduled_tokens;
    if (total_tokens == 0) return {};

    int hidden_size = config_.model.hidden_size;
    int vocab_size = config_.model.vocab_size;
    int64_t block_size = config_.cache.block_size;

    std::vector<RequestPtr> all_reqs;
    for (auto& r : sched_output.scheduled_new_reqs) all_reqs.push_back(r);
    for (auto& r : sched_output.scheduled_running_reqs) all_reqs.push_back(r);
    int num_reqs = static_cast<int>(all_reqs.size());

    // 为新请求分配 seq_id（从 pool 中取，上限由 n_seq_max / max_num_seqs 决定）
    for (auto& r : sched_output.scheduled_new_reqs) {
        if (r->seq_id_ < 0) {
            r->seq_id_ = alloc_seq_id();
        }
    }

    int decode_threshold = 1;
    if (kv_cache_mgr_->is_turboquant()) {
        decode_threshold = config_.cache.block_size;
    }

    struct ReqCategory {
        RequestPtr req;
        int num_sched;
        int category;
    };
    std::vector<ReqCategory> req_cats;
    req_cats.reserve(num_reqs);
    for (auto& r : all_reqs) {
        auto it = sched_output.num_scheduled_tokens.find(r->request_id);
        int ns = (it != sched_output.num_scheduled_tokens.end()) ? it->second : 1;
        bool has_context = r->num_computed_tokens > 0;
        bool is_below_threshold = ns <= decode_threshold;
        bool done_prefilling = !r->is_prefill();
        int cat;
        if (!has_context) {
            cat = 3;
        } else if (!is_below_threshold) {
            cat = 2;
        } else if (!done_prefilling) {
            cat = 1;
        } else {
            cat = 0;
        }
        req_cats.push_back({r, ns, cat});
    }
    std::stable_sort(req_cats.begin(), req_cats.end(),
                     [](const ReqCategory& a, const ReqCategory& b) {
                         return a.category < b.category;
                     });

    all_reqs.clear();
    for (auto& rc : req_cats) all_reqs.push_back(rc.req);

    int num_decodes = 0;
    int num_decode_tokens = 0;
    int num_prefills = 0;
    int num_prefill_tokens = 0;
    {
        int tok_offset = 0;
        for (auto& rc : req_cats) {
            if (rc.category <= 1) {
                num_decodes++;
                num_decode_tokens += rc.num_sched;
            } else {
                num_prefills++;
                num_prefill_tokens += rc.num_sched;
            }
            tok_offset += rc.num_sched;
        }
    }

    const bool mtp_enabled = speculative_engine_ && speculative_engine_->enabled();
    const bool mtp_candidate = mtp_enabled
        && tp_runtime_
        && num_reqs == 1 && total_tokens == 1 && !all_reqs[0]->is_prefill();
    bool run_mtp_verify = false;
    SpeculativeEngine::DraftOutput mtp_draft;
    const int mtp_kv_lookahead = mtp_candidate ? speculative_engine_->spec_width() : 0;

    int max_blocks_per_req = (config_.model.max_model_len + config_.cache.block_size - 1) / config_.cache.block_size;
    engine_bufs_.ensure(total_tokens, config_.server.max_num_seqs, hidden_size,
                        vocab_size,                         max_blocks_per_req, num_reqs, gpu_device_,
                        mtp_candidate ? speculative_engine_->spec_width() + 1 : 0);

    auto* input_ids = static_cast<int32_t*>(engine_bufs_.d_input_ids);
    auto* position_ids = static_cast<int64_t*>(engine_bufs_.d_position_ids);

    std::vector<int32_t> host_input_ids(total_tokens);
    std::vector<int64_t> host_position_ids(total_tokens);
    std::vector<int64_t> host_slot_mapping(total_tokens, -1);
    std::vector<int32_t> host_seq_lens(num_reqs, 0);
    std::vector<int32_t> host_token_to_seq(total_tokens, 0);
    std::vector<int32_t> host_token_positions(total_tokens, 0);
    int max_num_blocks_per_req = 0;

    std::vector<std::vector<int64_t>> req_block_ids(num_reqs);

    int token_offset = 0;
    for (int ri = 0; ri < num_reqs; ++ri) {
        auto& req = all_reqs[ri];
        auto it = sched_output.num_scheduled_tokens.find(req->request_id);
        int num_sched = (it != sched_output.num_scheduled_tokens.end()) ? it->second : 1;

        std::vector<uint64_t> prefix_hashes;
        if (sched_output.prefix_block_hashes.count(req->request_id)) {
            prefix_hashes = sched_output.prefix_block_hashes.at(req->request_id);
        }
        std::vector<int64_t> computed_bids;
        if (sched_output.computed_block_ids.count(req->request_id)) {
            computed_bids = sched_output.computed_block_ids.at(req->request_id);
        }
        int num_cached = 0;
        if (sched_output.num_cached_tokens.count(req->request_id)) {
            num_cached = sched_output.num_cached_tokens.at(req->request_id);
        }

        auto slots = kv_cache_mgr_->allocate_slots(
            req->request_id,
            req->num_computed_tokens + num_sched + mtp_kv_lookahead,
            prefix_hashes,
            num_cached,
            computed_bids);

        req_block_ids[ri].reserve(slots.size());
        for (auto& s : slots) req_block_ids[ri].push_back(s.block_id);
        max_num_blocks_per_req = std::max(max_num_blocks_per_req, static_cast<int>(slots.size()));

        if (req->is_prefill()) {
            int start_pos = req->num_computed_tokens;
            for (int i = 0; i < num_sched; ++i) {
                host_input_ids[token_offset + i] = req->prompt_token_ids[start_pos + i];
                host_position_ids[token_offset + i] = start_pos + i;
                host_token_to_seq[token_offset + i] = ri;
                host_token_positions[token_offset + i] = start_pos + i;
                int64_t global_slot = (start_pos + i) / block_size;
                int64_t block_offset = (start_pos + i) % block_size;
                if (global_slot < static_cast<int64_t>(slots.size())) {
                    host_slot_mapping[token_offset + i] =
                        slots[global_slot].block_id * block_size + block_offset;
                }
            }
        } else {
            int pos = req->num_computed_tokens;
            host_input_ids[token_offset] = req->output_token_ids.empty()
                ? req->prompt_token_ids.back()
                : req->output_token_ids.back();
            host_position_ids[token_offset] = pos;
            int64_t global_slot = pos / block_size;
            int64_t block_offset = pos % block_size;
            if (global_slot < static_cast<int64_t>(slots.size())) {
                host_slot_mapping[token_offset] =
                    slots[global_slot].block_id * block_size + block_offset;
            }
        }

        host_seq_lens[ri] = req->num_computed_tokens + num_sched;
        token_offset += num_sched;
    }

    std::vector<int32_t> host_cu_seqlens(num_reqs + 1, 0);
    for (int ri = 0; ri < num_reqs; ++ri) {
        auto it = sched_output.num_scheduled_tokens.find(all_reqs[ri]->request_id);
        int ns = (it != sched_output.num_scheduled_tokens.end()) ? it->second : 1;
        host_cu_seqlens[ri + 1] = host_cu_seqlens[ri] + ns;
    }

    int max_num_blocks = max_num_blocks_per_req;
    std::vector<int32_t> host_block_tables(num_reqs * max_num_blocks, -1);
    for (int ri = 0; ri < num_reqs; ++ri) {
        for (int bi = 0; bi < static_cast<int>(req_block_ids[ri].size()); ++bi) {
            host_block_tables[ri * max_num_blocks + bi] =
                static_cast<int32_t>(req_block_ids[ri][bi]);
        }
    }

    int32_t* d_block_tables = nullptr;
    int32_t* d_seq_lens = static_cast<int32_t*>(engine_bufs_.d_seq_lens);
    int64_t* d_slot_mapping = static_cast<int64_t*>(engine_bufs_.d_slot_mapping);
    int32_t* d_token_to_seq = static_cast<int32_t*>(engine_bufs_.d_token_to_seq);
    int32_t* d_token_positions = static_cast<int32_t*>(engine_bufs_.d_token_positions);

    if (max_num_blocks > 0) {
        d_block_tables = static_cast<int32_t*>(engine_bufs_.d_block_tables);
    }

    if (max_num_blocks > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_block_tables, host_block_tables.data(),
                                   num_reqs * max_num_blocks * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, compute_stream_));
    }
    CUDA_CHECK(cudaMemcpyAsync(d_seq_lens, host_seq_lens.data(),
                               num_reqs * sizeof(int32_t),
                               cudaMemcpyHostToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_slot_mapping, host_slot_mapping.data(),
                               total_tokens * sizeof(int64_t),
                               cudaMemcpyHostToDevice, compute_stream_));
    if (num_prefills > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_token_to_seq, host_token_to_seq.data(),
                                   total_tokens * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_token_positions, host_token_positions.data(),
                                   total_tokens * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, compute_stream_));
    }

    CUDA_CHECK(cudaMemcpyAsync(input_ids, host_input_ids.data(),
                               total_tokens * sizeof(int32_t),
                               cudaMemcpyHostToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpyAsync(position_ids, host_position_ids.data(),
                               total_tokens * sizeof(int64_t),
                               cudaMemcpyHostToDevice, compute_stream_));

    AttentionMetadata attn_meta;
    attn_meta.block_tables = d_block_tables;
    attn_meta.seq_lens = d_seq_lens;
    attn_meta.slot_mapping = d_slot_mapping;
    attn_meta.token_to_seq = d_token_to_seq;
    attn_meta.token_positions = d_token_positions;
    attn_meta.cu_seqlens = host_cu_seqlens.data();
    attn_meta.num_reqs = num_reqs;
    attn_meta.max_num_blocks_per_req = max_num_blocks;
    attn_meta.num_tokens = total_tokens;
    attn_meta.block_size = kv_cache_mgr_->block_size();
    attn_meta.is_prefill = (num_prefills > 0);
    attn_meta.num_decodes = num_decodes;
    attn_meta.num_decode_tokens = num_decode_tokens;
    attn_meta.num_prefills = num_prefills;
    attn_meta.num_prefill_tokens = num_prefill_tokens;
    attn_meta.decode_threshold = decode_threshold;
    attn_meta.host_token_to_seq = host_token_to_seq.data();

    if (!tp_runtime_) {
        throw std::runtime_error("Engine::run_forward requires TPRuntime (libllama path)");
    }

    std::unordered_map<int, std::vector<int32_t>> mtp_tokens_by_req;
    std::unordered_map<int, std::vector<float>> mtp_logprobs_by_req;
    std::unordered_map<int, int> mtp_kv_committed_by_req;

    LogitsSamplePlan sample_plan;
    int num_to_sample = 0;
    std::vector<int32_t> host_output_ids;
    std::vector<float> host_output_logprobs;

    if (mtp_candidate) {
        auto& req = all_reqs[0];
        const int32_t id_last = decode_input_token(*req);
        const int64_t n_past = req->num_computed_tokens;

        // 对齐 llama.cpp common/speculative.cpp: MTP draft 用 pending_h
        // (= prefill 末态的 h_{n_past-1}, 上一步 accept 留下的 h_{n_past-1}, shifted MTP 设计)。
        // 不要再做"初始主 forward"——主 ctx KV 在 n_past 位置只能写一次,
        // 由 step 末尾的 verify forward 统一写。
        const float* pending_h = speculative_engine_->pending_h(0);
        if (!pending_h) {
            throw std::runtime_error("MTP decode: pending_h not allocated");
        }

        speculative_engine_->begin_draft_kv_checkpoint(n_past);

        mtp_draft = speculative_engine_->draft(
            pending_h, id_last, n_past,
            tp_runtime_.get(), compute_stream_);

        speculative_engine_->rollback_draft_kv(tp_runtime_.get());

        if (mtp_draft.success && !mtp_draft.draft_tokens.empty()) {
            run_mtp_verify = true;
        }
    }

    if (run_mtp_verify) {
        auto& req = all_reqs[0];
        const int req_ri = 0;
        const int32_t id_last = decode_input_token(*req);
        const int64_t n_past = req->num_computed_tokens;
        const auto& draft = mtp_draft;

        const int n_draft = static_cast<int>(draft.draft_tokens.size());
        const int n_batch = 1 + n_draft;
        const int n_logits_rows = n_draft + 1;

        std::vector<int32_t> host_v_ids(static_cast<std::size_t>(n_batch));
        std::vector<int64_t> host_v_pos(static_cast<std::size_t>(n_batch));
        std::vector<int64_t> host_v_slots(static_cast<std::size_t>(n_batch));
        host_v_ids[0] = id_last;
        host_v_pos[0] = n_past;
        for (int i = 0; i < n_draft; ++i) {
            host_v_ids[static_cast<std::size_t>(1 + i)] = draft.draft_tokens[static_cast<std::size_t>(i)];
            host_v_pos[static_cast<std::size_t>(1 + i)] = n_past + 1 + i;
        }

        const auto& blocks = req_block_ids[req_ri];
        for (int i = 0; i < n_batch; ++i) {
            const int64_t p = host_v_pos[static_cast<std::size_t>(i)];
            const int64_t bi = p / block_size;
            if (bi < 0 || bi >= static_cast<int64_t>(blocks.size())) {
                throw std::runtime_error(
                    "MTP verify: invalid slot for position " + std::to_string(p)
                    + " request " + req->request_id);
            }
            host_v_slots[static_cast<std::size_t>(i)] =
                blocks[static_cast<std::size_t>(bi)] * block_size + (p % block_size);
        }

        engine_bufs_.ensure(n_batch, config_.server.max_num_seqs, hidden_size,
                            vocab_size, max_num_blocks_per_req, 1, gpu_device_,
                            n_logits_rows);

        std::vector<int32_t> host_v_token_to_seq(static_cast<std::size_t>(n_batch), 0);
        std::vector<int32_t> host_v_token_positions(static_cast<std::size_t>(n_batch));
        for (int i = 0; i < n_batch; ++i) {
            host_v_token_positions[static_cast<std::size_t>(i)] =
                static_cast<int32_t>(host_v_pos[static_cast<std::size_t>(i)]);
        }
        const int32_t host_v_seq_len = static_cast<int32_t>(n_past + n_batch);

        CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_input_ids, host_v_ids.data(),
                                   static_cast<std::size_t>(n_batch) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_position_ids, host_v_pos.data(),
                                   static_cast<std::size_t>(n_batch) * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_slot_mapping, host_v_slots.data(),
                                   static_cast<std::size_t>(n_batch) * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_seq_lens, &host_v_seq_len, sizeof(int32_t),
                                   cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_token_to_seq, host_v_token_to_seq.data(),
                                   static_cast<std::size_t>(n_batch) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_token_positions, host_v_token_positions.data(),
                                   static_cast<std::size_t>(n_batch) * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, compute_stream_));

        LogitsSamplePlan verify_sample_plan;
        for (int i = 0; i < n_logits_rows; ++i) {
            LogitsSampleSlot slot;
            slot.req_index = 0;
            slot.hidden_batch_index = i;
            slot.logits_row = i;
            verify_sample_plan.slots.push_back(slot);
        }

        {
            std::vector<int32_t> batch_positions(static_cast<std::size_t>(n_batch));
            for (int i = 0; i < n_batch; ++i) {
                batch_positions[static_cast<std::size_t>(i)] =
                    static_cast<int32_t>(host_v_pos[static_cast<std::size_t>(i)]);
            }
            const int32_t verify_seq_id = req->seq_id_;
            const std::vector<int32_t> batch_seq_ids(static_cast<std::size_t>(n_batch), verify_seq_id);
            std::vector<int8_t> batch_logits_flags(static_cast<std::size_t>(n_batch), 1);

            // mtp=false: verify 跑主模型（ctx_main_），更新主模型 KV 并获取 verify logits。
            // MTP 上下文（ctx_mtp_）仅由 draft() 走 forward_llama_mtp_draft 推进。
            tp_runtime_->forward_llama(
                host_v_ids, batch_positions, batch_seq_ids, batch_logits_flags,
                d_slot_mapping, d_block_tables, d_seq_lens,
                d_token_to_seq, d_token_positions,
                n_batch, 1, max_num_blocks_per_req, block_size,
                false, n_batch, 0, compute_stream_);

            copy_llama_pre_norm_rows(
                *tp_runtime_->rank_ctx(0).llama_runtime,
                n_batch, hidden_size, engine_bufs_.d_pre_norm_hidden,
                gpu_device_, compute_stream_);
        }

        // Copy pre-norm hidden states from device to host for MTP accept
        // (对齐 llama.cpp speculative.cpp: verify_h[seq_id] = h_tgt[seq_id][0..n_batch))
        std::vector<float> host_pre_norm_hidden(
            static_cast<size_t>(n_batch) * static_cast<size_t>(hidden_size));
        {
            CudaDeviceGuard guard(gpu_device_);
            CUDA_CHECK(cudaMemcpyAsync(
                host_pre_norm_hidden.data(),
                engine_bufs_.d_pre_norm_hidden,
                host_pre_norm_hidden.size() * sizeof(float),
                cudaMemcpyDeviceToHost,
                compute_stream_));
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        }

        speculative_engine_->update_verify_h(n_batch, host_pre_norm_hidden.data());

        copy_llama_logits_rows(
            *tp_runtime_->rank_ctx(0).llama_runtime,
            verify_sample_plan, engine_bufs_.d_logits, vocab_size,
            gpu_device_, compute_stream_);

        auto verified = speculative_engine_->sample_and_accept(
            draft.draft_tokens,
            static_cast<const float*>(engine_bufs_.d_logits),
            n_logits_rows, vocab_size,
            req->sampling_params, req->prompt_token_ids, req->output_token_ids,
            engine_bufs_.d_verify_staging,
            engine_bufs_.d_output_ids, engine_bufs_.d_output_logprobs,
            engine_bufs_.d_prompt_mask, engine_bufs_.d_output_bin_counts,
            engine_bufs_.d_rep_penalties, engine_bufs_.d_freq_penalties,
            engine_bufs_.d_pres_penalties, compute_stream_);

        speculative_engine_->accept(0, verified.n_draft_accepted, compute_stream_);

        const int num_committed = static_cast<int>(verified.accepted_tokens.size());
        kv_cache_mgr_->trim_request_to_tokens(req->request_id, n_past + num_committed);
        tp_runtime_->llama_main_seq_trim(0, n_past + num_committed);
        speculative_engine_->trim_draft_kv_after_accept(
            n_past, num_committed, tp_runtime_.get());

        mtp_tokens_by_req[req_ri] = verified.accepted_tokens;
        mtp_logprobs_by_req[req_ri] = verified.accepted_logprobs;
        mtp_kv_committed_by_req[req_ri] = num_committed;

        spdlog::info(
            "MTP verify: request={} accepted {}/{} draft, output_tokens={}{}",
            req->request_id,
            verified.n_draft_accepted, n_draft, num_committed,
            verified.all_drafts_accepted
                ? (verified.has_bonus ? " (+bonus)" : "") : " (rejected)");
    } else {
        sample_plan = build_logits_sample_plan(
            all_reqs, sched_output.num_scheduled_tokens);
        validate_logits_sample_plan(
            sample_plan, total_tokens, config_.server.max_num_seqs);

        {
            std::vector<int32_t> batch_positions(static_cast<std::size_t>(total_tokens));
            for (int i = 0; i < total_tokens; ++i) {
                batch_positions[static_cast<std::size_t>(i)] =
                    static_cast<int32_t>(host_position_ids[static_cast<std::size_t>(i)]);
            }
            const auto batch_seq_ids = build_batch_seq_ids(
                total_tokens, host_cu_seqlens.data(), all_reqs);
            std::vector<int8_t> batch_logits_flags(static_cast<std::size_t>(total_tokens), 0);
            for (const auto& slot : sample_plan.slots) {
                batch_logits_flags[static_cast<std::size_t>(slot.hidden_batch_index)] = 1;
            }

            tp_runtime_->forward_llama(
                host_input_ids, batch_positions, batch_seq_ids, batch_logits_flags,
                d_slot_mapping, d_block_tables, d_seq_lens,
                d_token_to_seq, d_token_positions,
                total_tokens, num_reqs, max_num_blocks, block_size,
                attn_meta.is_prefill, num_prefill_tokens, num_decode_tokens,
                compute_stream_);

            if (mtp_enabled && engine_bufs_.d_pre_norm_hidden) {
                copy_llama_pre_norm_rows(
                    *tp_runtime_->rank_ctx(0).llama_runtime,
                    total_tokens, hidden_size, engine_bufs_.d_pre_norm_hidden,
                    gpu_device_, compute_stream_);
            }
        }

        if (mtp_enabled && engine_bufs_.d_pre_norm_hidden) {
            // Copy pre-norm hidden states from device to host for MTP pending_h
            // (对齐 llama.cpp speculative.cpp:632-633 pending_h = h_tgt[n_tokens-1])
            std::vector<float> host_pre_norm_hidden(
                static_cast<size_t>(total_tokens) * static_cast<size_t>(hidden_size));
            {
                CudaDeviceGuard guard(gpu_device_);
                CUDA_CHECK(cudaMemcpyAsync(
                    host_pre_norm_hidden.data(),
                    engine_bufs_.d_pre_norm_hidden,
                    host_pre_norm_hidden.size() * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    compute_stream_));
                CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
            }

            int tok_off = 0;
            for (int ri = 0; ri < num_reqs; ++ri) {
                auto sched_it = sched_output.num_scheduled_tokens.find(all_reqs[ri]->request_id);
                int num_sched = (sched_it != sched_output.num_scheduled_tokens.end()) ? sched_it->second : 0;
                if (num_sched <= 0) continue;
                // Prefill 阶段只更新 pending_h (n_tokens 个 token 的最后一行)
                speculative_engine_->update_pending_h(
                    num_sched,
                    host_pre_norm_hidden.data() + static_cast<size_t>(tok_off) * hidden_size);
                tok_off += num_sched;
            }
            CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
        }

        num_to_sample = sample_plan.num_logits();
    }

    void* d_logits = nullptr;
    if (!run_mtp_verify && num_to_sample > 0) {
        d_logits = engine_bufs_.d_logits;
        copy_llama_logits_rows(
            *tp_runtime_->rank_ctx(0).llama_runtime,
            sample_plan, d_logits, vocab_size, gpu_device_, compute_stream_);
        CHECK_KERNEL_LAUNCH();
    }

    if (!run_mtp_verify && num_to_sample > 0 && d_logits) {
        bool any_penalty = false;
        for (const auto& slot : sample_plan.slots) {
            auto& req = all_reqs[slot.req_index];
            if (req->sampling_params.repetition_penalty != 1.0f ||
                req->sampling_params.frequency_penalty != 0.0f ||
                req->sampling_params.presence_penalty != 0.0f) {
                any_penalty = true;
                break;
            }
        }

        const int ns = num_to_sample;
        if (any_penalty) {
            std::vector<uint8_t> host_prompt_mask(static_cast<std::size_t>(ns) * vocab_size, 0);
            std::vector<int32_t> host_output_counts(static_cast<std::size_t>(ns) * vocab_size, 0);
            std::vector<float> host_rep_p(ns, 1.0f), host_freq_p(ns, 0.0f), host_pres_p(ns, 0.0f);

            for (int si = 0; si < ns; ++si) {
                auto& req = all_reqs[sample_plan.slots[static_cast<std::size_t>(si)].req_index];
                host_rep_p[static_cast<std::size_t>(si)] = req->sampling_params.repetition_penalty;
                host_freq_p[static_cast<std::size_t>(si)] = req->sampling_params.frequency_penalty;
                host_pres_p[static_cast<std::size_t>(si)] = req->sampling_params.presence_penalty;

                const int64_t row_off = static_cast<int64_t>(si) * vocab_size;
                for (auto tid : req->prompt_token_ids) {
                    if (tid >= 0 && tid < vocab_size) {
                        host_prompt_mask[static_cast<std::size_t>(row_off + tid)] = 1;
                    }
                }
                for (auto tid : req->output_token_ids) {
                    if (tid >= 0 && tid < vocab_size) {
                        host_output_counts[static_cast<std::size_t>(row_off + tid)]++;
                    }
                }
            }

            CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_prompt_mask, host_prompt_mask.data(),
                                       static_cast<std::size_t>(ns) * vocab_size * sizeof(uint8_t),
                                       cudaMemcpyHostToDevice, compute_stream_));
            CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_output_bin_counts, host_output_counts.data(),
                                       static_cast<std::size_t>(ns) * vocab_size * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, compute_stream_));
            CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_rep_penalties, host_rep_p.data(),
                                       ns * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));
            CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_freq_penalties, host_freq_p.data(),
                                       ns * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));
            CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_pres_penalties, host_pres_p.data(),
                                       ns * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));
            apply_penalties(d_logits, engine_bufs_.d_prompt_mask, engine_bufs_.d_output_bin_counts,
                            engine_bufs_.d_rep_penalties, engine_bufs_.d_freq_penalties,
                            engine_bufs_.d_pres_penalties,
                            ns, vocab_size, ScalarType::FLOAT32, compute_stream_);
        }

        host_output_ids.resize(static_cast<std::size_t>(ns));
        host_output_logprobs.resize(static_cast<std::size_t>(ns));

        std::vector<float> host_temps(static_cast<std::size_t>(ns));
        std::vector<int32_t> host_topk(static_cast<std::size_t>(ns));
        std::vector<float> host_topp(static_cast<std::size_t>(ns));
        std::vector<float> host_minp(static_cast<std::size_t>(ns));
        for (int si = 0; si < ns; ++si) {
            auto& req = all_reqs[sample_plan.slots[static_cast<std::size_t>(si)].req_index];
            host_temps[static_cast<std::size_t>(si)] = req->sampling_params.temperature;
            host_topk[static_cast<std::size_t>(si)] = req->sampling_params.top_k;
            host_topp[static_cast<std::size_t>(si)] = req->sampling_params.top_p;
            host_minp[static_cast<std::size_t>(si)] = req->sampling_params.min_p;
        }
        CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_temperatures, host_temps.data(),
                                   ns * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_top_k, host_topk.data(),
                                   ns * sizeof(int32_t), cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_top_p, host_topp.data(),
                                   ns * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(engine_bufs_.d_min_p, host_minp.data(),
                                   ns * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));

        gpu_filtered_sampling(d_logits, nullptr,
                              static_cast<int32_t*>(engine_bufs_.d_output_ids),
                              static_cast<float*>(engine_bufs_.d_output_logprobs),
                              static_cast<float*>(engine_bufs_.d_temperatures),
                              static_cast<int32_t*>(engine_bufs_.d_top_k),
                              static_cast<float*>(engine_bufs_.d_top_p),
                              static_cast<float*>(engine_bufs_.d_min_p),
                              ns, vocab_size, ScalarType::FLOAT32, 0, compute_stream_);

        CUDA_CHECK(cudaMemcpyAsync(host_output_ids.data(), engine_bufs_.d_output_ids,
                                   static_cast<std::size_t>(ns) * sizeof(int32_t),
                                   cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(host_output_logprobs.data(), engine_bufs_.d_output_logprobs,
                                   static_cast<std::size_t>(ns) * sizeof(float),
                                   cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    }

    std::vector<RequestOutput> outputs;
    if (run_mtp_verify || (num_to_sample > 0 && d_logits)) {
        int out_idx = 0;
        for (int ri = 0; ri < num_reqs; ++ri) {
            auto& req = all_reqs[ri];
            auto it = sched_output.num_scheduled_tokens.find(req->request_id);
            int num_sched = (it != sched_output.num_scheduled_tokens.end()) ? it->second : 1;
            bool prefill_done = !req->is_prefill()
                || (req->num_computed_tokens + num_sched >= req->num_prompt_tokens());

            RequestOutput out;
            out.request_id = req->request_id;
            out.seq_id_ = req->seq_id_;
            out.finished = false;
            out.prefill_complete = prefill_done;
            out.num_new_tokens = num_sched;

            if (prefill_done) {
                auto mtp_it = mtp_tokens_by_req.find(ri);
                if (mtp_it != mtp_tokens_by_req.end()) {
                    out.output_token_ids = mtp_it->second;
                    out.output_logprobs = mtp_logprobs_by_req[ri];
                    const int num_output_new = static_cast<int>(mtp_it->second.size());
                    auto kv_it = mtp_kv_committed_by_req.find(ri);
                    out.num_new_tokens = (kv_it != mtp_kv_committed_by_req.end())
                        ? kv_it->second : num_sched;
                    out_idx++;

                    bool is_eos = false;
                    bool is_stop = false;
                    for (int32_t sampled_id : out.output_token_ids) {
                        if (std::find(config_.model.eos_token_ids.begin(),
                                      config_.model.eos_token_ids.end(),
                                      sampled_id) != config_.model.eos_token_ids.end()) {
                            is_eos = true;
                            break;
                        }
                        if (std::find(req->sampling_params.stop_token_ids.begin(),
                                      req->sampling_params.stop_token_ids.end(),
                                      sampled_id) != req->sampling_params.stop_token_ids.end()) {
                            is_stop = true;
                            break;
                        }
                    }
                    bool hit_max = req->num_output_tokens + num_output_new
                        >= req->sampling_params.max_tokens;
                    bool below_min_tokens = req->num_output_tokens + num_output_new
                        <= req->sampling_params.min_tokens;
                    if (is_eos && req->sampling_params.ignore_eos) is_eos = false;
                    if (is_eos && below_min_tokens) is_eos = false;
                    if (is_stop && below_min_tokens) is_stop = false;
                    if (is_eos || is_stop || hit_max) {
                        out.finished = true;
                        out.finish_reason = (is_eos || is_stop) ? FinishReason::STOP : FinishReason::LENGTH;
                    }
                    if (req->stream_callback) {
                        req->stream_callback(out.output_token_ids, out.finished, out.finish_reason);
                    }
                    outputs.push_back(std::move(out));
                    continue;
                }

                int32_t sampled_id = host_output_ids[static_cast<std::size_t>(out_idx)];
                float logprob = host_output_logprobs[static_cast<std::size_t>(out_idx)];

                out_idx++;

                out.output_token_ids.push_back(sampled_id);
                out.output_logprobs.push_back(logprob);

                bool is_eos = std::find(config_.model.eos_token_ids.begin(),
                                        config_.model.eos_token_ids.end(),
                                        sampled_id) != config_.model.eos_token_ids.end();
                bool is_stop = std::find(req->sampling_params.stop_token_ids.begin(),
                                         req->sampling_params.stop_token_ids.end(),
                                         sampled_id) != req->sampling_params.stop_token_ids.end();
                bool hit_max = req->num_output_tokens + 1 >= req->sampling_params.max_tokens;
                bool below_min_tokens = req->num_output_tokens + 1 <= req->sampling_params.min_tokens;
                if (is_eos && req->sampling_params.ignore_eos) is_eos = false;
                if (is_eos && below_min_tokens) is_eos = false;
                if (is_stop && below_min_tokens) is_stop = false;
                if (is_eos || is_stop || hit_max) {
                    out.finished = true;
                    out.finish_reason = (is_eos || is_stop) ? FinishReason::STOP : FinishReason::LENGTH;
                }
                if (req->stream_callback) {
                    req->stream_callback(out.output_token_ids, out.finished, out.finish_reason);
                }
            }

            outputs.push_back(std::move(out));
        }
    } else {
        for (int ri = 0; ri < num_reqs; ++ri) {
            auto& req = all_reqs[ri];
            auto it = sched_output.num_scheduled_tokens.find(req->request_id);
            int num_sched = (it != sched_output.num_scheduled_tokens.end()) ? it->second : 1;
            bool prefill_done = !req->is_prefill()
                || (req->num_computed_tokens + num_sched >= req->num_prompt_tokens());

            RequestOutput out;
            out.request_id = req->request_id;
            out.seq_id_ = req->seq_id_;
            out.finished = false;
            out.prefill_complete = prefill_done;
            out.num_new_tokens = num_sched;
            outputs.push_back(std::move(out));
        }
    }

    const auto fwd_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_fwd_start).count();
    const char* phase = run_mtp_verify ? "mtp_verify"
        : (num_prefills > 0 ? (num_decodes > 0 ? "mixed" : "prefill") : "decode");
    spdlog::info(
        "[Forward] phase={} tokens={} reqs={} prefill_tok={} decode_tok={} {:.1f}ms",
        phase, total_tokens, num_reqs, num_prefill_tokens, num_decode_tokens,
        static_cast<double>(fwd_ms));

    return outputs;
}

const struct llama_vocab* Engine::vocab() const {
    if (!tp_runtime_) return nullptr;
    llama_model* m = tp_runtime_->llama_primary().model();
    if (!m) return nullptr;
    return llama_model_get_vocab(m);
}

}