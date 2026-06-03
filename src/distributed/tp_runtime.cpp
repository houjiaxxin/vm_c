// TPRuntime — column-TP 多 GPU 并行推理运行时
//
// 设计要点：
//   1. NcclComm 管理 allreduce（NCCL + internal CUDA kernel fallback）
//   2. 列式切分权重（通过 cparams.vm_c_tp / vm_c_tp_size 控制）
//   3. 图构建中插入 GGML_OP_VM_C_TP_ALLREDUCE 节点
//   4. 持久化 worker 线程：所有 rank 并行执行 decode
//
// 与旧版 CustomAR 的差异：
//   - 不再使用 custom_ar.cpp / TensorParallel（已删除）
//   - allreduce 回调由 NcclComm 在 ggml-cuda 后端注册
//   - 外部调用方接口保持不变（TPRuntime 类名和方法签名兼容）

#include "vm_c/distributed/tp_runtime.hpp"

#include <thread>

#include "vm_c/cuda/gpu_arch.hpp"
#include "vm_c/cuda/vm_c_tensor.h"
#include "vm_c/memory/kv_cache_manager.hpp"
#include "vm_c/official/ggml_backend_pool.hpp"
#include "vm_c/official/llama_runtime.hpp"
#include "vm_c/official/turboquant_kv_bridge.hpp"

#include "llama.h"

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace vm_c {

namespace {

void sync_tp_backends(TPRankContext* ranks, int tp_size) {
    int orig_device = 0;
    CUDA_CHECK(cudaGetDevice(&orig_device));
    for (int i = 0; i < tp_size; ++i) {
        CudaDeviceGuard guard(ranks[i].gpu_device);
        void* backend = official::GgmlBackendPool::instance().backend_for_device(ranks[i].gpu_device);
        ggml_backend_synchronize(static_cast<ggml_backend_t>(backend));
    }
    set_device_safe(orig_device);
}

}  // namespace

TPRuntime::TPRuntime(int tp_size, int base_gpu_device)
    : tp_size_(tp_size), base_gpu_device_(base_gpu_device) {
    ranks_.resize(static_cast<size_t>(tp_size));
    for (int i = 0; i < tp_size_; ++i) {
        ranks_[static_cast<size_t>(i)].gpu_device = base_gpu_device + i;
        ranks_[static_cast<size_t>(i)].rank = i;
    }
}

TPRuntime::~TPRuntime() {
    shutdown();
}

void TPRuntime::shutdown() {
    stop_worker_threads();
    for (int i = 0; i < tp_size_; ++i) {
        auto& ctx = ranks_[static_cast<size_t>(i)];
        CudaDeviceGuard guard(ctx.gpu_device);
        ctx.llama_runtime.reset();
        ctx.tq_bridge.reset();
        ctx.kv_cache_mgr.reset();
        for (auto* buf : ctx.uva_buffers) {
            if (buf) {
                CUDA_CHECK(cudaFreeHost(buf));
            }
        }
        ctx.uva_buffers.clear();
    }
    comm_.reset();
    initialized_ = false;
}

void TPRuntime::sync_all_ranks() {
    sync_tp_backends(ranks_.data(), tp_size_);
}

// --- 持久化 worker 线程 ----------------------------------------------------------

void TPRuntime::start_worker_threads() {
    workers_.reserve(static_cast<size_t>(tp_size_));
    for (int i = 0; i < tp_size_; ++i) {
        auto w = std::make_unique<Worker>();
        w->thread = std::thread([this, i, pw = w.get()]() {
            CudaDeviceGuard guard(ranks_[static_cast<size_t>(i)].gpu_device);
            while (true) {
                std::function<void()> local_work;
                {
                    std::unique_lock<std::mutex> lk(pw->mtx);
                    pw->cv.wait(lk, [pw] { return pw->work_ready || pw->stop; });
                    if (pw->stop) return;
                    local_work = std::move(pw->work);
                    pw->work_ready = false;
                }
                local_work();
                {
                    std::lock_guard<std::mutex> lk(pw->mtx);
                    pw->work_done = true;
                }
                pw->cv.notify_one();
            }
        });
        workers_.push_back(std::move(w));
    }
}

void TPRuntime::stop_worker_threads() {
    for (auto& wp : workers_) {
        {
            std::lock_guard<std::mutex> lk(wp->mtx);
            wp->stop = true;
        }
        wp->cv.notify_one();
    }
    for (auto& wp : workers_) {
        if (wp->thread.joinable()) wp->thread.join();
    }
    workers_.clear();
}

void TPRuntime::run_on_workers(std::function<void(int, TPRankContext&)> work_fn) {
    for (int i = 0; i < tp_size_; ++i) {
        auto& w = *workers_[static_cast<size_t>(i)];
        std::lock_guard<std::mutex> lk(w.mtx);
        w.work = [i, work_fn, &ctx = ranks_[static_cast<size_t>(i)]]() {
            work_fn(i, const_cast<TPRankContext&>(ctx));
        };
        w.work_done  = false;
        w.work_ready = true;
    }
    for (auto& wp : workers_) {
        wp->cv.notify_one();
    }
    for (auto& wp : workers_) {
        std::unique_lock<std::mutex> lk(wp->mtx);
        wp->cv.wait(lk, [&wp] { return wp->work_done; });
    }
}

// --- forward 方法 ----------------------------------------------------------------

void TPRuntime::initialize(const VmCConfig& config) {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (tp_size_ > device_count) {
        throw std::runtime_error("[TPRuntime] tp_size exceeds CUDA device count");
    }
    if (base_gpu_device_ + tp_size_ > device_count) {
        throw std::runtime_error("[TPRuntime] GPU range exceeds device count");
    }

    const int tp = tp_size_;
    const auto& m = config.model;
    auto require_divisible = [tp](int value, const char* name) {
        if (tp > 1 && value > 0 && value % tp != 0) {
            throw std::runtime_error(
                std::string("[TPRuntime] ") + name + " must be divisible by tp_size");
        }
    };
    require_divisible(m.num_attention_heads, "num_attention_heads");
    require_divisible(m.hidden_size, "hidden_size");
    if (m.num_key_value_heads >= tp) {
        require_divisible(m.num_key_value_heads, "num_key_value_heads");
    }
    if (m.num_experts > 0) {
        require_divisible(m.num_experts, "num_experts");
    }
    if (m.moe_intermediate_size > 0) {
        require_divisible(m.moe_intermediate_size, "moe_intermediate_size");
    }
    if (m.intermediate_size > 0) {
        require_divisible(m.intermediate_size, "intermediate_size");
    }

    // 创建 NcclComm（替代旧的 TensorParallel）
    std::vector<int> devices(static_cast<size_t>(tp_size_));
    for (int i = 0; i < tp_size_; ++i) {
        devices[static_cast<size_t>(i)] = base_gpu_device_ + i;
    }
    comm_ = std::make_unique<vm_c::tp::NcclComm>();
    if (!comm_->init(devices)) {
        throw std::runtime_error("[TPRuntime] NcclComm init failed");
    }

    // 创建每个 rank 的辅助 CUDA stream（主线程使用，不依赖 ggml backend）
    // 注意：不在此处调用 backend_for_device，避免 ggml backend 在主线程上初始化。
    // ggml backend 的初始化将延迟到 worker 线程（在 initialize_llama_graph 中），
    // 确保 backend 内部的 CUDA stream 创建在 worker 线程的 primary context 中。
    for (int i = 0; i < tp_size_; ++i) {
        auto& ctx = ranks_[static_cast<size_t>(i)];
        CudaDeviceGuard guard(ctx.gpu_device);
        CUDA_CHECK(cudaStreamCreate(&ctx.stream));
        ctx.stream_owned = true;
    }

    // P2P enable
    for (int i = 0; i < tp_size_; ++i) {
        for (int j = 0; j < tp_size_; ++j) {
            if (i == j) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(
                &can_access, ranks_[static_cast<size_t>(i)].gpu_device,
                ranks_[static_cast<size_t>(j)].gpu_device));
            if (!can_access) {
                throw std::runtime_error("[TPRuntime] P2P not supported between TP ranks");
            }
            CUDA_CHECK(cudaSetDevice(ranks_[static_cast<size_t>(i)].gpu_device));
            cudaError_t err = cudaDeviceEnablePeerAccess(
                ranks_[static_cast<size_t>(j)].gpu_device, 0);
            if (err == cudaErrorPeerAccessAlreadyEnabled) {
                spdlog::debug("[TPRuntime] P2P already enabled: rank {} -> rank {}, clearing error state", i, j);
                cudaGetLastError();
            } else if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("[TPRuntime] P2P enable failed: ") + cudaGetErrorString(err));
            }
        }
    }

    initialized_ = true;
    spdlog::info("[TPRuntime] initialized {} ranks (NcclComm, libllama forward path)", tp_size_);
    start_worker_threads();
    spdlog::info("[TPRuntime] {} persistent worker threads started", tp_size_);
}

void TPRuntime::profile_gpu_memory(const VmCConfig& config) {
    for (int i = 0; i < tp_size_; ++i) {
        auto& ctx = ranks_[static_cast<size_t>(i)];
        CudaDeviceGuard guard(ctx.gpu_device);
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        spdlog::info("[TPRuntime] Rank {} GPU {}: {:.1f} GB free / {:.1f} GB total",
            i, ctx.gpu_device,
            static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    (void) config;
}

void TPRuntime::initialize_kv_cache(const VmCConfig& config) {
    for (int i = 0; i < tp_size_; ++i) {
        auto& ctx = ranks_[static_cast<size_t>(i)];
        CudaDeviceGuard guard(ctx.gpu_device);
        const int num_kv_heads = std::max(1, config.model.num_key_value_heads / tp_size_);
        // [BUG-FIX] 必须使用 cfg.head_dim（来自 GGUF attention.key_length），
        // 不能再用 hidden_size/num_attention_heads 反推。
        // Qwen3.5-A3B: hidden=2048, n_head=16 → 128（错），实际 head_dim=256。
        // 这导致 TurboQuant slot_size 计算不一致（cache_mgr 134 vs bridge 262），
        // MHA 从 KV cache 读错位数据 → 乱码。
        const int head_dim = config.model.head_dim > 0
            ? config.model.head_dim
            : (config.model.num_attention_heads > 0
                ? config.model.hidden_size / config.model.num_attention_heads
                : 128);
        spdlog::info("[TP-KV-CACHE-INIT] rank={} num_kv_heads={} head_dim={} (config.head_dim={}, hidden={}, n_head={})",
                     i, num_kv_heads, head_dim,
                     config.model.head_dim, config.model.hidden_size, config.model.num_attention_heads);
        ctx.kv_cache_mgr = std::make_shared<KVCacheManager>(
            config.cache, config.model,
            config.model.num_hidden_layers,
            num_kv_heads, head_dim,
            ctx.gpu_device, false);

        const int64_t total_blocks = (static_cast<int64_t>(config.model.max_model_len) *
                                      std::max(1, config.server.max_num_seqs) +
                                      config.cache.block_size - 1) /
                                     config.cache.block_size;
        for (int g = 0; g < ctx.kv_cache_mgr->num_kv_cache_groups(); ++g) {
            ctx.kv_cache_mgr->allocate_gpu_blocks_for_group(g, total_blocks);
        }

        const int num_groups = ctx.kv_cache_mgr->num_kv_cache_groups();
        spdlog::info("[TPRuntime] Rank {} KV cache allocated: {} groups, {} blocks/group",
                     i, num_groups, total_blocks);
    }
}

void TPRuntime::initialize_llama_graph(const VmCConfig& config) {
    official::LlamaRuntime::ensure_backend();

    const int num_layers = config.model.num_hidden_layers;
    const bool use_tq = config.cache.kv_cache_quant_method.find("turboquant") != std::string::npos;

    // 在 worker 线程上初始化每个 rank 的 LlamaRuntime，而非 main 线程。
    // 原因：ggml backend 内部会创建 CUDA stream，stream 属于创建时所在线程的
    // primary context。如果在 main 线程创建 backend，后续 worker 线程调用
    // decode() 时使用同一个 stream 会触发 "CUDA Stream does not belong to
    // the expected context" 错误，且可能导致 GPU 计算数据损坏。
    //
    // Worker 线程有自己的 CudaDeviceGuard，backend 在对应 device 上初始化后，
    // stream 正确属于 worker 线程的 context，decode 不再报错。
    //
    // 注意：模型权重（GPU device memory）是跨线程共享的，不受线程切换影响。
    std::exception_ptr init_exception;
    std::mutex init_mutex;

    auto work = [&](int i, TPRankContext& ctx) {
        try {
            ctx.tq_bridge = std::make_unique<official::TurboQuantKvBridge>();
            if (use_tq) {
                ctx.tq_bridge->initialize(
                    config, ctx.kv_cache_mgr.get(), ctx.gpu_device, num_layers, tp_size_);
            }

            ctx.llama_runtime = std::make_unique<official::LlamaRuntime>();
            ctx.llama_runtime->set_turboquant_bridge(use_tq ? ctx.tq_bridge.get() : nullptr);
            ctx.llama_runtime->set_tensor_parallel((void*)comm_.get(), tp_size_);
            ctx.llama_runtime->load(
                config, ctx.gpu_device, std::move(ctx.llama_weight_snapshot));

            if (ctx.llama_runtime->context_main()) {
                llama_set_tp_context(
                    ctx.llama_runtime->context_main(),
                    (void*)comm_.get(), tp_size_);
                spdlog::info("[TPRuntime] Rank {} llama_set_tp_context(main) tp_size={}",
                             i, tp_size_);
            }
            // [vm_c 诊断] MTP ctx 也必须 set_tp_context，否则 MTP graph 的 add_reduce 全是 no-op
            if (ctx.llama_runtime->context_mtp()) {
                llama_set_tp_context(
                    ctx.llama_runtime->context_mtp(),
                    (void*)comm_.get(), tp_size_);
                spdlog::info("[TPRuntime] Rank {} llama_set_tp_context(mtp) tp_size={}",
                             i, tp_size_);
            } else {
                spdlog::warn("[TPRuntime] Rank {} has no MTP context; MTP graph would fall back "
                             "to tp_size=1 (add_reduce no-op)", i);
            }

            spdlog::info("[TPRuntime] Rank {} libllama ready (NcclComm allreduce)", i);
        } catch (...) {
            std::lock_guard<std::mutex> lk(init_mutex);
            init_exception = std::current_exception();
        }
    };

    run_on_workers(work);

    if (init_exception) {
        std::rethrow_exception(init_exception);
    }
}

void TPRuntime::verify_gpu_memory_after_llama(const VmCConfig& config) {
    for (int i = 0; i < tp_size_; ++i) {
        auto& ctx = ranks_[static_cast<size_t>(i)];
        CudaDeviceGuard guard(ctx.gpu_device);
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        spdlog::info("[TPRuntime] Rank {} GPU {} after model load: {:.1f} GB free / {:.1f} GB total",
            i, ctx.gpu_device,
            static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0),
            static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    (void) config;
}

void TPRuntime::llama_main_seq_trim(int32_t seq_id, int64_t n_keep) {
    for (int i = 0; i < tp_size_; ++i) {
        CudaDeviceGuard guard(ranks_[static_cast<size_t>(i)].gpu_device);
        ranks_[static_cast<size_t>(i)].llama_runtime->memory_seq_rm(
            seq_id, static_cast<int32_t>(n_keep), -1);
    }
}

void TPRuntime::forward_llama(
    const std::vector<int32_t>& tokens,
    const std::vector<int32_t>& positions,
    const std::vector<int32_t>& seq_ids,
    const std::vector<int8_t>& logits_flags,
    const int64_t* d_slot_mapping,
    const int32_t* d_block_tables,
    const int32_t* d_seq_lens,
    const int32_t* d_token_to_seq,
    const int32_t* d_token_positions,
    int num_tokens,
    int num_reqs,
    int max_num_blocks_per_req,
    int64_t block_size,
    bool is_prefill,
    int num_prefill_tokens,
    int num_decode_tokens,
    cudaStream_t primary_stream) {
    (void) primary_stream;
    if (num_tokens <= 0) return;

    // 并行执行各 rank 的 decode
    run_on_workers([&](int i, TPRankContext& ctx) {
        (void) i;
        if (!ctx.llama_runtime || !ctx.llama_runtime->loaded()) {
            throw std::runtime_error("[TPRuntime] LlamaRuntime not loaded");
        }
        llama_batch local_batch = official::LlamaBatchBuilder::build(
            tokens, positions, seq_ids, logits_flags);
        if (ctx.tq_bridge && ctx.kv_cache_mgr && ctx.kv_cache_mgr->is_turboquant()) {
            ctx.tq_bridge->set_batch_metadata(
                d_slot_mapping, d_block_tables, d_seq_lens,
                num_tokens, num_reqs, max_num_blocks_per_req, block_size,
                is_prefill, num_prefill_tokens, num_decode_tokens,
                d_token_to_seq, d_token_positions,
                ctx.stream);
        }
        ctx.llama_runtime->decode(local_batch, /*mtp=*/false);
        official::LlamaBatchBuilder::free(local_batch);
    });

    sync_all_ranks();
}

void TPRuntime::forward_llama_mtp_draft(
    const float* h_embd_row,
    int32_t token,
    int64_t position,
    float* output_pre_norm,
    float* output_logits,
    cudaStream_t primary_stream) {
    (void) primary_stream;
    if (!h_embd_row || !output_pre_norm || !output_logits) {
        throw std::runtime_error("[TPRuntime] forward_llama_mtp_draft: null pointer");
    }
    spdlog::info("[MTP-TP-DRAFT-IN] token={} pos={} h_embd_row first8=[{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}]",
        token, (long long)position,
        h_embd_row[0], h_embd_row[1], h_embd_row[2], h_embd_row[3],
        h_embd_row[4], h_embd_row[5], h_embd_row[6], h_embd_row[7]);

    const int n_embd = ranks_[0].llama_runtime->n_embd();
    const int n_vocab = ranks_[0].llama_runtime->n_vocab();

    // h_embd_row is already host memory, no need to copy
    // 直接使用 host pointer 构建 batch
    run_on_workers([&](int i, TPRankContext& ctx) {
        (void) i;
        if (!ctx.llama_runtime || !ctx.llama_runtime->loaded()) {
            throw std::runtime_error("[TPRuntime] LlamaRuntime not loaded");
        }

        llama_batch batch = official::LlamaBatchBuilder::build_mtp_draft(
            token, static_cast<int32_t>(position),
            h_embd_row, n_embd);

        ctx.llama_runtime->decode(batch, /*mtp=*/true);
        official::LlamaBatchBuilder::free(batch);
    });

    // Copy results from rank 0
    // 注意：MTP draft decode 运行在 ctx_mtp 上，logits 必须从 MTP 上下文读取，
    // 而非从 ctx_main。之前使用 context_main() + llama_get_logits 是致命 Bug，
    // 导致读取的是上一次主模型 forward 的过期 logits，draft 采样完全随机。
    {
        CudaDeviceGuard guard(ranks_[0].gpu_device);
        auto* ctx_mtp = ranks_[0].llama_runtime->context_mtp();
        float* logits = llama_get_logits(ctx_mtp);
        if (logits) {
            CUDA_CHECK(cudaMemcpyAsync(
                output_logits, logits,
                static_cast<size_t>(n_vocab) * sizeof(float),
                cudaMemcpyDeviceToDevice, primary_stream));
        }
        float* embd = ranks_[0].llama_runtime->embeddings_pre_norm_ith(0, /*mtp=*/true);
        if (embd) {
            CUDA_CHECK(cudaMemcpyAsync(
                output_pre_norm, embd,
                static_cast<size_t>(n_embd) * sizeof(float),
                cudaMemcpyDeviceToDevice, primary_stream));
        }
    }

    sync_all_ranks();
}

void TPRuntime::llama_mtp_seq_rollback(int32_t seq_id, int64_t n_past) {
    for (int i = 0; i < tp_size_; ++i) {
        CudaDeviceGuard guard(ranks_[static_cast<size_t>(i)].gpu_device);
        llama_memory_t mem = llama_get_memory(ranks_[static_cast<size_t>(i)].llama_runtime->context_mtp());
        if (!mem) {
            throw std::runtime_error("[TPRuntime] llama_mtp_seq_rollback: null memory");
        }
        if (!llama_memory_seq_rm(mem, seq_id, static_cast<int32_t>(n_past), -1)) {
            throw std::runtime_error("[TPRuntime] llama_mtp_seq_rollback: memory_seq_rm failed");
        }
    }
}

void TPRuntime::llama_mtp_seq_clear(int32_t seq_id) {
    for (int i = 0; i < tp_size_; ++i) {
        CudaDeviceGuard guard(ranks_[static_cast<size_t>(i)].gpu_device);
        llama_memory_t mem = llama_get_memory(ranks_[static_cast<size_t>(i)].llama_runtime->context_mtp());
        if (!mem) {
            throw std::runtime_error("[TPRuntime] llama_mtp_seq_clear: null memory");
        }
        const llama_pos pos_max_before = llama_memory_seq_pos_max(mem, seq_id);
        if (!llama_memory_seq_rm(mem, seq_id, 0, -1)) {
            throw std::runtime_error("[TPRuntime] llama_mtp_seq_clear: memory_seq_rm failed");
        }
        spdlog::info("[TPRuntime] llama_mtp_seq_clear: rank={} seq_id={} pos_max_before={}", i, seq_id, (int)pos_max_before);
    }
}

}  // namespace vm_c
