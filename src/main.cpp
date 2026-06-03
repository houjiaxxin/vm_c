#include "vm_c/core/engine.hpp"
#include "vm_c/server/http_server.hpp"
#include "vm_c/tokenizer/tokenizer.hpp"
#include "vm_c/tokenizer/incremental_detokenizer.hpp"
#include "vm_c/core/config.hpp"
#include "vm_c/core/gguf_reader.hpp"
#include "vm_c/memory/expert_cache.hpp"
#include "vm_c/cuda/gpu_arch.hpp"

#include "llama.h"

// ── llama_vocab → Tokenizer 适配器，完全对齐官方 tokenization ──
// GgufTokenizer 的 encode 无法正确处理特殊 token 和字节编码，
// 直接用 llama_vocab 的 tokenize(detokenize) 来替代。
class VocabTokenizer : public vm_c::Tokenizer {
public:
    VocabTokenizer(const struct llama_vocab* vocab, std::string chat_template)
        : vocab_(vocab) {
        set_chat_template(std::move(chat_template));
    }

    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = true) const override {
        // 先查所需缓冲区大小：返回负数时，绝对值=所需 token 数
        int n_tokens = llama_tokenize(vocab_, text.data(), (int)text.size(), nullptr, 0,
                                      add_special_tokens, true);
        if (n_tokens == 0) return {};
        if (n_tokens < 0) n_tokens = -n_tokens;
        std::vector<llama_token> buf((size_t)n_tokens);
        int n_written = llama_tokenize(vocab_, text.data(), (int)text.size(), buf.data(), (int)buf.size(),
                                       add_special_tokens, true);
        if (n_written < 0) return {};
        buf.resize((size_t)n_written);
        return std::vector<int32_t>(buf.begin(), buf.end());
    }

    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const override {
        return decode(ids, skip_special, false);
    }

    std::string decode(const std::vector<int32_t>& ids, bool skip_special,
                       bool) const override {
        // 严格对齐官方 common_detokenize（common/common.cpp:1704-1717）
        // - remove_special=false（官方固定传 false，从不依赖此参数去 EOS）
        // - unparse_special=!skip_special
        //   false 时 llama_token_to_piece 内部跳过 CONTROL/UNKNOWN（含 EOS）
        //   true 时渲染全部 token
        if (ids.empty()) return {};
        std::vector<llama_token> tokens(ids.begin(), ids.end());
        std::string result;
        result.resize(std::max(result.capacity(), tokens.size()));
        int n_chars = llama_detokenize(vocab_, tokens.data(), (int32_t)tokens.size(),
                                       &result[0], (int32_t)result.size(),
                                       false, !skip_special);
        if (n_chars < 0) {
            result.resize((size_t)(-n_chars));
            n_chars = llama_detokenize(vocab_, tokens.data(), (int32_t)tokens.size(),
                                       &result[0], (int32_t)result.size(),
                                       false, !skip_special);
        }
        if (n_chars < 0) return {};
        result.resize((size_t)n_chars);
        return result;
    }

    int32_t eos_token_id() const override { return llama_vocab_eos(vocab_); }
    int32_t pad_token_id() const override { return llama_vocab_pad(vocab_); }
    int32_t bos_token_id() const override { return llama_vocab_bos(vocab_); }
    bool has_bos() const override { return false; } // llama_tokenize 内部处理
    int vocab_size() const override { return llama_vocab_n_tokens(vocab_); }

private:
    const struct llama_vocab* vocab_;
};

#include <spdlog/spdlog.h>
#include <spdlog/cfg/env.h>
#include <nlohmann/json.hpp>
#include <cuda_runtime.h>

#include <unistd.h>
#include <sys/stat.h>
#include <iostream>
#include <string>
#include <csignal>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <sstream>
#include <algorithm>

static std::atomic<bool> g_running{true};
static std::condition_variable g_shutdown_cv;
static std::atomic<int> g_shutdown_signals{0};
static std::mutex g_step_mutex;
static std::condition_variable g_step_cv;
static std::atomic<bool> g_has_work{false};
static std::chrono::milliseconds g_poll_interval{500};
static std::chrono::milliseconds g_step_interval{10};

namespace {
const char* finish_reason_str(vm_c::FinishReason reason) {
    switch (reason) {
        case vm_c::FinishReason::STOP:   return "stop";
        case vm_c::FinishReason::LENGTH: return "length";
        case vm_c::FinishReason::ERROR:  return "error";
        case vm_c::FinishReason::ABORTED: return "abort";
    }
    return "stop";
}
}

static void signal_handler(int sig) {
    const int n = g_shutdown_signals.fetch_add(1, std::memory_order_relaxed) + 1;
    // WARNING: Do NOT call spdlog or any non-async-signal-safe function here!
    // spdlog::info() uses internal mutexes that can deadlock if SIGINT
    // arrives while a mutex is held. Use write() instead (async-signal-safe).
    g_running.store(false, std::memory_order_release);
    g_shutdown_cv.notify_all();
    if (n >= 2) {
        const char msg[] = "Second interrupt: exiting immediately\n";
        if (write(STDERR_FILENO, msg, sizeof(msg) - 1) < 0) {} // ignore
        std::_Exit(128 + sig);
    }
}

static vm_c::VmCConfig parse_args(int argc, char* argv[]) {
    vm_c::VmCConfig config;

    if (argc < 2) {
        std::cerr << "Usage: vm_c_server <model_path> [options]\n"
                  << "\nOptions:\n"
                  << "  --tokenizer <path>              Tokenizer path (default: same as model_path)\n"
                  << "  --kv-cache-dtype <dtype>        KV cache dtype (auto/float16/bfloat16/fp8_e4m3/fp8_e5m2/int8_per_token_head/fp8_per_token_head/nvfp4/turboquant_4bit_nc/turboquant_k8v4/turboquant_k3v4_nc/turboquant_3bit_nc)\n"
                  << "  --max-model-len <n>             Sequence cap: min(n, HF max_position_embeddings); omit uses HF\n"
                  << "  --gpu-memory-utilization <f>    GPU memory utilization (default: 0.90)\n"
                  << "  --gpu-safety-margin-mb <n>      GPU safety margin in MB (default: auto 2%% of total, 64-512)\n"
                  << "  --spec-method <method>          Speculative decoding: none/mtp (default: auto from GGUF)\n"
                  << "  --spec-draft-n-max <n>          Max MTP draft tokens per step (default: 5, llama.cpp: --spec-draft-n-max)\n"
                  << "  --spec-draft-n-min <n>          Min MTP draft tokens per step (default: 0, llama.cpp: --spec-draft-n-min)\n"
                  << "  --spec-draft-p-min <p>          Min draft token probability (default: 0, llama.cpp: --spec-draft-p-min)\n"
                  << "  --tensor-parallel-size <n>      Tensor parallel size (default: 1)\n"
                  << "  --gpu-devices <d1,d2,...>       GPU device IDs to use (default: 0,1,...,tp_size-1; overrides CUDA_VISIBLE_DEVICES)\n"
                  << "  --served-model-name <name>      Served model name\n"
                  << "  --host <addr>                   Host address (default: 0.0.0.0)\n"
                  << "  --port <port>                   Port number (default: 8000)\n"
                  << "  --system-fingerprint <fp>       System fingerprint string\n"
                  << "  --cpu-offload-gb <n>            CPU offload size in GB (default: 0)\n"
                  << "  --cpu-offload-params <params>   CPU offload params (can specify multiple times, e.g., --cpu-offload-params experts --cpu-offload-params weight)\n"
                  << "  --max-num-seqs <n>              Max number of sequences (default: "
                  << vm_c::ServerConfig::DEFAULT_MAX_NUM_SEQS << ")\n"
                  << "  --max-num-batched-tokens <n>    Max batched tokens per step (default: "
                  << vm_c::ServerConfig::DEFAULT_MAX_NUM_BATCHED_TOKENS << ")\n"
                  << "  --kv-offloading-size <n>        KV offloading size in GiB (default: 0)\n"
                  << "  --expert-gpu-cache <n>          GPU cache capacity for MoE experts (default: num_experts_per_tok)\n"
                  << "  --expert-cache-update-period <n> Token interval for cache update (default: "
                  << vm_c::ExpertCacheConfig::DEFAULT_UPDATE_PERIOD_TOKENS << ")\n"
                  << "  --expert-cache-history-window <n> Sliding window for access stats (default: 512)\n"
                  << "  --poll-interval <n>             Poll interval in ms (default: 500)\n"
                  << "  --thread-pool-size <n>          HTTP server thread pool size (default: 8)\n"
                  << "  --enable-prefix-caching         Enable prefix caching\n"
                  << "  --disable-prefix-caching        Disable prefix caching\n";
        std::exit(1);
    }

    config.model.model_dir = argv[1];
    config.server.served_model_name = argv[1];

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--tokenizer" && i + 1 < argc) {
            config.model.tokenizer_dir = argv[++i];
        } else if (arg == "--kv-cache-dtype" && i + 1 < argc) {
            std::string dtype_str = argv[++i];
            if (dtype_str == "auto") {
                config.cache.kv_cache_dtype = "auto";
            } else if (dtype_str == "float16") {
                config.cache.kv_cache_dtype = "fp16";
            } else if (dtype_str == "bfloat16") {
                config.cache.kv_cache_dtype = "bf16";
            } else if (dtype_str == "fp8_e4m3") {
                config.cache.kv_cache_dtype = "fp8_e4m3";
            } else if (dtype_str == "fp8_e5m2") {
                config.cache.kv_cache_dtype = "fp8_e5m2";
            } else if (dtype_str == "int8_per_token_head") {
                config.cache.kv_cache_dtype = "int8_per_token_head";
            } else if (dtype_str == "fp8_per_token_head") {
                config.cache.kv_cache_dtype = "fp8_per_token_head";
            } else if (dtype_str == "nvfp4") {
                config.cache.kv_cache_dtype = "nvfp4";
            } else if (dtype_str.find("turboquant") != std::string::npos) {
                config.cache.kv_cache_dtype = dtype_str;
                config.cache.kv_cache_quant_method = dtype_str;
            } else {
                spdlog::error("Unsupported --kv-cache-dtype: '{}'", dtype_str);
                std::exit(1);
            }
        } else if (arg == "--max-model-len" && i + 1 < argc) {
            config.model.max_model_len_from_cli = true;
            config.model.max_model_len = std::stoll(argv[++i]);
        } else if (arg == "--gpu-memory-utilization" && i + 1 < argc) {
            config.cache.gpu_memory_utilization = std::stof(argv[++i]);
        } else if (arg == "--gpu-safety-margin-mb" && i + 1 < argc) {
            config.cache.gpu_safety_margin_mb = std::stoll(argv[++i]);
        } else if (arg == "--tensor-parallel-size" && i + 1 < argc) {
            try {
                config.parallel.tensor_parallel_size = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                spdlog::error("Invalid value for --tensor-parallel-size: '{}'", argv[i]);
                std::exit(1);
            }
        } else if (arg == "--spec-method" && i + 1 < argc) {
            config.model.spec_method = argv[++i];
            config.model.spec_method_from_cli = true;
            spdlog::info("Speculative decoding method: {}", config.model.spec_method);
        } else if (arg == "--spec-draft-n-max" && i + 1 < argc) {
            try {
                config.model.mtp_spec_width = std::stoi(argv[++i]);
            } catch (const std::exception& e) {
                spdlog::error("Invalid value for --spec-draft-n-max: '{}'", argv[i]);
                std::exit(1);
            }
            if (config.model.mtp_spec_width <= 0) {
                spdlog::error("--spec-draft-n-max must be positive, got {}", config.model.mtp_spec_width);
                std::exit(1);
            }
            spdlog::info("MTP spec draft n_max: {}", config.model.mtp_spec_width);
        } else if (arg == "--spec-draft-n-min" && i + 1 < argc) {
            try {
                config.model.mtp_spec_n_min = std::stoi(argv[++i]);
            } catch (const std::exception&) {
                spdlog::error("Invalid value for --spec-draft-n-min: '{}'", argv[i]);
                std::exit(1);
            }
            if (config.model.mtp_spec_n_min < 0) {
                spdlog::error("--spec-draft-n-min must be non-negative, got {}", config.model.mtp_spec_n_min);
                std::exit(1);
            }
            spdlog::info("MTP spec draft n_min: {}", config.model.mtp_spec_n_min);
        } else if (arg == "--spec-draft-p-min" && i + 1 < argc) {
            try {
                config.model.mtp_spec_p_min = std::stof(argv[++i]);
            } catch (const std::exception&) {
                spdlog::error("Invalid value for --spec-draft-p-min: '{}'", argv[i]);
                std::exit(1);
            }
            if (config.model.mtp_spec_p_min < 0.0f || config.model.mtp_spec_p_min > 1.0f) {
                spdlog::error("--spec-draft-p-min must be in [0, 1], got {}", config.model.mtp_spec_p_min);
                std::exit(1);
            }
            spdlog::info("MTP spec draft p_min: {}", config.model.mtp_spec_p_min);
        } else if (arg == "--gpu-devices" && i + 1 < argc) {
            std::string devices_str = argv[++i];
            std::stringstream ss(devices_str);
            std::string token;
            while (std::getline(ss, token, ',')) {
                try {
                    config.parallel.gpu_devices.push_back(std::stoi(token));
                } catch (const std::exception& e) {
                    spdlog::error("Invalid GPU device ID in --gpu-devices: '{}'", token);
                    std::exit(1);
                }
            }
            if (config.parallel.gpu_devices.empty()) {
                spdlog::error("--gpu-devices requires at least one device ID");
                std::exit(1);
            }
            spdlog::info("GPU devices specified: {}", devices_str);
        } else if (arg == "--served-model-name" && i + 1 < argc) {
            config.server.served_model_name = argv[++i];
        } else if (arg == "--host" && i + 1 < argc) {
            config.server.host = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.server.port = std::stoi(argv[++i]);
        } else if (arg == "--system-fingerprint" && i + 1 < argc) {
            config.server.system_fingerprint = argv[++i];
        } else if (arg == "--cpu-offload-gb" && i + 1 < argc) {
            config.cache.cpu_offload_gb = std::stoll(argv[++i]);
        } else if (arg == "--cpu-offload-params" && i + 1 < argc) {
            config.cache.cpu_offload_params.insert(argv[++i]);
        } else if (arg == "--max-num-seqs" && i + 1 < argc) {
            config.server.max_num_seqs = std::stoi(argv[++i]);
        } else if (arg == "--max-num-batched-tokens" && i + 1 < argc) {
            config.server.max_num_batched_tokens = std::stoi(argv[++i]);
        } else if (arg == "--kv-offloading-size" && i + 1 < argc) {
            config.cache.kv_offloading_size = std::stoll(argv[++i]);
        } else if (arg == "--expert-gpu-cache" && i + 1 < argc) {
            config.cache.expert_gpu_cache_capacity = std::stoi(argv[++i]);
        } else if (arg == "--expert-cache-update-period" && i + 1 < argc) {
            config.cache.expert_cache_update_period = std::stoi(argv[++i]);
        } else if (arg == "--expert-cache-history-window" && i + 1 < argc) {
            config.cache.expert_cache_history_window = std::stoi(argv[++i]);
        } else if (arg == "--poll-interval" && i + 1 < argc) {
            config.server.poll_interval_ms = std::stoi(argv[++i]);
        } else if (arg == "--thread-pool-size" && i + 1 < argc) {
            config.server.thread_pool_size = std::stoi(argv[++i]);
        } else if (arg == "--enable-prefix-caching") {
            config.cache.enable_prefix_caching = true;
        } else if (arg == "--disable-prefix-caching") {
            config.cache.enable_prefix_caching = false;
        } else if (arg == "--enable-chunked-prefill") {
            config.scheduler.enable_chunked_prefill = true;
        } else if (arg == "--disable-chunked-prefill") {
            config.scheduler.enable_chunked_prefill = false;
        } else if (arg == "--long-prefill-token-threshold" && i + 1 < argc) {
            config.scheduler.long_prefill_token_threshold = std::stoi(argv[++i]);
        } else if (arg == "--offload-group-size" && i + 1 < argc) {
            config.offload.offload_group_size = std::stoi(argv[++i]);
        } else if (arg == "--offload-prefetch-step" && i + 1 < argc) {
            config.offload.offload_prefetch_step = std::stoi(argv[++i]);
        } else if (arg == "--kv-offloading-backend" && i + 1 < argc) {
            config.offload.kv_offloading_backend = argv[++i];
        }
    }

    return config;
}

struct PendingRequest {
    std::string request_id;
    std::vector<int32_t> output_tokens;
    bool finished = false;
    std::string finish_reason = "stop";
    std::mutex mutex;
    std::condition_variable cv;
    std::function<void(const std::string&, bool, const std::string&)> stream_callback;
};

int main(int argc, char* argv[]) {
    spdlog::cfg::load_env_levels();
    // 用户可通过 SPDLOG_LEVEL=info 环境变量覆盖为 info 级别
    if (!std::getenv("SPDLOG_LEVEL")) {
        spdlog::set_level(spdlog::level::debug);
    }

    vm_c::GpuArch::instance().detect();
    spdlog::info("GPU architecture detected: SM{}.{}, compute_dtype={}",
                 vm_c::gpu().sm_major(), vm_c::gpu().sm_minor(),
                 vm_c::gpu().supports_bf16() ? "BF16" : "FP16");

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    spdlog::info(R"(
  __   __  ___   __   __      ___
 |  \ /  \|   \ /  \ /  \    | __
 |   v   ||   v   v   v   |  | _|
  \_/\_/ |__/\_/\_/\_/\_/   |___|
  C++ Inference Engine - v0.1
)");

    auto config = parse_args(argc, argv);

    llama_backend_init();
    spdlog::info("libllama backend initialized");

    g_poll_interval = std::chrono::milliseconds(config.server.poll_interval_ms);
    g_step_interval = std::chrono::milliseconds(config.server.step_interval_ms);

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        spdlog::error("No CUDA devices found");
        return 1;
    }

    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        spdlog::info("GPU {}: {} ({} SMs, {:.1f} GB)",
                     i, prop.name, prop.multiProcessorCount,
                     prop.totalGlobalMem / 1e9);
    }

    int cuda_rt = 0;
    if (cudaRuntimeGetVersion(&cuda_rt) == cudaSuccess) {
        const int maj = cuda_rt / 1000;
        const int min = (cuda_rt % 1000) / 10;
        spdlog::info("CUDA runtime reported: {}.{} (driver-linked toolkit)", maj, min);
    }

    spdlog::info("Configuration:");
    spdlog::info("  Model: {}", config.model.model_dir);
    spdlog::info("  Tokenizer: {}", config.model.tokenizer_dir.empty() ? config.model.model_dir : config.model.tokenizer_dir);
    spdlog::info("  Served model name: {}", config.server.served_model_name);
    spdlog::info("  Max model len (CLI cap): {}{}",
                 config.model.max_model_len,
                 config.model.max_model_len_from_cli ? "" : " (use HF max_position_embeddings unless capped after load)");
    spdlog::info("  GPU memory utilization: {:.0f}%", config.cache.gpu_memory_utilization * 100);
    spdlog::info("  KV cache dtype: {}", config.cache.kv_cache_dtype);
    if (!config.cache.kv_cache_quant_method.empty()) {
        spdlog::info("  KV cache quant method: {}", config.cache.kv_cache_quant_method);
    }
    spdlog::info("  GPU devices: {}", config.parallel.gpu_devices.empty() ?
                 "default (0,1,...)" : [&]() {
                     std::string s;
                     for (size_t i = 0; i < config.parallel.gpu_devices.size(); ++i) {
                         if (i > 0) s += ",";
                         s += std::to_string(config.parallel.gpu_devices[i]);
                     }
                     return s;
                 }());
    spdlog::info("  KV CPU offload budget: {} GiB (--kv-offloading-size)", config.cache.kv_offloading_size);
    spdlog::info("  Tensor parallel: {}", config.parallel.tensor_parallel_size);
    spdlog::info("  Max num seqs: {}", config.server.max_num_seqs);
    spdlog::info("  Max num batched tokens: {}", config.server.max_num_batched_tokens);
    spdlog::info("  Prefix caching: {}", config.cache.enable_prefix_caching ? "on" : "off");
    spdlog::info("  CPU offload weights budget: {} GB (--cpu-offload-gb)", config.cache.cpu_offload_gb);
    if (!config.cache.cpu_offload_params.empty()) {
        std::ostringstream params_str;
        bool first = true;
        for (const auto& p : config.cache.cpu_offload_params) {
            if (!first) params_str << " ";
            params_str << p;
            first = false;
        }
        spdlog::info("  CPU offload params: {}", params_str.str());
    }

    // ── 检测是否为 GGUF 文件并创建 tokenizer ──
    std::unique_ptr<vm_c::GgufReader> gguf_reader;
    struct stat path_stat;
    bool is_gguf_path = false;
    if (stat(config.model.model_dir.c_str(), &path_stat) == 0 && S_ISREG(path_stat.st_mode)) {
        std::string p = config.model.model_dir;
        is_gguf_path = (p.size() >= 5) && (p.substr(p.size() - 5) == ".gguf" ||
                       p.find(".gguf") != std::string::npos);
    }

    std::unique_ptr<vm_c::Tokenizer> tokenizer;
    vm_c::ModelConfig model_cfg;

    if (is_gguf_path && config.model.tokenizer_dir.empty()) {
        // GGUF 模型: 从文件内部读取 tokenizer 和 config
        gguf_reader = std::make_unique<vm_c::GgufReader>();
        if (!gguf_reader->load(config.model.model_dir)) {
            spdlog::error("Failed to load GGUF file: {}", config.model.model_dir);
            return 1;
        }
        model_cfg = vm_c::parse_gguf_config(config.model.model_dir);
        tokenizer = vm_c::create_gguf_tokenizer(*gguf_reader);
        spdlog::info("Tokenizer loaded from GGUF metadata ({} tokens)", tokenizer->vocab_size());
    } else {
        // 从外部路径加载 tokenizer 和 config
        std::string tok_dir = config.model.tokenizer_dir.empty()
            ? config.model.model_dir
            : config.model.tokenizer_dir;
        tokenizer = vm_c::create_tokenizer(tok_dir);
        // GGUF 文件但显式指定了 tokenizer 目录: 模型 config 仍从 GGUF
        if (is_gguf_path) {
            model_cfg = vm_c::parse_gguf_config(config.model.model_dir);
        } else {
            model_cfg = vm_c::load_model_config_from_json(config.model.model_dir);
        }
    }

    {
        config.model.vocab_size = model_cfg.vocab_size;
        const int tk_vocab = tokenizer->vocab_size();
        const int cfg_vocab = config.model.vocab_size;
        if (tk_vocab < cfg_vocab) {
            spdlog::info("Tokenizer vocab_size ({}) < model config vocab_size ({}), "
                         "using tokenizer vocab_size for logits/sampling",
                         tk_vocab, cfg_vocab);
            config.model.vocab_size = tk_vocab;
        } else if (tk_vocab > cfg_vocab) {
            spdlog::warn("Tokenizer vocab_size ({}) > model config vocab_size ({}), "
                         "some tokens may not have embeddings",
                         tk_vocab, cfg_vocab);
        }

        // 从 GGUF config 中提取 MTP 层数等元数据
        if (model_cfg.mtp_predict_layers > 0) {
            config.model.mtp_predict_layers = model_cfg.mtp_predict_layers;
            // 仅当用户未通过 CLI 显式指定 spec_method 时，才使用 GGUF 元数据
            if (!config.model.spec_method_from_cli) {
                config.model.spec_method = model_cfg.spec_method;
                spdlog::info("GGUF: MTP predict layers={}, spec_method={}",
                             config.model.mtp_predict_layers, config.model.spec_method);
            } else {
                spdlog::info("GGUF: MTP predict layers={}, CLI spec_method='{}' (preserved)",
                             config.model.mtp_predict_layers, config.model.spec_method);
            }
        }
    }

    auto engine = std::make_unique<vm_c::Engine>(config);
    engine->initialize();

    spdlog::info("Effective runtime max_model_len={} (min of HF context and --max-model-len when set)",
                 engine->config().model.max_model_len);

    // 使用 llama.cpp 官方的 llama_vocab 替换 GgufTokenizer，确保 tokenization 完全对齐官方实现
    {
        const auto* vocab = engine->vocab();
        if (!vocab) {
            spdlog::error("Engine vocab not available");
            engine->shutdown();
            return 1;
        }
        std::string chat_tmpl = tokenizer->chat_template();
        if (chat_tmpl.empty()) {
            // GGUF 元数据未带 chat_template → 退回官方 Qwen3 模板
            spdlog::warn("No chat_template in GGUF, falling back to built-in Qwen3 template");
            chat_tmpl = vm_c::get_qwen3_fallback_template();
        } else {
            spdlog::info("Using chat_template from GGUF ({} chars)", chat_tmpl.size());
        }
        tokenizer = std::make_unique<VocabTokenizer>(vocab, chat_tmpl);
        spdlog::info("VocabTokenizer created: {} tokens, eos={}", tokenizer->vocab_size(), tokenizer->eos_token_id());
    }

    std::mutex pending_mutex;
    std::unordered_map<std::string, std::shared_ptr<PendingRequest>> pending_requests;

    vm_c::HttpServer server(config);
    server.set_chat_template(tokenizer->chat_template());

    server.set_generate_fn([&](const std::string& prompt,
                                 const vm_c::SamplingParams& params,
                                  const std::string& model) -> vm_c::GenerateResult {
        auto token_ids = tokenizer->encode(prompt, false);
        int prompt_tokens = static_cast<int>(token_ids.size());

        // vm_c 诊断日志 v3（已确认 tokenize/detokenize round-trip 正确）：
        //   [PROMPT-SUMMARY] 单行摘要：n_tokens, first/last 5 token id，roundtrip match
        //   VMC_LOG_PROMPT_VERBOSE=1 开启完整 PROMPT-FULL/HEX/TOKENS/RECONSTRUCT（默认 0）
        {
            static const char* env_verbose = getenv("VMC_LOG_PROMPT_VERBOSE");
            const bool verbose = env_verbose && env_verbose[0] == '1';
            if (verbose) {
                spdlog::info("[PROMPT-FULL] jinja_rendered ({} bytes): {}", prompt.size(), prompt);
                const size_t hex_len = std::min<size_t>(200, prompt.size());
                std::string hex;
                hex.reserve(hex_len * 3);
                for (size_t i = 0; i < hex_len; ++i) {
                    char buf[4];
                    std::snprintf(buf, sizeof(buf), "%02x ", static_cast<unsigned char>(prompt[i]));
                    hex += buf;
                }
                spdlog::info("[PROMPT-HEX] first {} bytes: {}", hex_len, hex);
                spdlog::info("[PROMPT-TOKENS] n_tokens={}", token_ids.size());
                for (size_t ti = 0; ti < token_ids.size(); ++ti) {
                    std::vector<int32_t> one_id = { token_ids[ti] };
                    std::string piece = tokenizer->decode(one_id, false);
                    spdlog::info("[PROMPT-TOKENS] [{:03d}] id={:6d}  piece=\"{}\"",
                                 ti, token_ids[ti], piece);
                }
                std::string reconstructed = tokenizer->decode(token_ids, false);
                bool roundtrip_match = (reconstructed == prompt);
                spdlog::info("[PROMPT-RECONSTRUCT] match={} bytes={} text={}",
                             roundtrip_match ? "YES" : "NO",
                             reconstructed.size(), reconstructed);
            } else {
                // 摘要：每个请求一行
                std::string reconstructed = tokenizer->decode(token_ids, false);
                bool roundtrip_match = (reconstructed == prompt);
                std::string first5, last5;
                for (int i = 0; i < std::min<size_t>(5, token_ids.size()); ++i) {
                    first5 += std::to_string(token_ids[i]) + " ";
                }
                for (int i = (int)std::max<size_t>(0, token_ids.size() - 5); i < (int)token_ids.size(); ++i) {
                    last5 += std::to_string(token_ids[i]) + " ";
                }
                spdlog::info("[PROMPT-SUMMARY] n_tokens={} first5=[{}] last5=[{}] rt_match={} (VMC_LOG_PROMPT_VERBOSE=1 开启详细)",
                             token_ids.size(), first5, last5, roundtrip_match ? "Y" : "N");
            }
        }

        auto pending = std::make_shared<PendingRequest>();

        vm_c::SamplingParams effective_params = params;
        for (auto eos_id : config.model.eos_token_ids) {
            if (std::find(effective_params.stop_token_ids.begin(),
                         effective_params.stop_token_ids.end(),
                         eos_id) == effective_params.stop_token_ids.end()) {
                effective_params.stop_token_ids.push_back(eos_id);
            }
        }

        auto detok = std::make_shared<vm_c::IncrementalDetokenizer>(
            tokenizer.get(), token_ids, effective_params);
        std::string accumulated_content;
        std::string accumulated_reasoning;

        std::string req_id = engine->submit_request(token_ids, effective_params, model,
            [detok, &accumulated_content, &accumulated_reasoning, pending](const std::vector<int32_t>& tokens, bool finished, vm_c::FinishReason reason) {
                std::lock_guard<std::mutex> lock(pending->mutex);
                pending->output_tokens.insert(pending->output_tokens.end(),
                                              tokens.begin(), tokens.end());
                pending->finished = finished;
                if (finished) {
                    pending->finish_reason = finish_reason_str(reason);
                    pending->cv.notify_one();
                }
            });

        g_has_work.store(true, std::memory_order_release);
        g_step_cv.notify_one();

        pending->request_id = req_id;
        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            pending_requests[req_id] = pending;
        }

        {
            std::unique_lock<std::mutex> lock(pending->mutex);
            while (!pending->finished && g_running) {
                pending->cv.wait_for(lock, g_poll_interval);
            }
            if (!pending->finished) {
                spdlog::info("Generate request {} cancelled due to shutdown", req_id);
                lock.unlock();
                engine->abort_request(req_id);
                lock.lock();
                {
                    std::lock_guard<std::mutex> lock2(pending_mutex);
                    pending_requests.erase(req_id);
                }
                const int partial = static_cast<int>(pending->output_tokens.size());
                return {"[ERROR: server shutting down]", "", "abort", prompt_tokens, partial};
            }
        }

        {
            std::lock_guard<std::mutex> lock(pending_mutex);
            pending_requests.erase(req_id);
        }

        int completion_tokens = static_cast<int>(pending->output_tokens.size());
        spdlog::info("Generate done: output_tokens={}", completion_tokens);

        // 非流式场景：直接 decode 全部 output tokens，不使用增量 detokenizer
        // 增量 detokenizer 的 delta 计算会包含 prompt 尾部 token 的解码文本，导致泄漏
        std::string full_raw = tokenizer->decode(pending->output_tokens, true);

        // 对齐官方 PEG parser 流程：
        // 完整文本 = prompt + output，prompt 末尾 = <|im_start|>assistant\n<think>\n
        // grammar: generation_prompt + optional(<think> + reasoning + </think>) + content(rest)
        // 模型输出中，</think> 之前的为 reasoning，之后为 text
        // 若模型未输出 </think>（EOS），则全部为 text（optional 块不匹配，走 content 兜底）
        std::string full_text;
        std::string full_reasoning;
        if (effective_params.enable_thinking) {
            static const std::string kThinkEnd = "</think>";
            auto end_pos = full_raw.find(kThinkEnd);
            if (end_pos != std::string::npos) {
                full_reasoning = full_raw.substr(0, end_pos);
                full_text = full_raw.substr(end_pos + kThinkEnd.size());
                while (!full_text.empty() && (full_text[0] == '\n' || full_text[0] == '\r')) {
                    full_text = full_text.substr(1);
                }
            } else {
                full_text = full_raw;
            }
        } else {
            full_text = full_raw;
        }

        {
            std::string tok_str;
            for (size_t ti = 0; ti < pending->output_tokens.size(); ++ti) {
                if (ti > 0) tok_str += ",";
                tok_str += std::to_string(pending->output_tokens[ti]);
            }
            spdlog::info("[RESULT] prompt_tokens={} completion_tokens={} raw_tokens=[{}]",
                         prompt_tokens, completion_tokens, tok_str);
        }

        // vm_c 诊断日志 v3：[DETOK] 已确认 tokenize/detokenize 一致（round-trip 验证通过）
        // 改用可选详细模式：默认只打 [RESULT-SUMMARY] 单行（已在上面打印 raw_tokens）
        // VMC_LOG_DETOK=1 开启逐 token DETOK
        {
            static const char* env_detok = getenv("VMC_LOG_DETOK");
            const bool enable = env_detok && env_detok[0] == '1';
            if (enable) {
                for (size_t ti = 0; ti < pending->output_tokens.size(); ++ti) {
                    int32_t id = pending->output_tokens[ti];
                    std::vector<int32_t> one_id = { id };
                    std::string piece = tokenizer->decode(one_id, false);
                    std::string hex;
                    const size_t hex_len = std::min<size_t>(32, piece.size());
                    hex.reserve(hex_len * 3);
                    for (size_t i = 0; i < hex_len; ++i) {
                        char buf[4];
                        std::snprintf(buf, sizeof(buf), "%02x ",
                                      static_cast<unsigned char>(piece[i]));
                        hex += buf;
                    }
                    spdlog::info("[DETOK] [{:03d}] id={:6d}  piece=\"{}\" hex={}",
                                 ti, id, piece, hex);
                }
            }
        }
        spdlog::info("[RESULT] full_raw ({} bytes): {}",
                     full_raw.size(), full_raw.empty() ? "(empty)" : full_raw);

        return {full_text, full_reasoning, pending->finish_reason, prompt_tokens, completion_tokens};
    });

    server.set_stream_fn([&](const std::string& prompt,
                              const vm_c::SamplingParams& params,
                              const std::string& model,
                               std::function<void(const std::string&, const std::string&, bool, const std::string&)> callback) -> int {
        auto token_ids = tokenizer->encode(prompt, false);
        int prompt_tokens = static_cast<int>(token_ids.size());
        spdlog::info("Stream request: prompt_tokens={}", prompt_tokens);

        vm_c::SamplingParams effective_params = params;
        for (auto eos_id : config.model.eos_token_ids) {
            if (std::find(effective_params.stop_token_ids.begin(),
                         effective_params.stop_token_ids.end(),
                         eos_id) == effective_params.stop_token_ids.end()) {
                effective_params.stop_token_ids.push_back(eos_id);
            }
        }

        std::mutex done_mutex;
        std::condition_variable done_cv;
        bool done = false;

        auto detokenizer = std::make_shared<vm_c::IncrementalDetokenizer>(
            tokenizer.get(), token_ids, effective_params);

        std::string req_id = engine->submit_request(token_ids, effective_params, model,
            [detokenizer, callback, &done_mutex, &done_cv, &done](const std::vector<int32_t>& tokens, bool finished, vm_c::FinishReason reason) {
                std::string reason_str = finish_reason_str(reason);
                if (!tokens.empty()) {
                    bool stop_terminated = (reason == vm_c::FinishReason::STOP);
                    auto result = detokenizer->update(tokens.back(), stop_terminated);
                    if (result.stop_hit) {
                        reason_str = "stop";
                    }
                    if (!result.delta_text.empty() || !result.delta_reasoning.empty() || finished) {
                        callback(result.delta_text, result.delta_reasoning, finished, reason_str);
                    }
                } else if (finished) {
                    callback("", "", true, reason_str);
                }
                if (finished) {
                    std::lock_guard<std::mutex> lock(done_mutex);
                    done = true;
                    done_cv.notify_one();
                }
            });

        g_has_work.store(true, std::memory_order_release);
        g_step_cv.notify_one();

        // 对齐 llama.cpp server：无生成时长上限，仅等待 finished 或服务 shutdown
        std::unique_lock<std::mutex> lock(done_mutex);
        while (!done && g_running) {
            done_cv.wait_for(lock, g_poll_interval);
        }
        if (!done) {
            spdlog::info("Stream request {} cancelled due to shutdown", req_id);
            lock.unlock();
            engine->abort_request(req_id);
        }

        return prompt_tokens;
    });

    server.set_abort_fn([&](const std::string& request_id) {
        engine->abort_request(request_id);
    });

    server.set_tokenize_fn([&](const std::string& text) -> std::vector<int32_t> {
        return tokenizer->encode(text);
    });

    server.set_detokenize_fn([&](const std::vector<int32_t>& ids, bool spaces_between_special_tokens) -> std::string {
        return tokenizer->decode(ids, true, spaces_between_special_tokens);
    });

    try {
        server.start();
    } catch (const std::exception& e) {
        spdlog::error("Failed to start HTTP server: {}", e.what());
        engine->shutdown();
        return 1;
    }

    spdlog::info("vm_c_server ready on {}:{} serving model '{}'",
                 config.server.host, config.server.port,
                 config.server.served_model_name);
    spdlog::info("Endpoints:");
    spdlog::info("  POST /v1/chat/completions  - Chat completions (streaming supported)");
    spdlog::info("  POST /v1/completions       - Text completions");
    spdlog::info("  GET  /v1/models            - List models");
    spdlog::info("  GET  /health               - Health check");

    {
        std::unique_lock<std::mutex> lock(g_step_mutex);
        while (g_running) {
            g_step_cv.wait_for(lock, g_step_interval, [&] {
                return !g_running || g_has_work.load(std::memory_order_acquire);
            });
            if (!g_running) break;
            lock.unlock();
            try {
                auto outputs = engine->step();
                if (outputs.empty()) {
                    if (engine->num_running_requests() > 0 || engine->num_waiting_requests() > 0) {
                        // schedule 无法推进时避免 10ms 空转刷屏
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                }
            } catch (const std::exception& e) {
                spdlog::error("[MAIN] engine->step() failed: {}", e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (engine->num_running_requests() > 0 || engine->num_waiting_requests() > 0) {
                g_has_work.store(true, std::memory_order_release);
            } else {
                g_has_work.store(false, std::memory_order_release);
            }
            lock.lock();
        }
    }

    spdlog::info("Shutting down...");
    server.stop();
    engine->shutdown();

    llama_backend_free();

    spdlog::info("vm_c_server stopped");
    return 0;
}
