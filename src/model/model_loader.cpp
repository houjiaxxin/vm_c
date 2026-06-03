#include "vm_c/model/model_loader.hpp"
#include "vm_c/core/config.hpp"
#include "vm_c/core/tensor.hpp"
#include "vm_c/core/gguf_reader.hpp"
#include "vm_c/cuda/gpu_arch.hpp"

#include <nlohmann/json.hpp>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string_view>
#include <atomic>
#include <omp.h>
#include <unordered_set>
#include <spdlog/spdlog.h>
#include <functional>
#include <future>

#define VMC_CHECK_CUDA(call, msg) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        spdlog::error("{}: {} (at {}:{})", msg, cudaGetErrorString(_err), __FILE__, __LINE__); \
        throw std::runtime_error(msg); \
    } \
} while(0)

namespace vm_c {

static std::string normalize_weight_name(const std::string& name) {
    // vLLM 参考: hf_to_vllm_mapper (qwen3_vl.py:1627-1634)
    //   "model.visual."         -> "visual."              (视觉权重，跳过)
    //   "lm_head."              -> "language_model.lm_head."
    //   "model.language_model." -> "language_model.model."
    //
    // vm_c ModelLoader 直接映射 GGUF tensor 名，无 vLLM 额外 .model. 包装层。
    // 所以我们将 model.language_model.xxx 直接归一化为 model.xxx。
    // lm_head 和 embed_tokens 等顶层权重预期带有 model. 前缀。
    static const std::pair<std::string, std::string> prefix_map[] = {
        {"model.language_model.model.", "model."},   // 最内层包装（Qwen3.5 常见）
        {"model.language_model.",       "model."},
        {"language_model.model.",       "model."},
        {"language_model.",             "model."},
        {"model.model.",                "model."},   // 可能的重复前缀
        {"base_model.model.",           "model."},   // PEFT/LoRA 模式
    };
    std::string result = name;
    for (auto& [alt, std] : prefix_map) {
        if (result.find(alt) == 0) {
            result = std + result.substr(alt.size());
            break;
        }
    }
    // 针对 lm_head 不带前缀的情况：直接保留为 lm_head.weight
    // extract_weight_any 会尝试多种 tensor 命名前缀。
    // 使用 string::find/replace 替代 std::regex_replace 以获得 10-50 倍加速
    {   auto p = result.find(".switch_mlp.");
        if (p != std::string::npos) result.replace(p, 13, ".experts."); }
    return result;
}

static std::string shape_to_str(const vm_c::Shape& shape) {
    std::string s;
    for (size_t i = 0; i < shape.ndim(); i++) {
        if (i > 0) s += "x";
        s += std::to_string(shape[i]);
    }
    return s;
}

using json = nlohmann::json;

SafeTensorsLoader::SafeTensorsLoader(const std::string& model_dir)
    : model_dir_(model_dir) {
    if (!load_index()) {
        throw std::runtime_error("Failed to load safetensors index from: " + model_dir);
    }
}

bool SafeTensorsLoader::load_index() {
    std::string index_path = model_dir_ + "/model.safetensors.index.json";
    std::string single_path = model_dir_ + "/model.safetensors";

    std::ifstream f(index_path);
    if (f.is_open()) {
        json j;
        f >> j;
        if (j.contains("weight_map")) {
            std::unordered_map<std::string, std::string> tensor_to_file;
            for (auto& [name, filename] : j["weight_map"].items()) {
                tensor_to_file[name] = filename.get<std::string>();
            }

            std::unordered_set<std::string> shard_files_set;
            for (auto& [name, filename] : tensor_to_file) {
                shard_files_set.insert(model_dir_ + "/" + filename);
            }

            shard_files_.assign(shard_files_set.begin(), shard_files_set.end());
            std::sort(shard_files_.begin(), shard_files_.end());

            spdlog::info("Loading {} tensors from {} shard files...",
                         tensor_to_file.size(), shard_files_.size());

            for (auto& shard_path : shard_files_) {
                std::ifstream sf(shard_path, std::ios::binary);
                if (!sf.is_open()) {
                    spdlog::error("Cannot open shard file: {}", shard_path);
                    continue;
                }

                uint64_t header_size;
                sf.read(reinterpret_cast<char*>(&header_size), 8);
                std::vector<char> header_buf(header_size);
                sf.read(header_buf.data(), header_size);

                try {
                    json header = json::parse(header_buf);
                    for (auto& [name, info] : header.items()) {
                        if (name == "__metadata__") continue;

                        auto tit = tensor_to_file.find(name);
                        if (tit == tensor_to_file.end()) continue;

                        WeightTensor wt;
                        wt.name = name;
                        wt.shard_file = shard_path;
                        auto& offsets = info["data_offsets"];
                        wt.offset = offsets[0].get<int64_t>();
                        wt.nbytes = offsets.size() > 1
                            ? offsets[1].get<int64_t>() - wt.offset
                            : 0;
                        auto& shape = info["shape"];
                        for (auto& s : shape) wt.shape.dims.push_back(s.get<int64_t>());

                        std::string dtype_str = info.value("dtype", gpu().supports_bf16() ? "BF16" : "F16");
                        if (dtype_str == "F32" || dtype_str == "FLOAT") wt.dtype = DataType::FLOAT32;
                        else if (dtype_str == "F16" || dtype_str == "FLOAT16") wt.dtype = DataType::FLOAT16;
                        else if (dtype_str == "BF16" || dtype_str == "BFLOAT16") wt.dtype = DataType::BFLOAT16;
                        else if (dtype_str == "F8_E4M3" || dtype_str == "FLOAT8_E4M3FN") wt.dtype = DataType::FLOAT8_E4M3;
                        else if (dtype_str == "F8_E5M2") wt.dtype = DataType::FLOAT8_E5M2;
                        else if (dtype_str == "U8" || dtype_str == "UINT8") wt.dtype = DataType::INT8;
                        else if (dtype_str == "I32" || dtype_str == "INT32") wt.dtype = DataType::INT32;
                        else if (dtype_str == "U32" || dtype_str == "UINT32") wt.dtype = DataType::UINT32;
                        else wt.dtype = gpu().supports_bf16() ? DataType::BFLOAT16 : DataType::FLOAT16;

                        meta_[name] = wt;
                    }
                } catch (const json::exception& e) {
                    spdlog::error("Failed to parse shard header {}: {}", shard_path, e.what());
                    continue;
                }
            }
        }
        spdlog::info("Loaded safetensors index: {} tensors", meta_.size());
        return true;
    }

    f.open(single_path, std::ios::binary);
    if (f.is_open()) {
        uint64_t header_size;
        f.read(reinterpret_cast<char*>(&header_size), 8);
        std::vector<char> header_buf(header_size);
        f.read(header_buf.data(), header_size);

        try {
            json header = json::parse(header_buf);
            for (auto& [name, info] : header.items()) {
                if (name == "__metadata__") continue;
                WeightTensor wt;
                wt.name = name;
                wt.shard_file = single_path;
                auto& offsets = info["data_offsets"];
                wt.offset = offsets[0].get<int64_t>();
                wt.nbytes = offsets.size() > 1
                    ? offsets[1].get<int64_t>() - wt.offset
                    : 0;
                auto& shape = info["shape"];
                for (auto& s : shape) wt.shape.dims.push_back(s.get<int64_t>());

                std::string dtype_str = info.value("dtype", gpu().supports_bf16() ? "BF16" : "F16");
                if (dtype_str == "F32" || dtype_str == "FLOAT") wt.dtype = DataType::FLOAT32;
                else if (dtype_str == "F16" || dtype_str == "FLOAT16") wt.dtype = DataType::FLOAT16;
                else if (dtype_str == "BF16" || dtype_str == "BFLOAT16") wt.dtype = DataType::BFLOAT16;
                else if (dtype_str == "F8_E4M3" || dtype_str == "FLOAT8_E4M3FN") wt.dtype = DataType::FLOAT8_E4M3;
                else if (dtype_str == "F8_E5M2") wt.dtype = DataType::FLOAT8_E5M2;
                else if (dtype_str == "U8" || dtype_str == "UINT8") wt.dtype = DataType::INT8;
                else if (dtype_str == "I32" || dtype_str == "INT32") wt.dtype = DataType::INT32;
                else if (dtype_str == "U32" || dtype_str == "UINT32") wt.dtype = DataType::UINT32;
                else wt.dtype = gpu().supports_bf16() ? DataType::BFLOAT16 : DataType::FLOAT16;

                meta_[name] = wt;
            }
        } catch (const json::exception& e) {
            spdlog::error("Failed to parse safetensors header: {}", e.what());
            return false;
        }
        shard_files_ = {single_path};
        spdlog::info("Loaded single safetensors file: {} tensors", meta_.size());
        return true;
    }

    spdlog::error("No safetensors file found in: {}", model_dir_);
    return false;
}

std::vector<std::string> SafeTensorsLoader::tensor_names() const {
    std::vector<std::string> names;
    for (auto& [n, _] : meta_) names.push_back(n);
    return names;
}

const WeightTensor& SafeTensorsLoader::tensor_meta(const std::string& name) const {
    return meta_.at(name);
}

bool SafeTensorsLoader::has_tensor(const std::string& name) const {
    return meta_.find(name) != meta_.end();
}

CpuTensor SafeTensorsLoader::load_tensor(const std::string& name) {
    auto& wt = meta_.at(name);
    CpuTensor tensor(wt.shape, wt.dtype);

    std::ifstream f(wt.shard_file, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open safetensors data file: " + wt.shard_file);
    }
    f.seekg(8 + wt.offset);
    f.read(static_cast<char*>(tensor.cpu_ptr()), wt.nbytes);

    return tensor;
}


void SafeTensorsLoader::load_shard_tensors(
    const std::string& shard_path,
    std::function<void(const std::string&, CpuTensor&&)> on_tensor) {
    // mmap 整个文件（vLLM safe_open 等效），消除 93273 次 seekg+read 系统调用
    int fd = open(shard_path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("Cannot open safetensors data file: " + shard_path);
    struct stat st; fstat(fd, &st);
    void* mapped = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (mapped == MAP_FAILED)
        throw std::runtime_error("mmap failed for: " + shard_path);

    uint64_t header_size;
    std::memcpy(&header_size, mapped, 8);
    std::vector<char> header_buf(header_size);
    std::memcpy(header_buf.data(), static_cast<const char*>(mapped) + 8, header_size);

    json header = json::parse(header_buf);
    for (auto& [name, info] : header.items()) {
        if (name == "__metadata__") continue;

        if (name.find("visual.") != std::string::npos ||
            name.find("vision.") != std::string::npos) continue;

        auto& offsets = info["data_offsets"];
        int64_t offset = offsets[0].get<int64_t>();
        int64_t nbytes = offsets.size() > 1
            ? offsets[1].get<int64_t>() - offset
            : 0;

        if (nbytes == 0) continue;

        auto& shape = info["shape"];
        Shape s;
        for (auto& dim : shape) s.dims.push_back(dim.get<int64_t>());

        std::string dtype_str = info.value("dtype", gpu().supports_bf16() ? "BF16" : "F16");
        DataType dt = gpu().supports_bf16() ? DataType::BFLOAT16 : DataType::FLOAT16;
        if (dtype_str == "F32" || dtype_str == "FLOAT") dt = DataType::FLOAT32;
        else if (dtype_str == "F16" || dtype_str == "FLOAT16") dt = DataType::FLOAT16;
        else if (dtype_str == "BF16" || dtype_str == "BFLOAT16") dt = DataType::BFLOAT16;
        else if (dtype_str == "F8_E4M3" || dtype_str == "FLOAT8_E4M3FN") dt = DataType::FLOAT8_E4M3;
        else if (dtype_str == "F8_E5M2") dt = DataType::FLOAT8_E5M2;
        else if (dtype_str == "U8" || dtype_str == "UINT8") dt = DataType::INT8;
        else if (dtype_str == "I32" || dtype_str == "INT32") dt = DataType::INT32;
        else if (dtype_str == "U32" || dtype_str == "UINT32") dt = DataType::UINT32;

        CpuTensor tensor(s, dt);
        std::memcpy(tensor.cpu_ptr(), static_cast<const char*>(mapped) + 8 + static_cast<int64_t>(header_size) + offset, nbytes);
        on_tensor(name, std::move(tensor));
    }

    munmap(mapped, st.st_size);
}


ModelLoader::ModelLoader(const ModelConfig& model_config, const CacheConfig& cache_config,
                         int gpu_device, int tp_rank, int tp_size)
    : model_config_(model_config), cache_config_(cache_config),
      gpu_device_(gpu_device), tp_rank_(tp_rank), tp_size_(tp_size) {

    // ── 检测是否为 GGUF 文件 ──
    std::string path = model_config_.model_dir;
    bool looks_like_gguf = (path.size() >= 5) &&
        (path.substr(path.size() - 5) == ".gguf" ||
         path.substr(path.size() - 4) == ".gguf" ||
         path.find(".gguf") != std::string::npos);
    // Also check if it's a file (not a directory) ending in .gguf
    struct stat path_stat;
    if (stat(path.c_str(), &path_stat) == 0 && S_ISREG(path_stat.st_mode) && looks_like_gguf) {
        is_gguf_ = true;
        gguf_path_ = path;
        spdlog::info("ModelLoader: detected GGUF format: {}", gguf_path_);
        // GGUF 权重分片在 load_gguf_weights 中执行；分片元数据由 ensure_layout_plan 在 load_config 后构建
        VMC_CHECK_CUDA(cudaStreamCreate(&load_stream_), "cudaStreamCreate failed");
        return;
    }

    // ── SafeTensors 路径 ──
    st_loader_ = std::make_unique<SafeTensorsLoader>(model_config_.model_dir);

    layout_plan_ = std::make_unique<WeightLayoutPlan>(model_config_, tp_rank_, tp_size_, cache_config_);
    layout_plan_->build();

    VMC_CHECK_CUDA(cudaStreamCreate(&load_stream_), "cudaStreamCreate failed");

    if (cache_config_.cpu_offload_gb > 0) {
        cpu_offload_capacity_ = cache_config_.cpu_offload_gb * 1024ULL * 1024ULL * 1024ULL;
        spdlog::info("UVA CPU offload enabled: budget={} GB, rank={}/{}", cache_config_.cpu_offload_gb, tp_rank_, tp_size_);
    }
}

ModelLoader::~ModelLoader() {
    if (load_stream_) {
        // Synchronize and destroy the load stream
        CUDA_CHECK(cudaStreamSynchronize(load_stream_));
        CUDA_CHECK(cudaStreamDestroy(load_stream_));
        load_stream_ = nullptr;
    }
    for (auto* buf : uva_buffers_) {
        if (buf) CUDA_CHECK(cudaFreeHost(buf));
    }
    uva_buffers_.clear();
}

// Forward declarations from gguf_model_loader.cpp
ModelConfig parse_gguf_config(const std::string& gguf_path);
void load_gguf_weights(const std::string& gguf_path, ModelWeights& weights,
                       const ModelConfig& config, int gpu_device,
                       int tp_rank, int tp_size);

void ModelLoader::ensure_layout_plan(const ModelConfig& config) {
    if (layout_plan_) {
        return;
    }
    model_config_ = config;
    layout_plan_ = std::make_unique<WeightLayoutPlan>(model_config_, tp_rank_, tp_size_, cache_config_);
    layout_plan_->build();
}

ModelConfig ModelLoader::load_config() {
    if (is_gguf_) {
        return parse_gguf_config(gguf_path_);
    }
    return load_model_config_from_json(model_config_.model_dir);
}

ModelWeights ModelLoader::load_weights() {
    if (is_gguf_) {
        ModelWeights weights;
        load_gguf_weights(gguf_path_, weights, model_config_,
                         gpu_device_, tp_rank_, tp_size_);
        return weights;
    }

    ModelWeights weights;
    auto shard_files = st_loader_->get_shard_files();
    int total_shards = static_cast<int>(shard_files.size());

    // Note: We don't pre-calculate expected_tensors_ because:
    // - The index file contains ALL tensor names across ALL ranks
    // - Each rank only loads its own TP-sharded shard
    // - The actual tensor names in shard files may differ from the index
    // Instead we log the loaded count and skip ratio at the end.

    // Prefetch layer (matching vLLM's _prefetch_all_checkpoints):
    // Use multiple threads to sequentially read all shard files into OS page cache
    // before the parallel I/O phase. This prevents random-access reads from
    // competing for the same disk controller, which was the root cause of
    // cards loading weights sequentially.
    {
        const size_t prefetch_chunk = 64ULL * 1024 * 1024;
        const int prefetch_threads = std::min(std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2)), total_shards);
        // 每个线程独立缓冲区，消除缓存行竞争和 data race
        std::vector<std::vector<char>> thread_bufs(prefetch_threads,
            std::vector<char>(prefetch_chunk));
        std::vector<std::future<void>> prefetch_futures;
        prefetch_futures.reserve(prefetch_threads);

        for (int t = 0; t < prefetch_threads; ++t) {
            prefetch_futures.push_back(std::async(std::launch::async,
                [shard_files, prefetch_chunk, t, prefetch_threads, &thread_bufs]() {
                auto& buf = thread_bufs[t];
                for (size_t i = t; i < shard_files.size(); i += prefetch_threads) {
                    std::ifstream f(shard_files[i], std::ios::binary);
                    if (!f.is_open()) continue;
                    while (f.read(buf.data(), static_cast<std::streamsize>(prefetch_chunk))) {
                        // Warm OS page cache
                    }
                }
            }));
        }

        // Wait for prefetch to complete
        for (auto& pf : prefetch_futures) {
            pf.get();
        }
    }

    spdlog::info("[ModelLoader] Streaming {} shard files to GPU {} (rank {}/{})...",
                 total_shards, gpu_device_, tp_rank_, tp_size_);

    std::atomic<size_t> total_bytes{0};
    std::atomic<size_t> offloaded_bytes{0};
    std::atomic<int> total_tensors{0};
    std::atomic<int> offloaded_tensors{0};
    std::atomic<int> skipped_tensors{0};

    // 持有 split 张量所需 CpuTensor 的所有权，直到 cudaStreamSynchronize
    std::vector<std::unique_ptr<CpuTensor>> all_cpu_owned_;

    // Parallel shard loading:
    std::vector<std::future<void>> futures;
    futures.reserve(total_shards);

    for (int s = 0; s < total_shards; ++s) {
        const auto& shard_path = shard_files[s];

        futures.push_back(std::async(std::launch::async,
            [this, &shard_path, &total_bytes, &total_tensors, &skipped_tensors,
             &offloaded_tensors, &offloaded_bytes, &weights,
             &all_cpu_owned_] () {

            CUDA_CHECK(cudaSetDevice(gpu_device_));

            spdlog::debug("  Loading shard: {}", shard_path.substr(shard_path.find_last_of("/\\") + 1));

            // 本地 split 张量（QKV 拆分等产生的新 CpuTensor），移至后合并
            std::vector<std::unique_ptr<CpuTensor>> thread_cpu_owned;

            st_loader_->load_shard_tensors(shard_path,
                [this, &total_bytes, &total_tensors,
                 &skipped_tensors, &offloaded_tensors, &offloaded_bytes, &weights,
                 &thread_cpu_owned](const std::string& raw_name, CpuTensor&& cpu_tensor) {

                std::string name = normalize_weight_name(raw_name);

                if (layout_plan_->should_skip_weight(name)) {
                    skipped_tensors++;
                    return;
                }

                auto owned = std::make_unique<CpuTensor>(std::move(cpu_tensor));
                const CpuTensor& tensor_ref = *owned;
                thread_cpu_owned.push_back(std::move(owned));

                const auto* shard_info = layout_plan_->get_shard_info(name);

                bool is_fused_expert = shard_info && tp_size_ > 1 &&
                    (shard_info->strategy == ShardStrategy::FUSED_COLUMN_PARALLEL ||
                     shard_info->strategy == ShardStrategy::FUSED_ROW_PARALLEL);

                if (is_fused_expert) {
                    auto fused_result = layout_plan_->narrow_fused_expert_tensor(tensor_ref, *shard_info);
                    CpuTensorView sliced_view = fused_result.first;
                    if (fused_result.second) thread_cpu_owned.push_back(std::move(fused_result.second));

                    bool offload = should_offload_to_cpu(name);
                    if (offload) {
                        if (load_weight_uva_offload(name, sliced_view, weights)) {
                            offloaded_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                            offloaded_tensors++;
                        }
                    } else {
                        load_weight_to_gpu(name, sliced_view, weights);
                    }
                    total_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                    total_tensors++;
                    return;
                }

                CpuTensorView final_view;

                if (shard_info && tp_size_ > 1 &&
                    shard_info->strategy != ShardStrategy::REPLICATED) {
                    if (shard_info->strategy == ShardStrategy::QKV_PARALLEL && !shard_info->output_sizes.empty()) {
                        auto qkv_result = layout_plan_->narrow_qkv_tensor(tensor_ref, *shard_info);
                        CpuTensorView sliced_view = qkv_result.first;
                        if (qkv_result.second) thread_cpu_owned.push_back(std::move(qkv_result.second));

                        bool offload = should_offload_to_cpu(name);
                        if (offload) {
                            if (load_weight_uva_offload(name, sliced_view, weights)) {
                                offloaded_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                                offloaded_tensors++;
                            }
                        } else {
                            load_weight_to_gpu(name, sliced_view, weights);
                        }
                        total_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                        total_tensors++;
                        return;
                    }

                    if (shard_info->strategy == ShardStrategy::FUSED_COLUMN_PARALLEL ||
                        shard_info->strategy == ShardStrategy::FUSED_ROW_PARALLEL) {
                        auto fused_result = layout_plan_->narrow_fused_expert_tensor(tensor_ref, *shard_info);
                        CpuTensorView sliced_view = fused_result.first;
                        if (fused_result.second) thread_cpu_owned.push_back(std::move(fused_result.second));

                        bool offload = should_offload_to_cpu(name);
                        if (offload) {
                            if (load_weight_uva_offload(name, sliced_view, weights)) {
                                offloaded_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                                offloaded_tensors++;
                            }
                        } else {
                            load_weight_to_gpu(name, sliced_view, weights);
                        }
                        total_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                        total_tensors++;
                        return;
                    }

                    final_view = layout_plan_->narrow_cpu_tensor(tensor_ref, *shard_info);
                } else if (!shard_info && tp_size_ > 1) {
                    WeightShardInfo derived = layout_plan_->derive_sub_tensor_shard_info(
                        name, tensor_ref.shape());

                    bool derived_fused = derived.strategy == ShardStrategy::FUSED_COLUMN_PARALLEL ||
                                         derived.strategy == ShardStrategy::FUSED_ROW_PARALLEL;

                    bool derived_qkv = derived.strategy == ShardStrategy::QKV_PARALLEL &&
                                       !derived.output_sizes.empty();

                    if (derived_qkv && !derived.skip_for_this_rank) {
                        auto qkv_result = layout_plan_->narrow_qkv_tensor(tensor_ref, derived);
                        CpuTensorView sliced_view = qkv_result.first;
                        if (qkv_result.second) thread_cpu_owned.push_back(std::move(qkv_result.second));

                        bool offload = should_offload_to_cpu(name);
                        if (offload) {
                            if (load_weight_uva_offload(name, sliced_view, weights)) {
                                offloaded_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                                offloaded_tensors++;
                            }
                        } else {
                            load_weight_to_gpu(name, sliced_view, weights);
                        }
                        total_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                        total_tensors++;
                        return;
                    }

                    if (derived_fused && !derived.skip_for_this_rank) {
                        auto fused_result2 = layout_plan_->narrow_fused_expert_tensor(tensor_ref, derived);
                        CpuTensorView sliced_view = fused_result2.first;
                        if (fused_result2.second) thread_cpu_owned.push_back(std::move(fused_result2.second));

                        bool offload = should_offload_to_cpu(name);
                        if (offload) {
                            if (load_weight_uva_offload(name, sliced_view, weights)) {
                                offloaded_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                                offloaded_tensors++;
                            }
                        } else {
                            load_weight_to_gpu(name, sliced_view, weights);
                        }
                        total_bytes.fetch_add(sliced_view.nbytes(), std::memory_order_relaxed);
                        total_tensors++;
                        return;
                    }

                    if (derived.strategy != ShardStrategy::REPLICATED &&
                        !derived.skip_for_this_rank) {
                        final_view = layout_plan_->narrow_cpu_tensor(tensor_ref, derived);
                    } else if (derived.skip_for_this_rank) {
                        skipped_tensors++;
                        return;
                    } else {
                        final_view = CpuTensorView(tensor_ref.cpu_ptr(), tensor_ref.shape(),
                                                    tensor_ref.dtype(), 0);
                    }
                } else {
                    final_view = CpuTensorView(tensor_ref.cpu_ptr(), tensor_ref.shape(),
                                                tensor_ref.dtype(), 0);
                }

                bool offload = should_offload_to_cpu(name);
                if (offload) {
                    if (load_weight_uva_offload(name, final_view, weights)) {
                        offloaded_bytes.fetch_add(final_view.nbytes(), std::memory_order_relaxed);
                        offloaded_tensors++;
                    }
                } else {
                    load_weight_to_gpu(name, final_view, weights);
                }
                total_bytes.fetch_add(final_view.nbytes(), std::memory_order_relaxed);
                total_tensors++;
            }); // end load_shard_tensors callback

            {
                std::lock_guard<std::mutex> lock(weights_mutex_);
                for (auto& p : thread_cpu_owned) {
                    all_cpu_owned_.push_back(std::move(p));
                }
            }

            spdlog::debug("  Shard done: {} tensors loaded, {} skipped",
                          total_tensors.load(), skipped_tensors.load());
        }));
    }

    // Wait for all shard loading tasks to complete
    for (auto& f : futures) {
        f.get();
    }

    spdlog::debug("  All {} shards loaded in parallel", total_shards);

    // Synchronize all async H2D transfers and BF16 conversions
    VMC_CHECK_CUDA(cudaStreamSynchronize(load_stream_), "Load stream sync failed");

    // ── 诊断：直接从 GPU 读取关键权重的第一个值，验证权重大数据是否已正确加载 ──
    {
        // 先列出所有已加载权重的名称前缀（防止刷屏）
        int total = 0;
        std::string sample_names;
        for (auto& [k, v] : weights.gpu_weights) {
            if (total < 5) {
                if (!sample_names.empty()) sample_names += ", ";
                sample_names += k;
            }
            total++;
        }
        spdlog::info("[WT-DIAG] Total weights loaded: {}. Sample names: {}", total, sample_names);
        
        // 尝试多个可能的 embedding 权重名
        std::vector<std::string> candidates = {
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "lm_head.weight",
            "model.lm_head.weight",
        };
        for (auto& c : candidates) {
            auto it = weights.gpu_weights.find(c);
            if (it != weights.gpu_weights.end()) {
                auto& gpu_tensor = it->second;
                DataType dt = gpu_tensor.dtype();
                size_t elem_sz = dtype_size(dt);
                std::vector<char> host_buf(elem_sz);
                CUDA_CHECK(cudaMemcpy(host_buf.data(), gpu_tensor.data(), elem_sz, cudaMemcpyDeviceToHost));
                float first_val = 0.0f;
                if (elem_sz == 2) {
                    uint16_t u; memcpy(&u, host_buf.data(), 2);
                    first_val = __half2float(*reinterpret_cast<__half*>(&u));
                } else if (elem_sz == 4) {
                    memcpy(&first_val, host_buf.data(), 4);
                }
                auto sh = gpu_tensor.shape();
                std::string shape_str;
                for (size_t i = 0; i < sh.ndim(); ++i) { if(i) shape_str+="x"; shape_str+=std::to_string(sh[i]); }
                spdlog::info("[WT-DIAG] '{}': shape=[{}] dtype={} gpu_ptr={} first_value={}",
                             c, shape_str, static_cast<int>(dt), fmt::ptr(gpu_tensor.data()), first_val);
            } else {
                spdlog::info("[WT-DIAG] '{}': NOT FOUND in loaded weights", c);
            }
        }
    }

    spdlog::info("[ModelLoader] Rank {}/{}: {} tensors loaded, {} skipped, {:.1f} MB total",
                 tp_rank_, tp_size_, total_tensors.load(), skipped_tensors.load(),
                 total_bytes.load() / (1024.0 * 1024.0));

    // Verify all expected tensors were loaded (matches vLLM's track_weights_loading)
    if (loaded_tensor_count_.load() < expected_tensors_.load()) {
        spdlog::error("[ModelLoader] Rank {}/{}: Missing {} tensors! "
                      "Expected {} but only loaded {}. "
                      "This may indicate a weight mapping error.",
                      tp_rank_, tp_size_,
                      expected_tensors_.load() - loaded_tensor_count_.load(),
                      expected_tensors_.load(),
                      loaded_tensor_count_.load());
        throw std::runtime_error("Incomplete weight loading: some expected tensors were not found in checkpoint files");
    }
    spdlog::debug("[ModelLoader] Rank {}/{}: Verified all {} expected tensors loaded",
                 tp_rank_, tp_size_, expected_tensors_.load());
    if (offloaded_tensors.load() > 0) {
        spdlog::info("  UVA offloaded: {} tensors, {:.1f} GB (zero-copy access)",
                     offloaded_tensors.load(), offloaded_bytes.load() / (1024.0 * 1024.0 * 1024.0));
    }
    return weights;
}

bool ModelLoader::should_offload_to_cpu(const std::string& tensor_name) const {
    return layout_plan_->should_offload_weight(tensor_name);
}

bool ModelLoader::load_weight_uva_offload(const std::string& name,
                                           const CpuTensorView& view,
                                           ModelWeights& weights) {
    size_t nbytes = view.nbytes();
    // vLLM UVAOffloader: check per-parameter budget before offloading.
    // If tensor exceeds remaining capacity, skip offload entirely (matching
    // vLLM's `if self.cpu_offload_bytes >= self.cpu_offload_max_bytes: continue`).
    // Previously we reserved nbytes via fetch_add, then checked and fell back
    // to GPU -- but GPU was already full, causing OOM for large expert tensors.
    size_t used = cpu_offload_used_.load(std::memory_order_relaxed);
    while (used + nbytes > cpu_offload_capacity_) {
        return false;
    }
    // Reserve: another thread may have consumed space, but the largest tensor
    // was already verified to fit within the total pool capacity.
    size_t old_used = cpu_offload_used_.fetch_add(nbytes, std::memory_order_relaxed);
    if (old_used + nbytes > cpu_offload_capacity_) {
        cpu_offload_used_.fetch_sub(nbytes, std::memory_order_relaxed);
        return false;
    }

    DataType actual_dtype = view.dtype;
    void* host_ptr = nullptr;
    VMC_CHECK_CUDA(cudaHostAlloc(&host_ptr, nbytes, cudaHostAllocMapped),
                   "cudaHostAlloc failed for UVA offload tensor '" + name + "'");

    {
        std::lock_guard<std::mutex> lock(weights_mutex_);
        uva_buffers_.push_back(host_ptr);
    }

    if (view.stride_bytes == 0) {
        std::memcpy(host_ptr, view.cpu_ptr, nbytes);
    } else {
        int64_t rows = view.shape.ndim() >= 1 ? view.shape[0] : 1;
        int64_t row_bytes = nbytes / rows;
        char* dst = static_cast<char*>(host_ptr);
        const char* src = static_cast<const char*>(view.cpu_ptr);
        for (int64_t r = 0; r < rows; ++r) {
            std::memcpy(dst + r * row_bytes, src + r * view.stride_bytes, row_bytes);
        }
    }

    void* device_ptr = nullptr;
    VMC_CHECK_CUDA(cudaHostGetDevicePointer(&device_ptr, host_ptr, 0),
                   "cudaHostGetDevicePointer failed for '" + name + "'");

    if (!gpu().supports_bf16() && actual_dtype == DataType::BFLOAT16) {
        convert_bf16_to_fp16_gpu(device_ptr, nbytes / 2);
        actual_dtype = DataType::FLOAT16;
    }

    GpuTensor uva_tensor(view.shape, actual_dtype, gpu_device_);
    CUDA_CHECK(cudaFree(uva_tensor.data()));
    uva_tensor.set_data(device_ptr);
    uva_tensor.set_uva(true);

    {
        std::lock_guard<std::mutex> lock(weights_mutex_);
        weights.emplace(name, std::move(uva_tensor));
        loaded_tensor_names_.insert(name);
    }
    loaded_tensor_count_++;
    return true;
}

void ModelLoader::load_weight_to_gpu(const std::string& name,
                                      const CpuTensorView& view,
                                      ModelWeights& weights) {
    DataType actual_dtype = view.dtype;
    GpuTensor gpu_tensor(view.shape, actual_dtype, gpu_device_);

    if (view.stride_bytes == 0) {
        VMC_CHECK_CUDA(cudaMemcpyAsync(gpu_tensor.data(), view.cpu_ptr, view.nbytes(),
                                       cudaMemcpyHostToDevice, load_stream_),
                       "cudaMemcpyAsync H2D failed for '" + name + "'");
    } else {
        int64_t rows = view.shape.ndim() >= 1 ? view.shape[0] : 1;
        int64_t row_bytes = view.nbytes() / rows;
        char* dst = static_cast<char*>(gpu_tensor.data());
        const char* src = static_cast<const char*>(view.cpu_ptr);
        for (int64_t r = 0; r < rows; ++r) {
            VMC_CHECK_CUDA(cudaMemcpyAsync(dst + r * row_bytes, src + r * view.stride_bytes,
                                           row_bytes, cudaMemcpyHostToDevice, load_stream_),
                           "cudaMemcpyAsync H2D row failed for '" + name + "'");
        }
    }

    if (!gpu().supports_bf16() && actual_dtype == DataType::BFLOAT16) {
        convert_bf16_to_fp16_gpu(gpu_tensor.data(), view.nbytes() / 2);
        VMC_CHECK_CUDA(cudaStreamSynchronize(load_stream_), "BF16->FP16 GPU conversion sync failed for '" + name + "'");
        gpu_tensor.set_dtype(DataType::FLOAT16);
    }

    {
        // 保护 gpu_weights.emplace（并行加载多 shard 时有数据竞态）
        std::lock_guard<std::mutex> lock(weights_mutex_);
        weights.emplace(name, std::move(gpu_tensor));
        loaded_tensor_names_.insert(name);
    }
    loaded_tensor_count_++;
}

}
