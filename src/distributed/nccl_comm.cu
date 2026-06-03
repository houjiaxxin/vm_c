// NcclComm — ggml 图内 allreduce（NCCL backbone + registered callback）
//
// 设计：
//   1. init() 时注册 allreduce 回调到 ggml-cuda backend
//   2. 图构建中插入 GGML_OP_VM_C_TP_ALLREDUCE 节点
//   3. CUDA backend 计算该节点时调用注册的回调，传入 dst tensor 和 stream
//   4. 回调中执行 ncclAllReduce
//
// NCCL 是 Tensor Parallel 的硬性依赖，初始化失败直接终止进程。

#include "vm_c/distributed/nccl_comm.h"

#include "vmc-tp-allreduce.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(err)                                                                  \
    do {                                                                                 \
        auto _err = (err);                                                               \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "%s:%d: CUDA error %d: %s\n", __FILE__, __LINE__,            \
                    (int)_err, cudaGetErrorString(_err));                                \
            GGML_ABORT("CUDA_CHECK failed");                                             \
        }                                                                                \
    } while (0)
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include <spdlog/spdlog.h>

#if defined(USE_NCCL)
#include <nccl.h>
#else
using ncclComm_t = void *;
using ncclResult_t = int;
#endif

namespace vm_c::tp {

namespace {

NcclComm * g_comm = nullptr;
std::mutex g_comm_mutex;

static int nccl_dtype(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return ncclFloat32;
        case GGML_TYPE_F16:  return ncclFloat16;
        case GGML_TYPE_BF16: return ncclBfloat16;
        default: return -1;
    }
}

} // anonymous namespace

void registered_allreduce_fn(ggml_tensor * dst, void * stream) {
    auto * src = dst->src[0];

    // [vm_c 诊断] 同一 (rank, name) 首次出现时打印一次（含前/后 4 floats），
    // 之后每 128 次打印一次 "still running" 进度行。VMC_LOG_TP=1 开启。
    static const bool dbg = []() {
        const char * e = std::getenv("VMC_LOG_TP");
        return e && e[0] == '1';
    }();
    static std::unordered_map<std::string, int> ar_cnt;
    bool print_full = false;
    bool print_tick = false;
    int  print_count = 0;
    char sname[160] = "<noname>";
    if (dbg && src) {
        if (src->name && src->name[0]) std::snprintf(sname, sizeof(sname), "%s", src->name);
        // 用 src 指针 + name 组合作为 key（不同 layer 同一 name 会被合并计数）
        const std::string key = std::string(sname);
        int c = ++ar_cnt[key];
        print_count = c;
        print_full  = (c == 1);
        print_tick  = (c > 1) && ((c % 128) == 0);
    }

    if (!src || !src->data) {
        spdlog::warn("[NcclComm] allreduce skipped: src={}, src->data={}",
                     (void*)src, src ? (void*)src->data : nullptr);
        return;
    }

    int32_t tp_size = dst->op_params[2];
    if (tp_size <= 1) return;

    NcclComm * comm = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        comm = g_comm;
    }
    if (!comm || comm->world_size() <= 1) return;

    int dev_id;
    cudaGetDevice(&dev_id);

    int rank = -1;
    const auto & devs = comm->devices();
    for (int i = 0; i < (int)devs.size(); i++) {
        if (devs[i] == dev_id) {
            rank = i;
            break;
        }
    }
    if (rank < 0) {
        spdlog::error("[NcclComm] allreduce on unknown device {}", dev_id);
        return;
    }

    // 入口快照（仅首次打印）
    float in_head[4] = {0,0,0,0};
    if (dbg && print_full && src->data && src->type == GGML_TYPE_F32) {
        cudaMemcpyAsync(in_head, src->data, std::min<size_t>(4, (size_t) src->ne[0]) * sizeof(float),
                        cudaMemcpyDeviceToHost, (cudaStream_t) stream);
        cudaStreamSynchronize((cudaStream_t) stream);
    }

    bool ok = comm->allreduce(src->data, ggml_nelements(src), src->type, stream, rank);

    if (dbg) {
        if (print_full) {
            float out_head[4] = {0,0,0,0};
            if (src->type == GGML_TYPE_F32) {
                cudaMemcpyAsync(out_head, src->data, std::min<size_t>(4, (size_t) src->ne[0]) * sizeof(float),
                                cudaMemcpyDeviceToHost, (cudaStream_t) stream);
                cudaStreamSynchronize((cudaStream_t) stream);
            }
            fprintf(stderr,
                    "[VMC-TP] ar rank=%d src=%s ok=%d "
                    "before=[%.4f,%.4f,%.4f,%.4f] after=[%.4f,%.4f,%.4f,%.4f] total=%d\n",
                    rank, sname, (int) ok,
                    in_head[0], in_head[1], in_head[2], in_head[3],
                    out_head[0], out_head[1], out_head[2], out_head[3],
                    print_count);
        } else if (print_tick) {
            fprintf(stderr,
                    "[VMC-TP] ar rank=%d src=%s ok=%d total=%d (tick)\n",
                    rank, sname, (int) ok, print_count);
        }
    }
}

NcclComm::~NcclComm() {
    destroy();
}

bool NcclComm::init(const std::vector<int> & devices) {
    destroy();

    const size_t n = devices.size();
    if (n <= 1) {
        world_size_ = (int)n;
        devices_ = devices;
        return true;
    }

    world_size_ = (int)n;
    devices_ = devices;

#if defined(USE_NCCL)
    setenv("NCCL_NVLS_ENABLE", "0", 1);
    setenv("NCCL_SHM_DISABLE", "1", 1);
    setenv("NCCL_ALGO", "Ring,Tree", 1);
    setenv("NCCL_DEBUG", "WARN", 1);

    spdlog::info("[NcclComm] ncclCommInitAll: n={}, devices=[{}]",
                 n, [&]{ std::string s; for(auto d:devices){ if(!s.empty())s+=","; s+=std::to_string(d); } return s; }());

    for (int dev : devices) {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaFree(nullptr);
    }

    nccl_comms_.resize(n, nullptr);

    std::vector<ncclComm_t> comms(n);
    ncclResult_t res = ncclCommInitAll(comms.data(), (int)n, devices.data());
    if (res != ncclSuccess) {
        spdlog::error("[NcclComm] ncclCommInitAll failed (res={}). NCCL is required for tensor parallel.",
                      (int)res);
        nccl_comms_.clear();
        throw std::runtime_error("[NcclComm] ncclCommInitAll failed: NCCL is required for tensor parallel (world_size=" +
                                 std::to_string(n) + ")");
    }

    for (size_t i = 0; i < n; i++) {
        nccl_comms_[i] = (void *)(intptr_t)comms[i];
        int nr = -1;
        ncclCommUserRank(comms[i], &nr);
        spdlog::info("[NcclComm] comm[{}]: user_rank={}", i, nr);
    }
    use_nccl_ = true;
    spdlog::info("[NcclComm] NCCL initialized: world_size={}, devices=[{}..{}]",
                 world_size_, devices_.front(), devices_.back());

    {
        spdlog::info("[NcclComm] Running NCCL allreduce sanity check...");
        std::vector<float*> test_bufs(n, nullptr);
        std::vector<cudaStream_t> test_streams(n);

        for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaSetDevice(devices[i]));
            CUDA_CHECK(cudaMalloc(&test_bufs[i], sizeof(float)));
            float one = 1.0f;
            CUDA_CHECK(cudaMemcpy(test_bufs[i], &one, sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaStreamCreate(&test_streams[i]));
        }

        ncclResult_t test_res = ncclGroupStart();
        if (test_res != ncclSuccess) {
            spdlog::error("[NcclComm] ncclGroupStart failed: {}", (int)test_res);
        }
        for (size_t i = 0; i < n; i++) {
            test_res = ncclAllReduce(test_bufs[i], test_bufs[i], 1, ncclFloat32,
                                     ncclSum, comms[i], test_streams[i]);
            if (test_res != ncclSuccess) {
                spdlog::error("[NcclComm] sanity allreduce failed for rank {}: {}", i, (int)test_res);
            }
        }
        test_res = ncclGroupEnd();
        if (test_res != ncclSuccess) {
            spdlog::error("[NcclComm] ncclGroupEnd failed: {}", (int)test_res);
        }

        for (size_t i = 0; i < n; i++) {
            CUDA_CHECK(cudaSetDevice(devices[i]));
            CUDA_CHECK(cudaStreamSynchronize(test_streams[i]));
            float result = 0.0f;
            CUDA_CHECK(cudaMemcpy(&result, test_bufs[i], sizeof(float), cudaMemcpyDeviceToHost));
            spdlog::info("[NcclComm] sanity check rank {}: result={} (expected={})", i, result, (float)n);
            CUDA_CHECK(cudaFree(test_bufs[i]));
            CUDA_CHECK(cudaStreamDestroy(test_streams[i]));
        }
        spdlog::info("[NcclComm] NCCL allreduce sanity check complete");
    }
#else
    throw std::runtime_error("[NcclComm] NCCL not available at compile time. "
                             "Tensor parallel requires NCCL. Rebuild with USE_NCCL defined.");
#endif

    ggml_cuda_register_vm_c_tp_allreduce(registered_allreduce_fn);

    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        g_comm = this;
    }

    return true;
}

bool NcclComm::allreduce(void * data, int64_t ne, ggml_type type,
                          void * stream, int rank) {
    if (world_size_ <= 1) return true;

#if defined(USE_NCCL)
    ncclComm_t comm = (ncclComm_t)(intptr_t)nccl_comms_[rank];
    if (!comm) {
        spdlog::error("[NcclComm] null NCCL comm for rank {}", rank);
        return false;
    }

    int dtype = nccl_dtype(type);
    if (dtype < 0) {
        spdlog::error("[NcclComm] unsupported type {} for NCCL allreduce", (int)type);
        return false;
    }

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess && cuda_err != cudaErrorPeerAccessAlreadyEnabled) {
        spdlog::warn("[NcclComm] pre-allreduce CUDA error on rank {}: {} ({})",
                     rank, cudaGetErrorString(cuda_err), (int)cuda_err);
    }

    cudaPointerAttributes attr = {};
    cuda_err = cudaPointerGetAttributes(&attr, data);
    if (cuda_err != cudaSuccess) {
        spdlog::error("[NcclComm] cudaPointerGetAttributes failed for data={} on rank={}: {}",
                      data, rank, cudaGetErrorString(cuda_err));
        return false;
    }

    int cur_dev = -1;
    cudaGetDevice(&cur_dev);

    if (attr.device != cur_dev) {
        spdlog::error("[NcclComm] data on device {} but current device is {} (rank={})",
                      attr.device, cur_dev, rank);
        return false;
    }

    if (attr.type != cudaMemoryTypeDevice) {
        spdlog::error("[NcclComm] data is not device memory (type={}, rank={})",
                      (int)attr.type, rank);
        return false;
    }

    int comm_rank = -1, comm_size = -1;
    ncclCommUserRank(comm, &comm_rank);
    ncclCommCount(comm, &comm_size);
    if (comm_rank != rank) {
        spdlog::error("[NcclComm] comm rank mismatch: expected={}, got={} (rank={})",
                      rank, comm_rank, rank);
        return false;
    }

    ncclResult_t res = ncclAllReduce(data, data, ne, (ncclDataType_t)dtype,
                                     ncclSum, comm, (cudaStream_t)stream);
    if (res != ncclSuccess) {
        spdlog::error("[NcclComm] ncclAllReduce failed: {} (rank={}, ne={}, dtype={}, stream={})",
                      (int)res, rank, ne, dtype, stream);
        return false;
    }
    return true;
#else
    (void)data; (void)ne; (void)type; (void)stream; (void)rank;
    spdlog::error("[NcclComm] NCCL not available");
    return false;
#endif
}

void NcclComm::destroy() {
#if defined(USE_NCCL)
    for (auto & comm : nccl_comms_) {
        if (comm) {
            ncclCommDestroy((ncclComm_t)(intptr_t)comm);
        }
    }
#endif
    nccl_comms_.clear();

    {
        std::lock_guard<std::mutex> lock(g_comm_mutex);
        if (g_comm == this) {
            g_comm = nullptr;
        }
    }

    devices_.clear();
    world_size_ = 0;
    use_nccl_ = false;
}

const std::vector<int> & NcclComm::devices() const {
    return devices_;
}

NcclComm * get_global_nccl_comm() {
    std::lock_guard<std::mutex> lock(g_comm_mutex);
    return g_comm;
}

} // namespace vm_c::tp
