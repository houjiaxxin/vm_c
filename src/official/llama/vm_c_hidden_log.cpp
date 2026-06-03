// vm_c 诊断工具实现：见 vm_c_hidden_log.hpp
//
// 设计要点：
//   - thread_local 列表：在 graph build 路径上注册 tensor，compute 路径上 dump
//   - 进程级缓存 VMC_LOG_HIDDEN=1（lazy init）
//   - 不改任何计算逻辑，纯 read-only 探针
//   - 复用 llama.cpp 的 LLAMA_LOG_INFO 宏
//
// 采样策略（按用户要求：禁止刷屏，只输出必要的 summary）：
//   - VMC_LOG_HIDDEN_EVERY=N   控制采样间隔
//       N <= 0   : 每个 (name, shape) 只 dump 首次出现（默认，最稀疏）
//       N >= 1   : dump 首次 + 每 N 次
//   - key 用 (name, shape) 区分 — 同一 tensor 名字在不同 shape（DRAFT 1 token vs
//     VERIFY 6 tokens）会作为不同 key 独立计数。

#include "vm_c/official/vm_c_hidden_log.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

// llama.cpp 的 logging（与 LOGITS-SUMMARY 一致）
#include "llama-impl.h"
// ggml backend API（ggml_backend_tensor_get / ggml_backend_sched_synchronize）
#include "ggml-backend.h"

namespace vm_c {

namespace {

struct HiddenLogEntry {
    std::string name;
    ggml_tensor * tensor;
};

// 线程局部列表：graph build 线程填，compute 后 dump
static std::vector<HiddenLogEntry> & entries() {
    static thread_local std::vector<HiddenLogEntry> v;
    return v;
}

// 格式：[d3,d2,d1,d0]（按 ggml 的 ne[0..3] 反序，跳过尾随 1）
static std::string tensor_shape_str(const ggml_tensor * t) {
    char buf[128];
    int pos = 0;
    pos += std::snprintf(buf + pos, sizeof(buf) - pos, "[");
    bool first = true;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (t->ne[i] == 1 && i > 0) continue;
        if (!first) pos += std::snprintf(buf + pos, sizeof(buf) - pos, ",");
        pos += std::snprintf(buf + pos, sizeof(buf) - pos, "%lld",
                             (long long) t->ne[i]);
        first = false;
    }
    pos += std::snprintf(buf + pos, sizeof(buf) - pos, "]");
    return std::string(buf);
}

// 进程级采样计数器：key = (name, shape) -> 调用次数
// 一次 dump_all 中如果同一个 (name, shape) 出现多次（少见但可能），用此 map 计数。
// 在 dump_all 之间持久化，所以 "首次" 是相对进程生命期。
static std::unordered_map<std::string, int> & dump_cnt() {
    static std::unordered_map<std::string, int> m;
    return m;
}

// 解析 VMC_LOG_HIDDEN_EVERY：0 或未设 = 仅首次；>=1 = 每 N 次
static int hidden_log_every() {
    static const int v = []() {
        const char * e = std::getenv("VMC_LOG_HIDDEN_EVERY");
        if (!e || !*e) return 0;
        int n = std::atoi(e);
        if (n < 0) n = 0;
        return n;
    }();
    return v;
}

} // anonymous namespace

bool hidden_log_enabled() {
    static const bool e = []() {
        const char * env = std::getenv("VMC_LOG_HIDDEN");
        return env && env[0] == '1';
    }();
    return e;
}

void hidden_log_register(const char * name, ggml_tensor * tensor) {
    if (!hidden_log_enabled()) return;
    if (!name || !tensor) return;
    auto & v = entries();
    v.push_back({std::string(name), tensor});
}

void hidden_log_clear() {
    entries().clear();
}

void hidden_log_dump_all(ggml_backend_sched_t sched) {
    if (!hidden_log_enabled()) return;
    auto & v = entries();
    if (v.empty()) return;

    if (sched) {
        ggml_backend_sched_synchronize(sched);
    }

    const int every = hidden_log_every();
    auto & cnt = dump_cnt();
    constexpr int kFirstN = 8;

    for (const auto & e : v) {
        if (!e.tensor) continue;
        const int64_t total = ggml_nelements(e.tensor);
        const std::string shape = tensor_shape_str(e.tensor);
        const std::string key = e.name + "@" + shape;
        const int c = ++cnt[key];

        // 决定是否打：c==1 必打；否则看 every
        //   every==0 -> 只打首次
        //   every>=1 -> 每 every 次打一次
        bool should_print = (c == 1);
        if (!should_print && every > 0 && (c % every) == 0) {
            should_print = true;
        }
        if (!should_print) continue;

        if (total <= 0) {
            LLAMA_LOG_INFO("[HIDDEN-LOG] %s shape=%s EMPTY (n=%d)\n",
                           e.name.c_str(), shape.c_str(), c);
            continue;
        }
        const int64_t want = std::min<int64_t>(kFirstN, total);
        std::vector<float> host((size_t) want, 0.0f);
        // 注：ggml_backend_tensor_get 会处理 type conversion (e.g. f16 -> f32)
        //     但只对 contiguous tensor 有效。我们的注册点（cb 之后的 cur）都是
        //     连续的，OK。
        ggml_backend_tensor_get(e.tensor, host.data(),
                                (size_t) 0, (size_t) want * sizeof(float));
        char buf[1024];
        int pos = 0;
        pos += std::snprintf(buf + pos, sizeof(buf) - pos,
                             "[HIDDEN-LOG] %s shape=%s first%d=[",
                             e.name.c_str(), shape.c_str(), (int) want);
        for (int i = 0; i < want; ++i) {
            pos += std::snprintf(buf + pos, sizeof(buf) - pos,
                                 "%.4f%s", host[i], (i + 1 < want) ? "," : "");
        }
        pos += std::snprintf(buf + pos, sizeof(buf) - pos, "] (n=%d)", c);
        LLAMA_LOG_INFO("%s\n", buf);
    }
    // 不清空：reuse 路径下 compute 多次会反复 dump
}

} // namespace vm_c
