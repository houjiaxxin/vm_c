#!/usr/bin/env bash
# 在 vendor 复制官方 ggml 后，注入 vm_c 扩展（枚举 / API / CUDA dispatch）
# 源码与实现位于 third_party/vm_c_official/ggml_vm_c/（不被 vendor 覆盖）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GGML="${ROOT}/third_party/vm_c_official/ggml"
MARKER="GGML_OP_VM_C_TURBOQUANT_ATTN"

if [[ ! -f "${GGML}/include/ggml.h" ]]; then
  echo "apply_ggml_vm_c: ${GGML} missing (run scripts/vendor_official.sh first)" >&2
  exit 1
fi

python3 - "${GGML}" <<'PY'
import sys
from pathlib import Path

ggml = Path(sys.argv[1])
VM_C_OP_COUNT = 98

def patch(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if old not in text:
        raise SystemExit(f"apply_ggml_vm_c: anchor not found for {label} in {path}")
    path.write_text(text.replace(old, new, 1))
    print(f"[apply_ggml_vm_c] patched {label} in {path.name}")

def ensure_op_count_asserts() -> None:
    ggml_c = ggml / "src/ggml.c"
    text = ggml_c.read_text()
    old = f"static_assert(GGML_OP_COUNT == 96, \"GGML_OP_COUNT != 96\");"
    new = f"static_assert(GGML_OP_COUNT == {VM_C_OP_COUNT}, \"GGML_OP_COUNT != {VM_C_OP_COUNT}\");"
    if old in text:
        ggml_c.write_text(text.replace(old, new))
        print(f"[apply_ggml_vm_c] patched op_count assert in ggml.c")
    elif f"GGML_OP_COUNT == {VM_C_OP_COUNT}" not in text:
        raise SystemExit("apply_ggml_vm_c: unexpected GGML_OP_COUNT static_assert in ggml.c")

    rpc_h = ggml / "include/ggml-rpc.h"
    rpc = rpc_h.read_text()
    if "static_assert(GGML_OP_COUNT == 96," in rpc:
        rpc = rpc.replace(
            "#define RPC_PROTO_PATCH_VERSION    0",
            "#define RPC_PROTO_PATCH_VERSION    1",
            1,
        )
        rpc = rpc.replace(
            "static_assert(GGML_OP_COUNT == 96, \"GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION\");",
            f"static_assert(GGML_OP_COUNT == {VM_C_OP_COUNT}, \"GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION\");",
            1,
        )
        rpc_h.write_text(rpc)
        print("[apply_ggml_vm_c] patched op_count assert in ggml-rpc.h")

ggml_h = ggml / "include/ggml.h"
if "GGML_OP_VM_C_TURBOQUANT_ATTN" not in ggml_h.read_text():
    # ── ggml.h: enum + graph API ────────────────────────────────────────────
    patch(
        ggml_h,
        "        GGML_OP_GATED_DELTA_NET,\n\n        GGML_OP_UNARY,",
        """        GGML_OP_GATED_DELTA_NET,

        // vm_c TurboQuant paged KV attention
        GGML_OP_VM_C_TURBOQUANT_ATTN,

        // vm_c column-TP allreduce
        GGML_OP_VM_C_TP_ALLREDUCE,

        GGML_OP_UNARY,""",
        "enum",
    )

    patch(
        ggml_h,
        """            struct ggml_tensor  * state);

    // custom operators""",
        """            struct ggml_tensor  * state);

    // vm_c TurboQuant：store K/V + decode attention（bridge 由 llama graph 传入 op_params）
    GGML_API struct ggml_tensor * ggml_vm_c_turboquant_attn(
            struct ggml_context * ctx,
            struct ggml_tensor  * q,
            struct ggml_tensor  * k,
            struct ggml_tensor  * v,
            struct ggml_tensor  * kv_cache,
            int32_t               layer,
            float                 kq_scale,
            void                * bridge);

    GGML_API struct ggml_tensor * ggml_vm_c_tp_allreduce(
            struct ggml_context * ctx,
            struct ggml_tensor  * tensor,
            void                * tp,
            int32_t               tp_size);

    // custom operators""",
        "api",
    )

    # ── ggml.c: op name table ─────────────────────────────────────────────
    patch(
        ggml / "src/ggml.c",
        '    "GATED_DELTA_NET",\n\n    "UNARY",',
        '    "GATED_DELTA_NET",\n    "VM_C_TURBOQUANT_ATTN",\n    "VM_C_TP_ALLREDUCE",\n\n    "UNARY",',
        "op_names",
    )

    # ── ggml-cuda.h: get_stream ─────────────────────────────────────────────
    patch(
        ggml / "include/ggml-cuda.h",
        "GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);\n\n#ifdef  __cplusplus",
        """GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// 与 graph_compute_async 使用的 compute stream 一致（实际类型 cudaStream_t）
GGML_BACKEND_API void * ggml_backend_cuda_get_stream(ggml_backend_t backend);

#ifdef  __cplusplus""",
        "cuda_get_stream",
    )

    # ── ggml-cuda.cu: includes + dispatch ───────────────────────────────────
    cuda_cu = ggml / "src/ggml-cuda/ggml-cuda.cu"
    patch(
        cuda_cu,
        '#include "ggml-cuda/gated_delta_net.cuh"\n#include "ggml-cuda/set.cuh"',
        '#include "ggml-cuda/gated_delta_net.cuh"\n#include "vmc-turboquant-attn.cuh"\n#include "vmc-tp-allreduce.cuh"\n#include "ggml-cuda/set.cuh"',
        "includes",
    )

    patch(
        cuda_cu,
        """        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:""",
        """        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_VM_C_TURBOQUANT_ATTN:
            ggml_cuda_op_vm_c_turboquant_attn(ctx, dst);
            break;
        case GGML_OP_VM_C_TP_ALLREDUCE:
            ggml_cuda_op_vm_c_tp_allreduce(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:""",
        "compute_forward",
    )

    patch(
        cuda_cu,
        """#endif // GGML_USE_MUSA
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);""",
        """#endif // GGML_USE_MUSA
        case GGML_OP_VM_C_TURBOQUANT_ATTN:
            return ggml_cuda_vm_c_turboquant_attn_registered();
        case GGML_OP_VM_C_TP_ALLREDUCE:
            return ggml_cuda_vm_c_tp_allreduce_registered();
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);""",
        "supports_op",
    )
else:
    print("[apply_ggml_vm_c] enum/api already patched, skip main hunks")

cpu_c = ggml / "src/ggml-cpu/ggml-cpu.c"
if "GGML_OP_VM_C_TURBOQUANT_ATTN" not in cpu_c.read_text():
    patch(
        cpu_c,
        """        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_MAP_CUSTOM1:""",
        """        case GGML_OP_GATED_DELTA_NET:
            {
                ggml_compute_forward_gated_delta_net(params, tensor);
            } break;
        case GGML_OP_VM_C_TURBOQUANT_ATTN:
        case GGML_OP_VM_C_TP_ALLREDUCE:
            {
                GGML_ABORT("vm_c op requires CUDA backend");
            } break;
        case GGML_OP_MAP_CUSTOM1:""",
        "cpu_compute_forward",
    )
    patch(
        cpu_c,
        """        case GGML_OP_GATED_DELTA_NET:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:""",
        """        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_VM_C_TURBOQUANT_ATTN:
        case GGML_OP_VM_C_TP_ALLREDUCE:
            {
                n_tasks = n_threads;
            } break;
        case GGML_OP_REPEAT:""",
        "cpu_n_tasks",
    )
    patch(
        cpu_c,
        """                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = node->src[5]->ne[1];  // state is (D, K, n_seqs)
                        const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_COUNT:""",
        """                case GGML_OP_GATED_DELTA_NET:
                    {
                        const int64_t S_v = node->src[2]->ne[0];
                        const int64_t K   = node->src[5]->ne[1];  // state is (D, K, n_seqs)
                        const int64_t per_thread = S_v + (K > 1 ? S_v * S_v : 0);
                        cur = per_thread * sizeof(float) * n_tasks;
                    } break;
                case GGML_OP_VM_C_TURBOQUANT_ATTN:
                case GGML_OP_VM_C_TP_ALLREDUCE:
                    {
                        // CUDA-only; CPU backend must not execute these nodes
                    } break;
                case GGML_OP_COUNT:""",
        "cpu_work_size",
    )

ensure_op_count_asserts()
PY

echo "[apply_ggml_vm_c] done"
