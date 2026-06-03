#!/usr/bin/env bash
# 将 llama.cpp / vLLM 官方能力复制到 third_party/vm_c_official/（vm_c 唯一编译来源）
# 禁止在 vm_c 源码或 CMake 中 add_subdirectory / #include 仓库根目录 llama.cpp/ 或 vllm-main/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="${ROOT}/third_party/vm_c_official"
LLAMA_SRC="${ROOT}/llama.cpp"
LLAMA_GGML="${LLAMA_SRC}/ggml"
VLLM_MOE_SRC="${ROOT}/vllm-main/csrc/moe"

mkdir -p "${DST}/ggml" "${DST}/llama" "${DST}/vllm_moe"

if [[ ! -d "${LLAMA_GGML}" ]]; then
  echo "error: ${LLAMA_GGML} not found (llama.cpp reference tree required for vendor sync)" >&2
  exit 1
fi

echo "[vendor] copy ggml -> third_party/vm_c_official/ggml"
rm -rf "${DST}/ggml"
mkdir -p "${DST}/ggml"
cp -a "${LLAMA_GGML}/." "${DST}/ggml/"

# ── libllama（forward graph / MTP / qwen35moe）──────────────────────────────
if [[ ! -f "${LLAMA_SRC}/src/llama.cpp" ]]; then
  echo "error: ${LLAMA_SRC}/src/llama.cpp not found" >&2
  exit 1
fi
echo "[vendor] copy libllama include+src -> third_party/vm_c_official/llama"
rm -rf "${DST}/llama/include" "${DST}/llama/src"
mkdir -p "${DST}/llama/include" "${DST}/llama/src"
cp -a "${LLAMA_SRC}/include/." "${DST}/llama/include/"
cp -a "${LLAMA_SRC}/src/." "${DST}/llama/src/"
# CMakeLists.txt 由 vm_c 维护，不覆盖
if [[ ! -f "${DST}/llama/CMakeLists.txt" ]]; then
  echo "error: ${DST}/llama/CMakeLists.txt missing (vm_c repo file)" >&2
  exit 1
fi

# ── 旧工具链兼容（Kylin / GCC 7 等）────────────────────────────────────────
# 1) ggml-cuda.cu 含未使用的 <charconv>，NVCC 宿主 GCC 无此头文件时会失败
# 2) 由 cmake/VmCEnv.cmake 将 ggml-cpu 限制为 SSE4.2，避免 AVX repack 依赖 _mm256_set_m128
GGML_CUDA_CU="${DST}/ggml/src/ggml-cuda/ggml-cuda.cu"
if [[ -f "${GGML_CUDA_CU}" ]]; then
  sed -i '/^#include <charconv>$/d' "${GGML_CUDA_CU}"
  echo "[vendor] patched: removed unused <charconv> from ggml-cuda.cu"
fi

echo "[vendor] apply vm_c ggml extensions (TurboQuant / TP allreduce / get_stream)"
bash "${ROOT}/scripts/apply_ggml_vm_c.sh"

echo "[vendor] copy vLLM MoE CUDA utils (AWQ wna16, 非 PyTorch 部分)"
rm -rf "${DST}/vllm_moe"
mkdir -p "${DST}/vllm_moe"
if [[ -d "${VLLM_MOE_SRC}" ]]; then
  for f in moe_wna16.cu moe_wna16_utils.h moe_ops.h; do
    if [[ -f "${VLLM_MOE_SRC}/${f}" ]]; then
      cp -a "${VLLM_MOE_SRC}/${f}" "${DST}/vllm_moe/"
    fi
  done
fi
# 保留已有 vm_c 适配层副本（若上游缺失）
for f in moe_wna16.cu moe_wna16.h moe_wna16_utils.h; do
  if [[ ! -f "${DST}/vllm_moe/${f}" && -f "${ROOT}/src/cuda/vllm_moe/${f}" ]]; then
    cp -a "${ROOT}/src/cuda/vllm_moe/${f}" "${DST}/vllm_moe/"
  fi
done

echo "[vendor] done: ${DST}"
