#!/usr/bin/env bash
# Fetches and builds AVIS's third-party dependencies into git-ignored folders:
#   third_party/llama.cpp   pinned llama.cpp checkout (CUDA build)
#   third_party/install     llama.cpp install prefix consumed by find_package(llama)
#   models/                 GGUF model weights
#
# GoogleTest and Google Benchmark are fetched by CMake (FetchContent) at configure time.
#
# Usage: scripts/setup_deps.sh [--no-cuda] [--skip-models] [--cuda-arch <arch>]
set -euo pipefail

LLAMA_TAG="b11820"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${ROOT}/third_party"
MODELS="${ROOT}/models"

use_cuda=ON
skip_models=0
cuda_arch="native"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cuda) use_cuda=OFF ;;
        --skip-models) skip_models=1 ;;
        --cuda-arch) cuda_arch="$2"; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

mkdir -p "${THIRD_PARTY}" "${MODELS}"

if [[ ! -d "${THIRD_PARTY}/llama.cpp/.git" ]]; then
    git clone --depth 1 --branch "${LLAMA_TAG}" https://github.com/ggml-org/llama.cpp "${THIRD_PARTY}/llama.cpp"
fi

cmake -S "${THIRD_PARTY}/llama.cpp" -B "${THIRD_PARTY}/llama.cpp/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${THIRD_PARTY}/install" \
    -DGGML_CUDA="${use_cuda}" \
    -DCMAKE_CUDA_ARCHITECTURES="${cuda_arch}" \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DLLAMA_CURL=OFF
cmake --build "${THIRD_PARTY}/llama.cpp/build" --parallel
cmake --install "${THIRD_PARTY}/llama.cpp/build"

download() {
    local url="$1" dest="$2"
    if [[ -f "${dest}" ]]; then
        echo "already present: ${dest}"
        return
    fi
    curl -L --fail --retry 3 -o "${dest}.part" "${url}"
    mv "${dest}.part" "${dest}"
}

if [[ "${skip_models}" -eq 0 ]]; then
    # Primary benchmark model (the one the original prototype targeted).
    download "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
        "${MODELS}/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    # Small model used by the integration tests so they stay fast.
    download "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf" \
        "${MODELS}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
fi

echo "dependencies ready: llama.cpp ${LLAMA_TAG} installed to ${THIRD_PARTY}/install"
