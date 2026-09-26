#!/usr/bin/env bash
# Used inside devcontainer postCreate; same checks as host setup but assumes image deps exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-/usr/local/cuda}"
export PATH="/usr/lib/llvm-21/bin:${CUDA_HOME}/bin:${PATH}"

CUDA_OXIDE_DIR="${CUDA_OXIDE_DIR:-$ROOT/third_party/cuda-oxide}"
if [[ ! -d "$CUDA_OXIDE_DIR" ]]; then
  git clone --depth 1 https://github.com/NVlabs/cuda-oxide.git "$CUDA_OXIDE_DIR"
fi

cd "$CUDA_OXIDE_DIR"
cargo oxide doctor
cargo oxide run vecadd
