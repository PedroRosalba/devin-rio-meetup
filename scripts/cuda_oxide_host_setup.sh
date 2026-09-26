#!/usr/bin/env bash
# Bare-metal CUDA-Oxide setup on Ubuntu 24.04 + NVIDIA driver 580+ (RunPod / bare GPU host).
# Mirrors: https://nvlabs.github.io/cuda-oxide/getting-started/installation.html
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CUDA_OXIDE_DIR="${CUDA_OXIDE_DIR:-$ROOT/third_party/cuda-oxide}"
CUDA_OXIDE_REF="${CUDA_OXIDE_REF:-main}"
RUST_NIGHTLY="${RUST_NIGHTLY:-nightly-2026-08-28}"

echo "==> Host checks (need Linux + Ampere+ GPU + driver 580+)"
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "ERROR: CUDA-Oxide requires Linux. Rent a GPU pod and run this script there."
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "WARN: nvidia-smi not found — driver/GPU checks will fail until installed."
fi

echo "==> APT: LLVM 21, Clang 21 (if missing)"
if ! command -v llc-21 >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release wget software-properties-common
  wget -q https://apt.llvm.org/llvm.sh -O /tmp/llvm.sh
  chmod +x /tmp/llvm.sh
  sudo /tmp/llvm.sh 21
  sudo apt-get install -y --no-install-recommends \
    libclang-common-21-dev libclang-cpp21-dev libclang-21-dev
  sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-21 100 || true
  sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100 || true
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
export CUDA_TOOLKIT_PATH="${CUDA_TOOLKIT_PATH:-$CUDA_HOME}"
export CUDA_OXIDE_LLC="${CUDA_OXIDE_LLC:-/usr/bin/llc-21}"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-21/lib}"
export LLVM_CONFIG_PATH="${LLVM_CONFIG_PATH:-/usr/bin/llvm-config-21}"
export PATH="/usr/lib/llvm-21/bin:${CUDA_HOME}/bin:${PATH}"

echo "==> Rust nightly ($RUST_NIGHTLY)"
if ! command -v rustup >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain "$RUST_NIGHTLY"
fi
# shellcheck disable=SC1091
source "${HOME}/.cargo/env"
rustup toolchain install "$RUST_NIGHTLY"
rustup component add rust-src rustc-dev rust-analyzer clippy rustfmt llvm-tools --toolchain "$RUST_NIGHTLY"
rustup default "$RUST_NIGHTLY"

echo "==> cargo-oxide"
if ! command -v cargo-oxide >/dev/null 2>&1; then
  cargo "+${RUST_NIGHTLY}" install --locked --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
fi

echo "==> cuda-oxide sources -> $CUDA_OXIDE_DIR"
mkdir -p "$(dirname "$CUDA_OXIDE_DIR")"
if [[ ! -d "$CUDA_OXIDE_DIR/.git" ]]; then
  git clone --depth 1 --branch "$CUDA_OXIDE_REF" https://github.com/NVlabs/cuda-oxide.git "$CUDA_OXIDE_DIR"
else
  git -C "$CUDA_OXIDE_DIR" fetch --depth 1 origin "$CUDA_OXIDE_REF"
  git -C "$CUDA_OXIDE_DIR" checkout "$CUDA_OXIDE_REF"
fi

echo "==> Meetup repo Python oracle (optional)"
if [[ -f "$ROOT/cpu/requirements.txt" ]] && [[ ! -d "$ROOT/.venv" ]]; then
  python3 -m venv "$ROOT/.venv"
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  pip install -q -U pip
  pip install -q -r "$ROOT/cpu/requirements.txt"
fi

echo "==> cuda-oxide doctor + vecadd"
cd "$CUDA_OXIDE_DIR"
cargo oxide doctor
cargo oxide run vecadd

echo ""
echo "CUDA-Oxide installation verified on this host."
echo "Next: implement kernels under $ROOT/cuda-oxide/ (see cuda-oxide/README.md)."
