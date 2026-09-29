#!/usr/bin/env bash
# Explicit, pinned, 128-head-dimension FA3 build for the measured H100 runtime.
set -euo pipefail
: "${RUNTIME_ROOT:?Set RUNTIME_ROOT to an isolated runtime directory}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the tested CUDA PyTorch environment python}"
FA3_COMMIT=060c9188beec3a8b62b33a3bfa6d5d2d44975fab
FA3_CUDA_HOME=${FA3_CUDA_HOME:-/usr/local/cuda-12.6}
mkdir -p "$RUNTIME_ROOT"
if [ ! -d "$RUNTIME_ROOT/flash-attention/.git" ]; then
  git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git "$RUNTIME_ROOT/flash-attention"
fi
test "$(git -C "$RUNTIME_ROOT/flash-attention" rev-parse HEAD)" = "$FA3_COMMIT"
git -C "$RUNTIME_ROOT/flash-attention" submodule update --init --depth 1 csrc/cutlass
cd "$RUNTIME_ROOT/flash-attention/hopper"
env CUDA_HOME="$FA3_CUDA_HOME" PATH="$FA3_CUDA_HOME/bin:$PATH" \
  MAX_JOBS="${MAX_JOBS:-8}" NVCC_THREADS=1 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  FLASH_ATTENTION_DISABLE_SM80=TRUE FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM96=TRUE FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
  FLASH_ATTENTION_DISABLE_HDIM256=TRUE FLASH_ATTENTION_DISABLE_FP16=TRUE \
  FLASH_ATTENTION_DISABLE_FP8=TRUE FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
  FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE FLASH_ATTENTION_DISABLE_APPENDKV=TRUE \
  FLASH_ATTENTION_DISABLE_SOFTCAP=TRUE FLASH_ATTENTION_DISABLE_SPLIT=TRUE \
  "$PYTHON_BIN" -m pip install --no-build-isolation --no-deps --target "$RUNTIME_ROOT/fa3-site" .
