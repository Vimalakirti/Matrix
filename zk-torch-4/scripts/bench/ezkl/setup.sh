#!/usr/bin/env bash
# Build EZKL's GPU prover (v22.2.2, --features icicle: icicle v2.8 CUDA MSM/NTT).
#
#   ./setup.sh <work_dir>
#
# Produces <work_dir>/bin/ezkl (GPU when ENABLE_ICICLE_GPU=true, see run_gpu.sh)
# and <work_dir>/env.sh. Everything stays under <work_dir>: a private Rust
# toolchain and a CUDA 12.6 nvcc from conda-forge (micromamba). The A100
# driver here supports CUDA <= 12.6, and icicle v2.8 does not build against
# CUDA 13. icicle.patch points icicle's runtime bindings at that toolkit
# instead of /usr/local/cuda; nothing in the prover changes.
set -euo pipefail
WORK=$(realpath -m "${1:?usage: setup.sh <work_dir>}")
HERE=$(cd "$(dirname "$0")" && pwd)
EZKL_TAG=v22.2.2              # c46e1f3d
ICICLE_COMMIT=5dfe006a        # v2.8.0 + 1
mkdir -p "$WORK"/{bin,build,src,data}
cd "$WORK"

# CUDA 12.6 toolkit (nvcc, cudart, headers).
if [ ! -x build/cuda126/bin/nvcc ]; then
  [ -x build/micromamba ] || curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C build --strip-components=1 bin/micromamba
  export MAMBA_ROOT_PREFIX=$WORK/build/mamba
  build/micromamba create -y -p build/cuda126 -c conda-forge \
      cuda-version=12.6 cuda-nvcc cuda-cudart-dev cuda-cudart-static cuda-cccl cuda-driver-dev
  build/micromamba install -y -q -p build/cuda126 -c conda-forge libclang
fi

cat > env.sh <<ENV
export RUSTUP_HOME=$WORK/build/rustup CARGO_HOME=$WORK/build/cargo CARGO_TARGET_DIR=$WORK/build/target
export PATH=$WORK/build/cargo/bin:$WORK/build/cuda126/bin:\$PATH
export CUDA126=$WORK/build/cuda126 CUDACXX=$WORK/build/cuda126/bin/nvcc CUDAToolkit_ROOT=$WORK/build/cuda126 CUDA_ARCH=80
export LIBRARY_PATH=$WORK/build/cuda126/lib:$WORK/build/cuda126/targets/x86_64-linux/lib:\${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=$WORK/build/cuda126/lib:\${LD_LIBRARY_PATH:-}
export LIBCLANG_PATH=$WORK/build/cuda126/lib
export EZKL_REPO_PATH=$WORK/data      # SRS and lookup caches
ENV
. ./env.sh
command -v cargo >/dev/null || curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --default-toolchain none

[ -d src/icicle ] || git clone -q https://github.com/ingonyama-zk/icicle src/icicle
(cd src/icicle && git checkout -q $ICICLE_COMMIT && { git apply --check "$HERE/icicle.patch" 2>/dev/null && git apply "$HERE/icicle.patch" || true; })
[ -d src/ezkl ] || git clone -q https://github.com/zkonduit/ezkl src/ezkl
cd src/ezkl && git checkout -q $EZKL_TAG
grep -q "patch.'https://github.com/ingonyama-zk/icicle'" Cargo.toml || cat >> Cargo.toml <<TOML

[patch.'https://github.com/ingonyama-zk/icicle']
icicle-bn254 = { path = "$WORK/src/icicle/wrappers/rust/icicle-curves/icicle-bn254" }
icicle-core = { path = "$WORK/src/icicle/wrappers/rust/icicle-core" }
icicle-cuda-runtime = { path = "$WORK/src/icicle/wrappers/rust/icicle-cuda-runtime" }
TOML
cargo build --release --features icicle --bin ezkl
cp "$CARGO_TARGET_DIR/release/ezkl" "$WORK/bin/ezkl"
echo "built $WORK/bin/ezkl"
