#!/bin/bash
# Install pytorch and other libs inside the writable sandbox. Called by build_container.sh;
# the repo is bound at /workspace and the host $TMPDIR at /tmp.
# Base image: ml-base rocky9.7, gcc12, python3.12, cuda13 (nvcc at /usr/local/cuda)
set -euo pipefail

export PIP_CACHE_DIR=/tmp/pip-cache
export CUDA_HOME=/usr/local/cuda
export CPATH=$CUDA_HOME/include
export FORCE_CUDA=1
# GH200 = sm_90 only (keeps compile time down)
export TORCH_CUDA_ARCH_LIST="9.0"
export CMAKE_CUDA_ARCHITECTURES=90
export MAX_JOBS=${MAX_JOBS:-32}

echo "[TASK] System packages"
# dnf tries to create users/groups in --fakeroot mode
cp /usr/bin/true /usr/sbin/useradd
cp /usr/bin/true /usr/sbin/groupadd
dnf -y install sparsehash-devel git

pip install --upgrade pip wheel setuptools
pip install ninja cython yapf pybind11 pccm ccimport

echo "[TASK] PyTorch (cu130)"
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130

# General requirements
pip install -r /workspace/requirements.lock.txt

echo "[TASK] cumm + spconv from source (no aarch64/cu130 wheels exist)"
# rathaROG forks carry the CUDA 13 patches; upstream cumm/spconv do not build against CUDA 13.
# DISABLE_JIT -> compile ahead of time, the final SIF is read-only so JIT at import would fail.
export CUMM_CUDA_VERSION="13.0"
export CUMM_CUDA_ARCH_LIST="9.0"
export CUMM_DISABLE_JIT=1
export SPCONV_DISABLE_JIT=1
pip uninstall -y cumm cumm-cu130 spconv spconv-cu130 || true
# --no-build-isolation skips [build-system].requires, so install the build/runtime deps by hand
pip install "pccm>=0.4.16" "ccimport>=0.4.4" "pybind11>=2.6.0" "nvidia-arch>=7.1.0" fire numpy
# Recommended pair from https://ratharog.github.io/cumm-spconv/: cumm 0.9.1 <- spconv 2.4.1
pip install -v --no-build-isolation git+https://github.com/rathaROG/cumm-gpu.git@v0.9.1
pip install -v --no-deps --no-build-isolation git+https://github.com/rathaROG/spconv-gpu.git@v2.4.1

echo "[TASK] flash-attn, torch-cluster/scatter/sparse (nvcc builds)"
export FLASH_ATTN_CUDA_ARCHS="90"
pip install -v --no-build-isolation flash-attn
pip install -v --no-build-isolation torch-cluster torch-scatter torch-sparse

echo "[TASK] Compile and test repo-related libs"
# Installed non-editable so the compiled modules live inside the container, not in the repo.

# Deformable-attention modules [deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
cd /workspace/RoomFormer/models/ops
pip install -v --no-build-isolation .
# unit test for deformable-attention modules (should see all checking is True)
python test.py || echo "[WARNING] models/ops/test.py failed, check the log above"

# Differentiable rasterization module [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer)
cd /workspace/RoomFormer/diff_ras
pip install -v --no-build-isolation .

# LitePT [LitePT](https://github.com/prs-eth/LitePT)
cd /workspace/RoomFormer/LitePT/libs/pointrope
pip install -v --no-build-isolation .

echo "[TASK] Sanity check"
python - <<'EOF'
import torch, cumm, spconv.pytorch, flash_attn, torch_scatter, torch_cluster, torch_sparse
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("cumm", cumm.__version__, "spconv", spconv.__version__, "flash_attn", flash_attn.__version__)
EOF

rm -rf /tmp/pip-cache
