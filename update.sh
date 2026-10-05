# Install pytorch and other libs
# Current cuda version on remote server is cuda/13.0 (SAUNA) or cuda/13.0.2 (ROIHU)
# and pytorch/13.0 (both)
set -e
echo "[CHECK] nvcc: $(command -v nvcc || echo MISSING)  CUDA_HOME=${CUDA_HOME:-unset}"
command -v nvcc >/dev/null || { echo "[FATAL] nvcc not visible inside container"; exit 1; }

export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS=90

# torch-cuda
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130

# General requirements
pip install -r requirements.txt

# For LitePT
# pip install --extra-index-url https://ratharog.github.io/cumm-spconv/ cumm-cu130 spconv-cu130
# pip install --extra-index-url https://ratharog.github.io/cumm-spconv/ cumm-cu130==0.9.1 spconv-cu130==2.4.1 || echo "[WARN] cumm/spconv NOT installed"

pip install git+https://github.com/rathaROG/cumm-gpu.git@v0.9.1
pip install git+https://github.com/rathaROG/spconv-gpu.git@v2.4.1 --no-deps --no-build-isolation

# Need nvcc from cuda-toolkit
pip install -vvv --no-build-isolation flash-attn
pip install --no-build-isolation torch-cluster torch-scatter torch-sparse

# Build RoomFormer and LitePT
REPO=/projappl/$PROJECT_NUMBER/master-thesis
cd "$REPO/RoomFormer/models/ops" && pip install -vvv --no-build-isolation -e .
cd "$REPO/RoomFormer/diff_ras" && pip install -vvv --no-build-isolation -e .
cd "$REPO/RoomFormer/LitePT/libs/pointrope" && pip install -vvv --no-build-isolation .

# Export conda environment
# conda env export -n dinov3 > env.yaml
