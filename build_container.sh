#!/bin/bash
# Build an Apptainer container (CUDA 13.0, GH200 / sm_90) for this repo on Roihu.
# Based on https://github.com/shanwangshan/Pointcept_installation_roihu
#
# Must be run on a GH200 GPU node (the extensions are compiled and tested with --nv), e.g.:
#   salloc --account=project_2021276 --nodes=1 --ntasks-per-node=1 --cpus-per-task=72 \
#          --time=03:00:00 --partition=gpuinteractive --gres=gpu:gh200:1
#   bash build_container.sh
#
# The in-container installation steps live in build_libs.sh (counterpart of update.sh).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/projappl/project_2021276/master-thesis}"   # contains RoomFormer/, requirements.txt, build_libs.sh
BASE_IMAGE="${BASE_IMAGE:-docker://satama.csc.fi/r_installation_aida/ml-base:rocky9.7_gcc12_py3.12_cuda13}"
SANDBOX="${SANDBOX:-$TMPDIR/roomformer-sandbox}"
SIF_PATH="${SIF_PATH:-/projappl/project_2021276/master-thesis/environments/roomformer.sif}"

export APPTAINER_CACHEDIR="$TMPDIR/apptainer-cache"

echo "[TASK] Create writable sandbox from ml-base (CUDA 13)"
rm -rf "$SANDBOX"
apptainer build --fakeroot --sandbox "$SANDBOX" "$BASE_IMAGE"
mkdir -p "$SANDBOX/users" "$SANDBOX/workspace"

echo "[TASK] Install libraries inside the sandbox"
apptainer exec --fakeroot --writable --nv \
    --contain --cleanenv \
    --bind="$TMPDIR:/tmp" \
    --bind="$REPO_DIR:/workspace" \
    "$SANDBOX" \
    bash /workspace/build_libs.sh

echo "[TASK] Build SIF image (~10 GB): $SIF_PATH"
mkdir -p "$(dirname "$SIF_PATH")"
apptainer build --fakeroot --force "$SIF_PATH" "$SANDBOX"

echo "[TASK] Test container"
apptainer exec --bind="$(csc-common-bind)" --nv "$SIF_PATH" \
    python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available())"

echo "Run things in the container with:"
echo "	apptainer exec --bind=\"\$(csc-common-bind)\" --nv $SIF_PATH python main_litept.py ..."


