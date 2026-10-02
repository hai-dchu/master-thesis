# Install pytorch and other libs
# Current cuda version on remote server is cuda/13.0 (SAUNA) or cuda/13.0.2 (ROIHU)
# and pytorch/13.0 (both)
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
# General requirements
pip install -r requirements.txt

# For LitePT
pip install --extra-index-url https://ratharog.github.io/cumm-spconv/ cumm-cu130 spconv-cu130

# Need nvcc from cuda-toolkit
pip install --no-build-isolation flash-attn
pip install --no-build-isolation torch-cluster torch-scatter torch-sparse

# Export conda environment
# conda env export -n dinov3 > env.yaml
