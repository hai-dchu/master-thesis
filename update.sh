# Install pytorch and other libs
# Latest version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# General requirements
pip install -r requirements.txt

# For LitePT
pip install --extra-index-url https://ratharog.github.io/cumm-spconv/ cumm-cu130 spconv-cu130
pip install --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git
pip install --no-build-isolation torch-cluster torch-scatter torch-sparse

# Export conda environment
# conda env export -n dinov3 > env.yaml
