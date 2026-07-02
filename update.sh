# Install pytorch and other libs
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Export conda environment
conda env export -n dinov3 > env.yaml
