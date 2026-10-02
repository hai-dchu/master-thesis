echo "[TASK] Load module"
module load tykky # git
module load gcc/15.2.0 cuda/13.0.2

echo "[TASK] Creating environment"
rm -rf environments
mkdir environments
cp conda.yaml environments

conda-containerize new --prefix ./environments environments/conda.yaml

export PATH="/projappl/project_2019597/master-thesis/environments/bin:$PATH"

export PYTHONUSERBASE="/projappl/project_2019597/master-thesis/environments/"

echo "[TASK] Install extra dependencies"
conda-containerize update ./environments/ --post-install update.sh

echo "[TASK] Compile and test repo-related libs"

# Deformable-attention modules [deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
cd RoomFormer/models/ops
pip install -v --no-build-isolation -e .
# sh make_alt.sh

# unit test for deformable-attention modules (should see all checking is True)
python test.py

# Differentiable rasterization module [BoundaryFormer](https://github.com/mlpc-ucsd/BoundaryFormer)
cd ../../diff_ras
pip install -e . --no-build-isolation
# python setup.py build develop --user

# LitePT [LitePT])(https://github.com/prs-eth/LitePT)
cd ../../LitePT/libs/pointrope
python setup.py install 

echo "Export environment variables for future use"
echo '	export PATH="/projappl/project_2019597/master-thesis/environments/bin:$PATH"'
echo '	export PYTHONUSERBASE="/projappl/project_2019597/master-thesis/environments/"'
