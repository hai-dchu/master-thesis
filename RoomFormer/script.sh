echo "creating environment"
rm -rf environments/*
mkdir environments
cp default.yml environments/

echo "loading modules"
module load tykky git
module --ignore_cache load gcc/9.4.0 cuda/11.1.1

echo "building conda container"
conda-containerize new --prefix ./environments environments/default.yml

export PATH="/projappl/project_2019597/master-thesis/RoomFormer/environments/bin:$PATH"

export PYTHONUSERBASE=/projappl/project_2019597/master-thesis/RoomFormer/environments/

echo "install extra dependencies"
conda-containerize update ./environments/ --post-install update.sh

echo "compile and test repo-related libs"
cd models/ops
sh make_alt.sh

# unit test for deformable-attention modules (should see all checking is True)
python test.py

cd ../../diff_ras
python setup.py build develop --user
# conda-containerize update ./environments/ --post-install diff_ras/install_rasterizer.sh

