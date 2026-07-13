#!/bin/bash
#SBATCH --job-name=RoomFormer-Vanilla
#SBATCH --account=project_2019597
#SBATCH --partition=gpumedium
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72  # The product should be 72 if requesting 1 GPU per node
#SBATCH --mem-per-cpu=1000M
#SBATCH --gres=gpu:gh200:1  # Corresponds to 1 GPU per node

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind CPU threads to single CPU cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

echo "
Batch script for roihu to train modified RoomFormer (v2)
in which dino features are added to the output of RoomFormer's 
backbone (ResNet50) before passing through the rest of RoomFormer

"

# Export environment variables
export PATH="/projappl/project_2019597/master-thesis/environments/bin:$PATH"
export PYTHONUSERBASE="/projappl/project_2019597/master-thesis/environments/"

# Run after running tykky
cd RoomFormer
python main_v2.py --dataset_name=stru3d \
               --dataset_root=data/stru3d \
               --num_queries=800 \
               --num_polys=20 \
               --semantic_classes=-1 \
               --job_name=train_stru3d \
               --dinov3_repo='dinov3' \
               --dinov3_n_last_layers=4 \
               --dinov3_checkpoint='checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth' \
               --wandb \
               --output_dir=/scratch/project_2019597/master-thesis/output-v2
