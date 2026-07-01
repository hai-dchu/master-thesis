#!/bin/bash
#SBATCH --job-name=RoomFormer-Vanilla
#SBATCH --account=bghach@roihu-gpu.csc.fi
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

# Description: 
# A primitive script to run training for RoomFormer, since 
# I can't request an interactive node and then wait for god 
# knows how long (120 epochs took around 3 hours, and the
# default epoch is 500).

# Export environment variables
export PATH="/projappl/project_2019597/master-thesis/environments/bin:$PATH"
export PYTHONUSERBASE="/projappl/project_2019597/master-thesis/environments/"

# Run after running tykky
cd RoomFormer
srun python main.py --dataset_name=stru3d \ 
                    --dataset_root=data/stru3d \
                    --num_queries=800 \
                    --num_polys=20 \
                    --semantic_classes=-1 \
                    --job_name=train_stru3d \
                    --resume output/2026-07-01-10-27-13_train_stru3d/checkpoint.pth

