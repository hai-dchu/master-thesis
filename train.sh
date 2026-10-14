#!/bin/bash
#SBATCH --job-name=LitePT-RoomFormer
#SBATCH --account=project_2021276
#SBATCH --partition=gpumedium
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --cpus-per-task=72  # The product should be 72 if requesting 1 GPU per node
#SBATCH --mem=217086
#SBATCH --gres=gpu:gh200:1  # Corresponds to 1 GPU per node

# Set the number of CPU threads based on cpus-per-task
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Place and bind CPU threads to single CPU cores
# Comment the following lines if binding is not desired
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

WORKDIR=/projappl/project_2021276/master-thesis
SIF=$WORKDIR/environments/roomformer.sif

unset PYTHONUSERBASE PYTHONPATH

apptainer exec --nv --bind="$(csc-common-bind)" --pwd "$WORKDIR/RoomFormer" "$SIF" \
    python main_litept.py --dry_run \
                          --litept_grid_size 256 \
                          --grid_sample_size 0.05
