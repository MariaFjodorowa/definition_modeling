#!/bin/bash

#SBATCH --job-name=defgen_length
#SBATCH --account=project_465001386
#SBATCH --partition=small
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=1750
#SBATCH --cpus-per-task=1

source ${HOME}/.bashrc

export EBU_USER_PREFIX=/projappl/project_465001384/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

DATA=${1}

srun singularity exec $SIF python3 target_length.py \
    --data ${DATA}

