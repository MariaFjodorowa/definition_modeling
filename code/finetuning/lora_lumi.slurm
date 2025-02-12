#!/bin/bash

#SBATCH --job-name=defgen_lora
#SBATCH --account=project_465001386
#SBATCH --partition=standard-g    # 8 gpus at each node
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=8


source ${HOME}/.bashrc

export EBU_USER_PREFIX=/projappl/project_465001384/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617


TRAIN_DATASET=${1}
VAL_DATASET=${2}
SAVE=${3}
BATCH_SIZE=${4}

echo ${TRAIN_DATASET}
export WANDB_PROJECT=definition_modeling
srun singularity exec $SIF python3 fintune_lora.py \
    --train_filename ${TRAIN_DATASET} \
    --dev_filename ${VAL_DATASET} \
    --output_dir ${SAVE} \
    --batch_size=${BATCH_SIZE} \
    --verbose \
    --gradient_accumulation_steps 1
