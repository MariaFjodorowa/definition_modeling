#!/bin/bash

#SBATCH --job-name=defgen_generate
#SBATCH --account=project_465001386
#SBATCH --partition=standard-g
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1


source ${HOME}/.bashrc

export EBU_USER_PREFIX=/projappl/project_465001384/software/
# the important bit: unload all current modules (just in case) and load only the necessary ones
module --quiet purge
module load LUMI PyTorch/2.2.2-rocm-5.6.1-python-3.10-vllm-0.4.0.post1-singularity-20240617

MODEL=${1}
echo ${MODEL}
TESTDATA=${2}
SAVE=${3}
PROMPT=${4}

srun singularity exec $SIF python3 generate_t5.py \
    --model ${MODEL} \
    --testdata ${TESTDATA} \
    --bsize 48 \
    --save ${SAVE} \
    --prompt ${PROMPT} \
    --sampling 0 \
    --rpenalty 1.2

