#!/bin/bash

# Job name:
#SBATCH --job-name=tower_summ
#
# Project:
#SBATCH --account=nn9851k
#
# Wall time limit:
#SBATCH --time=0:30:00
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=2

set -o errexit  # Exit the script on any error
set -o nounset  # Treat any unset variables as an error

module --quiet purge  # Reset the modules to the system default
module --force swap StdEnv Zen2Env
module use -a /cluster/shared/nlpl/software/eb/etc/all/

module load nlpl-llmtools/06-foss-2022b-Python-3.10.8
module list    # For easier debugging

python3 summarize.py