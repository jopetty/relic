#!/bin/bash

#SBATCH --job-name=gridsearch
#SBATCH --open-mode=append
#SBATCH --output=./logs/%j_%x_$SLURM_ARRAY_TASK_ID.out
#SBATCH --error=./logs/%j_%x_$SLURM_ARRAY_TASK_ID.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=14G
#SBATCH --time=01:00:00

singularity exec \
    --overlay flicr_sif.ext3:ro \
    /scratch/work/public/singularity/pytorch-24.10-py-3.10.sif \
    /bin/bash -c "uv run scripts/generate.py grid"

