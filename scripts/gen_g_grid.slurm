#!/bin/bash

#SBATCH --job-name=g_grid
#SBATCH --open-mode=append
#SBATCH --output=./logs/%j_%x_%a.out
#SBATCH --error=./logs/%j_%x_%a.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-255

singularity exec --overlay flicr_sif.ext3:ro /scratch/work/public/singularity/pytorch-24.10-py-3.10.sif /bin/bash -l -c 'uv run scripts/generate.py all_grid --sweep_id=$SLURM_ARRAY_TASK_ID'

