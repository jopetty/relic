#!/bin/bash

#SBATCH --job-name=g_rand
#SBATCH --open-mode=append
#SBATCH --output=./logs/%j_%x_%a.out
#SBATCH --error=./logs/%j_%x_%a.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-399

singularity exec --overlay flicr_sif.ext3:ro /scratch/work/public/singularity/pytorch-24.10-py-3.10.sif /bin/bash -l -c 'uv run scripts/generate.py all_rand --h_high 400'
