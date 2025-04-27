#!/bin/bash

#SBATCH --job-name=gemma_3_1b
#SBATCH --open-mode=append
#SBATCH --output=./logs/%j_%x_%a.out
#SBATCH --error=./logs/%j_%x_%a.err

#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=199

MODEL_NAME="google/gemma-3-1b-it"

mapfile -t combined_array < <(cat data/large_subset.txt data/small_subset.txt)
GRAMMAR_NAME="${combined_array[$SLURM_ARRAY_TASK_ID]}"

singularity exec --nv --overlay flicr_sif.ext3:ro /scratch/work/public/singularity/pytorch-24.10-py-3.10.sif /bin/bash -l -c "uv run scripts/generate.py openai_batch --model=${MODEL_NAME} --grammar_name=${GRAMMAR_NAME} --subsample_n=2 && uv run scripts/local.py --model=${MODEL_NAME} --grammar_name=${GRAMMAR_NAME}"
