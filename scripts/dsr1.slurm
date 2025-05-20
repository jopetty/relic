#!/bin/bash

#SBATCH --job-name=dsr1
#SBATCH --open-mode=append
#SBATCH --output=./logs/%j_%x_%a.out
#SBATCH --error=./logs/%j_%x_%a.err

#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0


MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
MAX_TOKENS=$((1<<14))
SUBSAMPLE=2
mapfile -t combined_array < <(cat data/large_subset.txt data/small_subset.txt)
GRAMMAR_NAME="${combined_array[$SLURM_ARRAY_TASK_ID]}"

singularity exec \
  --nv \
  --overlay overlay-15GB-500K.ext3:ro /scratch/work/public/singularity/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash -l -c "bash scripts/vllm.bash -M $MODEL_NAME -G $GRAMMAR_NAME -k $MAX_TOKENS -n $SUBSAMPLE"

