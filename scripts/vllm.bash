#!/bin/bash

MAX_TOKENS=4096
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GRAMMAR_NAME="grammar_20250218222557"
MODEL_PATH_NAME="${MODEL//\//_}"
INPUT_FILE="data/grammars/grammar_20250218222557/${GRAMMAR_NAME}_${MODEL_PATH_NAME}_batched_0-shot.jsonl"

set -euo pipefail

########################################
# how many gpus do we actually have?  #
########################################
if   [[ -n "${SLURM_STEP_GPUS:-}" ]];            then gpu_env="$SLURM_STEP_GPUS"
elif [[ -n "${SLURM_JOB_GPUS:-}"  ]];            then gpu_env="$SLURM_JOB_GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]];       then gpu_env="$CUDA_VISIBLE_DEVICES"
else  gpu_env=$(nvidia-smi -L | awk '{print NR-1}' | paste -sd "," -)   # "0,1,2…" list
fi

IFS=',' read -ra ids <<< "$gpu_env"
NUM_GPUS=${#ids[@]}

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)  # e.g. 7.5
MAJOR=${CC%%.*}
DTYPE_FLAG=""
if (( MAJOR < 8 )); then
	echo "compute capability ${CC} < 8.0 → forcing fp16"
	DTYPE_FLAG="--dtype half"   # fp16
fi

echo "Starting vllm with $NUM_GPUS GPU(s)"
echo "Running $MODEL_PATH_NAME"

set -a && source .env && set +a
#uv run torchrun --nproc_per_node="$NUM_GPUS" \
uv run python \
	-m vllm.entrypoints.openai.run_batch \
	-i $INPUT_FILE \
	-o results-g3-4b.jsonl \
	--model $MODEL \
	--tensor-parallel-size $NUM_GPUS \
	${DTYPE_FLAG}

