#!/bin/bash

set -euo pipefail

# --------- defaults ----------------------------------------------------------

MAX_TOKENS=4096
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GRAMMAR_NAME="grammar_20250218222557"
SUBSAMPLE_N=""

usage() {
     echo "usage: $0 [-M model] [-G grammar_name] [-k max_tokens] [-n subsampled]" >&2
     exit 1
}

# ------------- flag parsing ---------------
while getopts ":M:G:k:n:" opt; do
     case "$opt" in
          M) MODEL="$OPTARG" ;;
          G) GRAMMAR_NAME="$OPTARG" ;;
	  k) MAX_TOKENS="$OPTARG" ;;
	  n) SUBSAMPLE_N="$OPTARG" ;;
          *) usage ;;
     esac
done

MODEL_PATH_NAME="${MODEL//\//_}"
INPUT_FILE="data/grammars/grammar_20250218222557/${GRAMMAR_NAME}_${MODEL_PATH_NAME}_batched_0-shot.jsonl"

# To use all available GPUs on the current slurm node, we extract the
# number of accelerators available and pass this value to vllm. 
if   [[ -n "${SLURM_STEP_GPUS:-}" ]];            then gpu_env="$SLURM_STEP_GPUS"
elif [[ -n "${SLURM_JOB_GPUS:-}"  ]];            then gpu_env="$SLURM_JOB_GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]];       then gpu_env="$CUDA_VISIBLE_DEVICES"
else  gpu_env=$(nvidia-smi -L | awk '{print NR-1}' | paste -sd "," -)   # "0,1,2…" list
fi

IFS=',' read -ra ids <<< "$gpu_env"
NUM_GPUS=${#ids[@]}

# HPC has a mix of RTX8000s, A100s, V100s, and H100s.
# The RTX8000s are stuck on an older "compute capability" which doesn't
# support bfloat16; so if we're running on a node with RTXs, we need
# to manually override vLLM's dtype
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1)  # e.g. 7.5
MAJOR=${CC%%.*}
DTYPE_FLAG=""
if (( MAJOR < 8 )); then
     echo "compute capability ${CC} < 8.0 → forcing fp16"
     DTYPE_FLAG="--dtype half"   # fp16
fi

echo "Starting vllm with $NUM_GPUS GPU(s)"
echo "Running $MODEL_PATH_NAME"

set -a && source .env && set +a && \
uv run scripts/generate.py \
     --model $MODEL \
     --grammar_name $GRAMMAR_NAME \
     ${SUBSAMPLE_n:+--subsample_n="$SUBSAMPLE_N"} && \
uv run python \
     -m vllm.entrypoints.openai.run_batch \
     -i $INPUT_FILE \
     -o results-g3-4b.jsonl \
     --model $MODEL \
     --tensor-parallel-size $NUM_GPUS \
     ${DTYPE_FLAG}

