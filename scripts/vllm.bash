#!/bin/bash

set -euo pipefail

# --------- defaults ----------------------------------------------------------

MAX_TOKENS=$((1<<14))
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
GRAMMAR_NAME="grammar_20250218222557"
SUBSAMPLE_N=""

usage() {
     echo "usage: $0 [-M model] [-G grammar_name] [-k max_tokens] [-n subsample]" >&2
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

# -------------- derived values --------------

MODEL_PATH_NAME="${MODEL//\//_}"
INPUT_FILE="data/grammars/${GRAMMAR_NAME}/${GRAMMAR_NAME}_${MODEL_PATH_NAME}_batched_0-shot.jsonl"
OUTPUT_FILENAME=""

# ---------- output file ----------
if [[ -z "$OUTPUT_FILENAME" ]]; then
     if command -v sha1sum &>/dev/null; then
          HASH=$(printf '%s' "${MODEL}${GRAMMAR_NAME}" | sha1sum | awk '{print $1}' | head -c 8)
     else  # macos /bsd fallback
          HASH=$(printf '%s' "${MODEL}${GRAMMAR_NAME}" | shasum | awk '{print $1}' | head -c 8)
     fi
     OUTPUT_FILENAME="batch_${HASH}_results.jsonl"
fi
OUTPUT_FILE="data/grammars/${GRAMMAR_NAME}/${OUTPUT_FILENAME}"

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

# ---------------- BUILD COMMANDS ----------------------

GEN_CMD=(uv run scripts/generate.py openai_batch
     --model "$MODEL"
     --grammar_name "$GRAMMAR_NAME"
     --max_new_tokens "$MAX_TOKENS")
[[ -n "$SUBSAMPLE_N" ]] && GEN_CMD+=(--subsample_n "$SUBSAMPLE_N")

VLLM_CMD=(uv run python
     -m vllm.entrypoints.openai.run_batch
     -i "$INPUT_FILE"
     -o "$OUTPUT_FILE"
     --model "$MODEL"
     --tensor-parallel-size "$NUM_GPUS")
[[ -n "$DTYPE_FLAG" ]] && VLLM_CMD+=("$DTYPE_FLAG")

set -a && source .env && set +a

echo "Generating prompts..."
"${GEN_CMD[@]}"

echo "Starting vLLM with $NUM_GPUS GPU(s) for $MODEL_PATH_NAME"
"${VLLM_CMD[@]}"

echo "Results written to $OUTPUT_FILE"
