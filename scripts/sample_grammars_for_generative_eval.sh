#!/usr/bin/env bash

# Usage: ./sample_and_generate.sh -n 5 -s 3 -m deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

set -e

# Defaults
N=20
N_SHOTS=0
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# Parse args using getopts
while getopts "n:s:m:" opt; do
  case $opt in
    n) N="$OPTARG" ;;
    s) N_SHOTS="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    *) echo "Usage: $0 [-n number] [-s n_shots] [-m model]"; exit 1 ;;
  esac
done

# Helper: sample N lines from a file
sample_from_file() {
  FILE="$1"
  if command -v shuf &>/dev/null; then
    shuf -n "$N" "$FILE"
  elif command -v gshuf &>/dev/null; then
    gshuf -n "$N" "$FILE"
  else
    echo "Error: shuf or gshuf not found."; exit 1
  fi
}

echo "Sampling $N grammars from data/large_subset.txt and data/small_subset.txt..."
GRAMMAR_NAMES=$( (sample_from_file data/large_subset.txt; sample_from_file data/small_subset.txt) | sort -u )

echo "Running generation on sampled grammars..."
for GRAMMAR_NAME in $GRAMMAR_NAMES; do
  echo "→ Running on $GRAMMAR_NAME"
  uv run scripts/generate.py openai_batch \
    --grammar_name "$GRAMMAR_NAME" \
    --evaluation generate \
    --n_shots "$N_SHOTS" \
    --model "$MODEL"
done
