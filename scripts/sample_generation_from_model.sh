#!/usr/bin/env bash

# Usage: ./sample_generation_from_model.sh -n 5 -s 3 -m gpt-4.1-nano

set -e

# Defaults
N=20
N_SHOTS=0
MODEL="gpt-4.1-nano"

# Parse args using getopts
while getopts "n:s:m:" opt; do
  case $opt in
    n) N="$OPTARG" ;;
    s) N_SHOTS="$OPTARG" ;;
    m) MODEL="$OPTARG" ;;
    *) echo "Usage: $0 [-n number] [-s n_shots] [-m model]"; exit 1 ;;
  esac
done

# read each line into an array
grammar_names=()
while IFS= read -r line; do
  grammar_names+=("$line")
done < data/large_subset.txt

# Subsample for testing
grammar_names=("${grammar_names[@]:0:${N}}")
echo "Read in ${#grammar_names[@]} grammars from data/large_subset.txt"

# Check to see if the batch exists
for g_name in "${grammar_names[@]}"; do
  if [[ ! -f "data/grammars/$g_name/${g_name}_${MODEL}_batched_${N_SHOTS}-shot_generate.jsonl" ]]; then
    echo "→ Running on $g_name"
    uv run scripts/generate.py openai_batch \
      --grammar_name "$g_name" \
    --evaluation generate \
    --n_shots "$N_SHOTS" \
    --model "$MODEL"
  fi
done

for g_name in "${grammar_names[@]}"; do
  echo "Processing grammar: $g_name"
  uv run scripts/upload.py openai_batch --grammar_name="$g_name" --model="$MODEL" --eval_task="generate"
done

