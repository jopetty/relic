#!/bin/zsh

# Read in the list of grammar names from data/small_subset.txt into an array
# read each line into an array
grammar_names=()
while IFS= read -r line; do
  grammar_names+=("$line")
done < data/small_subset.txt

# Subsample for testing
grammar_names=("${grammar_names[@]:80:20}")

echo "Read in ${#grammar_names[@]} grammars from data/small_subset.txt"

# Check to see if the batch exists
for g_name in "${grammar_names[@]}"; do
  if [[ ! -f "data/grammars/$g_name/${g_name}_o3_batched_0-shot.jsonl" ]]; then
    uv run scripts/generate.py openai_batch --grammar_name="$g_name" --model="o3" --subsample_n 2
  fi
done

for g_name in "${grammar_names[@]}"; do
  echo "Processing grammar: $g_name"
  uv run scripts/upload.py openai_batch --grammar_name="$g_name" --model="o3"
done
