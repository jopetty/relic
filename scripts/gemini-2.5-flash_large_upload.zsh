#!/bin/zsh

# Read in the list of grammar names from data/large_subset.txt into an array
# read each line into an array
grammar_names=()
while IFS= read -r line; do
  grammar_names+=("$line")
done < data/large_subset.txt

# Subsample for testing
grammar_names=("${grammar_names[@]:60:40}")

echo "Read in ${#grammar_names[@]} grammars from data/large_subset.txt"

N_SHOTS=1
MODEL="gemini-2.5-flash"
# Check to see if the batch exists
for g_name in "${grammar_names[@]}"; do
  if [[ ! -f "data/grammars/$g_name/${g_name}_${MODEL}_batched_$((N_SHOTS*2))-shot.jsonl" ]]; then
    uv run scripts/generate.py google_batch --grammar_name="$g_name" --model=$MODEL --n_shots=$N_SHOTS
  fi
done

for g_name in "${grammar_names[@]}"; do
  echo "Processing grammar: $g_name"
  uv run scripts/upload.py google_batch --grammar_name="$g_name" --model=$MODEL --n_shots=$N_SHOTS
done
