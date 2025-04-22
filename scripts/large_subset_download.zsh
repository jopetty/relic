#!/bin/zsh

# Read in the list of grammar names from data/large_subset.txt into an array
# read each line into an array
grammar_names=()
while IFS= read -r line; do
  grammar_names+=("$line")
done < data/large_subset.txt

# Subsample for testing
grammar_names=("${grammar_names[@]}")

echo "Read in ${#grammar_names[@]} grammars from data/large_subset.txt"

for g_name in "${grammar_names[@]}"; do
  echo "Processing grammar: $g_name"
  uv run scripts/download.py openai_batch --grammar_name="$g_name"
done
