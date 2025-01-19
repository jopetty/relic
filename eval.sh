#!/bin/bash

LAUNCH_SLURM_PATH="/scratch/myh2014/formal-gym/eval.slurm"


# Check if directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Store the base directory path
base_dir="$1"

# Check if the directory exists
if [ ! -d "$base_dir" ]; then
    echo "Error: Directory '$base_dir' does not exist"
    exit 1
fi

# Iterate through all subdirectories
for subdir in "$base_dir"*/; do
    if [ -d "$subdir" ]; then
        echo "Processing checkpoint in: $subdir"
        sbatch "$LAUNCH_SLURM_PATH" "blimp --model_dir $subdir --bsz 128"
    fi
done

