#!/bin/bash

LAUNCH_SLURM_PATH="/scratch/myh2014/formal-gym/eval.slurm"


# # Check if directory argument is provided
# if [ $# -ne 1 ]; then
#     echo "Usage: $0 <directory_path>"
#     exit 1
# fi

# # Store the base directory path
# base_dir="$1"

# # Check if the directory exists
# if [ ! -d "$base_dir" ]; then
#     echo "Error: Directory '$base_dir' does not exist"
#     exit 1
# fi

# # Iterate through all subdirectories
# for subdir in "$base_dir"*/; do
#     if [ -d "$subdir" ]; then
#         echo "Processing checkpoint in: $subdir"
#         sbatch "$LAUNCH_SLURM_PATH" "blimp --model_dir $subdir --bsz 128"
#     fi
# done

# for task in blimp
# do  
#     for dataset in crasp_100k dyck
#     do  
#         for seed in 1 2
#         do
#             # pruned
#             sbatch "$LAUNCH_SLURM_PATH" "$task --model_dir ./output/scaling_laws_transfer/c4/pythia-160m/$dataset-500-lr5e4-$seed/checkpoint-10000/ --load_pruned_model True --log_alpha_path ./output/prune/$dataset/pythia-160m/$seed/checkpoint-500/aggregated_params.npy"

#             # regular
#             sbatch "$LAUNCH_SLURM_PATH" "$task --model_dir ./output/scaling_laws_transfer/c4/pythia-160m/$dataset-500-lr5e4-$seed/checkpoint-10000/ --load_pruned_model False" 

#             for dummy_val in 0 1 2 3 4
#             do
#                 sbatch "$LAUNCH_SLURM_PATH" "$task --model_dir ./output/scaling_laws_transfer/c4/pythia-160m/$dataset-500-lr5e4-$seed/checkpoint-10000/ --load_pruned_model True --log_alpha_path ./output/prune/$dataset/pythia-160m/$seed/checkpoint-500/dp$dummy_val.npy"
#             done
#         done
#     done
# done

for seed in 1 2
do
    # regular
    sbatch "$LAUNCH_SLURM_PATH" "$task --model_dir ./output/scaling_laws_transfer/baseline/pythia-160m/c4-lr5e4-$seed/checkpoint-10000/ --load_pruned_model False" 
done
