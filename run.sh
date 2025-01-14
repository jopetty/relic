
LAUNCH_SLURM_PATH="/scratch/myh2014/formal-gym/run.slurm"

# for dataset in depth6 depth9
# do 
#     for seed in 0 1 2
#     do 
#         sbatch "$LAUNCH_SLURM_PATH" "--dataset $dataset --seed $seed --warmup_steps 500 --total_steps 10000 --bsz 16 --sampling_p 0.2"
#     done
# done

# for sampling_p in 0 1
# do 
#     for seed in 0 1 2
#     do 
#         sbatch "$LAUNCH_SLURM_PATH" "--dataset depth6 --seed $seed --warmup_steps 500 --total_steps 10000 --bsz 16 --sampling_p $sampling_p"
#     done
# done

# trying regularization

# for dataset in depth6 depth9
# do 
#     for seed in 0 
#     do
#         for kl_weight in 1 0.1 0.01
#         do 
#             sbatch "$LAUNCH_SLURM_PATH" "--dataset $dataset --seed $seed --kl_loss --warmup_steps 500 --total_steps 10000 --bsz 16 --sampling_p 0.2 --kl_weight $kl_weight"
#         done 
#     done
# done

# trying longer training

# sbatch "$LAUNCH_SLURM_PATH" "--dataset slimpj --seed 0 --warmup_steps 1000 --total_steps 100000 --bsz 16 --gradient_accumulation_steps 4 --save_every 10000"

# sbatch "$LAUNCH_SLURM_PATH" "--dataset slimpj --seed 0 --warmup_steps 1000 --total_steps 100000 --bsz 16 --gradient_accumulation_steps 4 --save_every 10000 --lr 1e-4"

# for lr in 1e-3 6e-4 3e-4 1e-4
# do
#     sbatch "$LAUNCH_SLURM_PATH" "--dataset slimpj --seed 0 --warmup_steps 1000 --total_steps 50000 --bsz 32 --gradient_accumulation_steps 4 --save_every 10000 --lr $lr --shrink_and_perturb --lr_scheduler cosine_with_min_lr"
# done

# swapping from training to merging

# for dataset in depth6 depth9
# do 
#     for seed in 0 
#     do
#         for gradient_accumulation_steps in 2 4
#         do 
#             sbatch "$LAUNCH_SLURM_PATH" "--dataset $dataset --seed $seed --warmup_steps 500 --total_steps 50000 --bsz 32 --sampling_p 1.0 --save_every 5000 --gradient_accumulation_steps $gradient_accumulation_steps --lr_scheduler cosine_with_min_lr"
#         done 
#     done
# done


# for revision in step0 # step128 step512 step1000 step50000 main
# do 
#     sbatch "$LAUNCH_SLURM_PATH" "--revision $revision --output_dir output/step0"
# done 

# for step in 500 1000 2000 3000 4000 5000
# do 
#     sbatch "$LAUNCH_SLURM_PATH" "--model_name outputs/checkpoint-$step --output_dir output/transfer/$step"
# done 

# for task in arithmetic cogs code
# do 
#     for model in ./output/cogs/pythia-160m/checkpoint-500/ ./output/arithmetic/pythia-160m/checkpoint-500/ ./output/code/pythia-160m/checkpoint-500/
#     do
#         model_name=$(basename "$model")
#         echo "Task: $task. Model: $model_name"
#         sbatch "$LAUNCH_SLURM_PATH" "--model_name $model --data_dir ./data/tokenized/$task --output_dir ./output/transfer/$task/$model_name --save_every 5000"
#     done
# done

for task in crasp_100k dyck fom min s5
do 
    for model in EleutherAI/pythia-160m 
    do
        model_name=$(basename "$model")
        echo "Task: $task. Model: $model_name"
        sbatch "$LAUNCH_SLURM_PATH" "--model_name $model --data_dir ./data/tokenized/$task --output_dir ./output/optimal_language/$task/$model_name --save_steps 500 --max_steps 500"
    done
done

# sbatch "$LAUNCH_SLURM_PATH" "--model_name EleutherAI/pythia-160m --data_dir ./data/tokenized/wikitext --output_dir ./output/transfer/wikitext/pythia-160 --save_steps 125 --max_steps 1000"


