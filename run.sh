
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

for dataset in depth6 depth9
do 
    for seed in 0 
    do
        for gradient_accumulation_steps in 2 4
        do 
            sbatch "$LAUNCH_SLURM_PATH" "--dataset $dataset --seed $seed --warmup_steps 500 --total_steps 50000 --bsz 32 --sampling_p 1.0 --save_every 5000 --gradient_accumulation_steps $gradient_accumulation_steps --lr_scheduler cosine_with_min_lr"
        done 
    done
done

