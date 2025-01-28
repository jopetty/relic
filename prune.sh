LAUNCH_SLURM_PATH="/scratch/myh2014/formal-gym/prune.slurm"

for target_sparsity in 0.3 0.4 0.5 0.6 0.7
do
    for dataset in cross
    do  
        for seed in 0 1 2 
        do 
            sbatch $LAUNCH_SLURM_PATH "--target_sparsity $target_sparsity --max_steps 5000 --sparsity_warmup_steps 1000 --model_name ./output/scaling_laws/$dataset/EleutherAI/pythia-160m/$seed/checkpoint-500/ --data_dir ./data/tokenized/$dataset --output_dir ./output/prune/$dataset/pythia-160m/$seed/checkpoint-500/$target_sparsity --bsz 4 --save_steps 5000"
        done
    done 
done
