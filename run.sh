
LAUNCH_SLURM_PATH="/scratch/myh2014/formal-gym/run.slurm"

for dataset in depth6 depth9
do 
    for seed in 0 1 2
    do 
        sbatch "$LAUNCH_SLURM_PATH" "--dataset $dataset --seed $seed --warmup_steps 500 --total_steps 10000 --bsz 16 --sampling_p 0.2"
    done
done

for sampling_p in 0 1
do 
    for seed in 0 1 2
    do 
        sbatch "$LAUNCH_SLURM_PATH" "--dataset depth6 --seed $seed --warmup_steps 500 --total_steps 10000 --bsz 16 --sampling_p $sampling_p"
    done
done