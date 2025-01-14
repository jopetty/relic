# Experiment 2: Which formal language to train on?

LAUNCH_SLURM_PATH="/scratch/myh2014/formal-gym/run.slurm"

# for task in code cogs arithmetic wikitext dyck fom min s5 crasp_100k
# do 
#     for model in EleutherAI/pythia-160m 
#     do
#         model_name=$(basename "$model")
#         echo "Task: $task. Model: $model_name"
#         sbatch "$LAUNCH_SLURM_PATH" "--model_name $model --data_dir ./data/tokenized/$task --output_dir ./output/optimal_language_lr_5e-5_nw/$task/$model_name --save_steps 500 --max_steps 500 --lr 5e-5 --warmup_steps 0 --min_lr_rate 0.1"
#     done
# done

# for task in code cogs arithmetic wikitext dyck fom min s5 crasp_100k
# do 
#     for model in EleutherAI/pythia-160m 
#     do
#         model_name=$(basename "$model")
#         echo "Task: $task. Model: $model_name"
#         sbatch "$LAUNCH_SLURM_PATH" "--model_name $model --data_dir ./data/tokenized/$task --output_dir ./output/optimal_language_lr_5e-5_flat/$task/$model_name --save_steps 500 --max_steps 500 --lr 5e-5 --warmup_steps 0 --min_lr_rate 1.0"
#     done
# done


# part 2: transfer


for task in code cogs arithmetic 
do 
    for model in crasp_100k dyck fom s5 # code cogs arithmetic wikitext min
    do
        echo "Task: $task. Model: $model"
        sbatch "$LAUNCH_SLURM_PATH" "--model_name ./output/optimal_language_lr_5e-5_nw/$model/pythia-160m/checkpoint-500 --data_dir ./data/tokenized/$task --output_dir ./output/optimal_language_transfer/$task/$model --save_steps 5000 --max_steps 5000"
    done
done

# for task in code cogs arithmetic 
# do 
#     sbatch "$LAUNCH_SLURM_PATH" "--model_name EleutherAI/pythia-160m --data_dir ./data/tokenized/$task --output_dir ./output/optimal_language_transfer/$task/pythia-160m --save_steps 5000 --max_steps 5000"
# done