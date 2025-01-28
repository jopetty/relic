LAUNCH_SLURM_PATH="/scratch/myh2014/formal-gym/run.slurm"

# part 1: generate initializations

# for task in rep dyck_v2 cross
# do 
#     for model in EleutherAI/pythia-160m
#     do
#         for seed in 0 1 2
#         do 
#             echo "Task: $task. Model: $model"
#             sbatch "$LAUNCH_SLURM_PATH" "--model_name $model --reinit True --data_dir ./data/tokenized/$task --output_dir ./output/scaling_laws/$task/$model/$seed --save_steps 500 --max_steps 4000 --lr 5e-4 --seed $seed"
#         done
#     done
# done

# baselines

# sbatch "$LAUNCH_SLURM_PATH" "--model_name EleutherAI/pythia-160m --data_dir babylm --output_dir ./output/scaling_laws_transfer/baseline/pythia-160m/babylm-lr5e4 --save_steps 10000 --max_steps 10000 --lr 5e-4 --warmup_steps 1000 --bsz 16 --reinit True"

# sbatch "$LAUNCH_SLURM_PATH" "--model_name EleutherAI/pythia-160m --data_dir c4 --output_dir ./output/scaling_laws_transfer/baseline/pythia-160m/c4-lr5e5 --save_steps 10000 --max_steps 10000 --lr 5e-4 --warmup_steps 1000 --bsz 16 --reinit True"

# part 2: run transfer to wikitext, c4, babylm

# for seed in 0 1 2
# do 
#     for initial_task in fom
#     do
#         for timestep in 500 
#         do
#             sbatch "$LAUNCH_SLURM_PATH" "--model_name ./output/scaling_laws/$initial_task/pythia-160m/$seed/checkpoint-$timestep --data_dir c4 --output_dir ./output/scaling_laws_transfer/c4/pythia-160m/$initial_task-$timestep-lr5e4-$seed --save_steps 10000 --max_steps 10000 --lr 5e-4 --warmup_steps 1000 --bsz 16 --seed $seed --use_callback True"
#         done
#     done
# done


# generate inits 

# for seed in 0 1 2 
# do
#     for task in crasp_100k min dyck fom s5 
#     do 
#         for model in pythia-410m
#         do
#             echo "Task: $task. Model: $model"
#             sbatch "$LAUNCH_SLURM_PATH" "--model_name EleutherAI/$model --reinit True --data_dir ./data/tokenized/$task --output_dir ./output/scaling_laws/$task/$model/$seed --save_steps 500 --max_steps 4000 --lr 5e-4 --seed $seed"
#         done
#     done
# done


# final transfer:

# for seed in 0 1 2
# do 
    # sbatch "$LAUNCH_SLURM_PATH" "--model_name EleutherAI/pythia-160m --data_dir c4 --output_dir ./output/scaling_laws_transfer/baseline/pythia-160m/c4-lr5e4-$seed --save_steps 100 --max_steps 10000 --lr 5e-4 --warmup_steps 1000 --bsz 16 --reinit True --seed $seed"

    # part 2: run transfer to c4
#     for initial_task in crasp_100k min dyck fom s5
#     do
#         for timestep in 500 1000 2000 4000 
#         do
#             sbatch "$LAUNCH_SLURM_PATH" "--model_name ./output/scaling_laws/$initial_task/pythia-160m/$seed/checkpoint-$timestep --data_dir c4 --output_dir ./output/scaling_laws_transfer/c4/pythia-160m/$initial_task-$timestep-lr5e4-$seed --save_steps 10000 --max_steps 10000 --lr 5e-4 --warmup_steps 1000 --bsz 16 --seed $seed"
#         done
#     done
# done

# ablations

# for initial_task in random_binary_dataset random_int_string_dataset # shuffle_spans_4 shuffle_spans_8 shuffle_spans_4_nd shuffle_spans_8_nd full_shuffle # unigram bigram trigram # 
# do
#     for timestep in 500 
#     do  
#         for seed in 0 1 2
#         do
#             sbatch "$LAUNCH_SLURM_PATH" "--model_name EleutherAI/pythia-160m --reinit True --data_dir ./data/tokenized/$initial_task --output_dir ./output/scaling_laws/$initial_task/pythia-160m/$seed --save_steps 500 --max_steps 500 --lr 5e-4 --seed $seed"
#         done
#     done
# done


# for initial_task in c4_2 # shuffle_spans_4 shuffle_spans_8 shuffle_spans_4_nd shuffle_spans_8_nd full_shuffle # unigram bigram trigram # 
# do
#     for timestep in 500 
#     do  
#         for seed in 0 1 2
#         do
#             sbatch "$LAUNCH_SLURM_PATH" "--model_name EleutherAI/pythia-160m --reinit True --data_dir c4_2 --output_dir ./output/scaling_laws/$initial_task/pythia-160m/$seed --save_steps 500 --max_steps 500 --lr 5e-4 --seed $seed"
#         done
#     done
# done


for seed in 0 1 2
do 
    for initial_task in random_binary_dataset random_int_string_dataset c4_2
    do
        for timestep in 500 
        do
            sbatch "$LAUNCH_SLURM_PATH" "--model_name ./output/scaling_laws/$initial_task/pythia-160m/$seed/checkpoint-$timestep --data_dir c4 --output_dir ./output/scaling_laws_transfer/c4/pythia-160m/$initial_task-$timestep-lr5e4-$seed --save_steps 10000 --max_steps 10000 --lr 5e-4 --warmup_steps 1000 --bsz 16 --seed $seed"
        done
    done
done


