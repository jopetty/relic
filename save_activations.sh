# activations for trained mdoels
# for task in fom 
# do
#     for seed in 0 1 2
#     do
#         python save_avg_activations.py main --super_dir ./output/scaling_laws_transfer/c4/pythia-160m/$task-500-lr5e4-$seed/ --data_dir c4 --blimp False

#         python save_avg_activations.py main --super_dir ./output/scaling_laws_transfer/c4/pythia-160m/$task-500-lr5e4-$seed/ --blimp True
#     done
# done


for task in fom cross
do
    for seed in 0 1 2
    do
        python save_avg_activations.py main --super_dir ./output/scaling_laws_transfer/c4/pythia-160m/$task-500-lr5e4-$seed --data_dir c4 --blimp False
    done
done

# blimp

# for task in fom cross
# do
#     for seed in 0 1 2
#     do
#         python save_avg_activations.py save_avg_activation --initialize_from ./output/scaling_laws_transfer/c4$task/EleutherAI/pythia-160m/$seed/checkpoint-500/ --data_dir ./data/tokenized/$task 
#     done
# done

