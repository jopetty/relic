for seed in 1 2
do
    for task in crasp_100k dyck
    do  
        # for task_seed in 1 2 3 4 5
        # do
        #     python aggregate_pruning.py make_dummy_params --super_dir ./output/prune/$task/pythia-160m/$seed/checkpoint-500/ --output_name dp$task_seed.npy --seed $task_seed
        # done

        python aggregate_pruning.py main --super_dir ./output/prune/$task/pythia-160m/$seed/checkpoint-500/ 
    done
done

