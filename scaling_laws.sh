LAUNCH_SLURM_PATH="/scratch/myh2014/formal-gym/run.slurm"

for task in crasp_100k min dyck fom s5 
do 
    for model in EleutherAI/pythia-160m
    do
        echo "Task: $task. Model: $model"
        sbatch "$LAUNCH_SLURM_PATH" "--model_name $model --reinit True --data_dir ./data/tokenized/$task --output_dir ./output/scaling_laws/$task/$model --save_steps 500 --max_steps 10000 --lr 5e-4"
    done
done