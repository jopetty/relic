#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:59
#SBATCH --mem=64GB
#SBATCH --job-name=data_gen
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=myh2014@nyu.edu


extra_args=$1

module purge

cd /scratch/myh2014/formal-gym

srun singularity exec  \
--overlay /scratch/myh2014/singularity/50G.ext3:ro \
/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif  /bin/bash -c \
"source /ext3/env.sh; conda activate unsloth_env; python -m src.data.utils make_dummy_tasks --data_dir ./data/tokenized/cross --out_dir ./data/tokenized/cross-variants; python -m src.data.utils make_dummy_tasks --data_dir ./data/tokenized/fom --out_dir ./data/tokenized/fom-variants" 
