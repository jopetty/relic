#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:59
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1 
#SBATCH --job-name=midtraining
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=myh2014@nyu.edu

# beefier config for meta-training and language modeling

extra_args=$1

module purge

cd /scratch/myh2014/formal-gym

# srun singularity exec --nv \
# --overlay /scratch/myh2014/singularity/50G.ext3:ro \
# /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif  /bin/bash -c \
# "source /ext3/env.sh; conda activate skillit; python main.py $extra_args" 

# srun singularity exec --nv \
# --overlay /scratch/myh2014/singularity/50G.ext3:ro \
# /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif  /bin/bash -c \
# "source /ext3/env.sh; conda activate unsloth_env; python run.py" 

srun singularity exec --nv \
--overlay /scratch/myh2014/singularity/50G.ext3:ro \
/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif  /bin/bash -c \
"source /ext3/env.sh; conda activate unsloth_env; python finetune.py $extra_args" 


# srun singularity exec --nv \
# --overlay /scratch/myh2014/singularity/50G.ext3:ro \
# /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif  /bin/bash -c \
# "source /ext3/env.sh; conda activate skillit; python finetune_notrl.py $extra_args" 