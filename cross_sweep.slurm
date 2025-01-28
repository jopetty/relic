#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:59
#SBATCH --mem=64GB
#SBATCH --job-name=data_gen
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=myh2014@nyu.edu

module purge
cd /scratch/myh2014/formal-gym

for p in 0.2 0.3 0.4 0.5; do
    for num_symbols in 32 64 128 256; do
        srun singularity exec \
            --overlay /scratch/myh2014/singularity/50G.ext3:ro \
            /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
            /bin/bash -c \
            "source /ext3/env.sh; \
             conda activate unsloth_env; \
             python -m src.crasp.grammar generate_cross \
                --file_dir ./data/ \
                --num_symbols ${num_symbols} \
                --p ${p}"
    done
done