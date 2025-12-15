#!/bin/bash
#SBATCH --job-name=__gpu__
#SBATCH --output=slurm/bc_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1 # GPU per nodes
##SBATCH --gres=gpu:1

# env
source ~/.dbcata/bin/activate

# run
python -m scripts.train_bcmodel
