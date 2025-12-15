#!/bin/bash
#SBATCH --job-name=__gpu__
#SBATCH --output=slurm/train_%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# env
source ~/.dbcata/bin/activate

# run
python -m scripts.train 
