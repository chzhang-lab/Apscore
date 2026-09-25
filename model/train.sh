#!/bin/bash
#SBATCH -J train
#SBATCH -o train.out
#SBATCH -e train.err
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -p GPU
#SBATCH -t 20160

source /opt/apps/cuda_11.7.1_515.65.01.sh

source /opt/apps/anaconda3-2023.03-1.sh

conda activate apscore

python train.py --data_dir /path/to/preprocessed/dataset/
