#!/bin/bash
#SBATCH -J scoring
#SBATCH -o scoring.out
#SBATCH -e scoring.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=10G
#SBATCH -p GPU
#SBATCH -t 20160

source /opt/apps/anaconda3-2023.03-1.sh

conda activate apscore

python scoring.py --mode batch \
    --input_dir /path/to/coreset/ \
    -m /path/to/checkpoint.pth \
    -o coreset_scoring.dat
