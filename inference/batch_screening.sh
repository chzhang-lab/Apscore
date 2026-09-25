#!/bin/bash
#SBATCH -J screening
#SBATCH -o screening.out
#SBATCH -e screening.err
#SBATCH -N 1
#SBATCH -n 48
#SBATCH --mem=10G
#SBATCH -p CPU
#SBATCH -t 20160

source /opt/apps/anaconda3-2023.03-1.sh

conda activate apscore

python screening.py \
    -pb /path/to/protein_base/ \
    -lb /path/to/ligand_base/ \
    -m /path/to/checkpoint.pth \
    -o ./screening_results/
