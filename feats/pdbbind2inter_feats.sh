#!/bin/bash
#SBATCH -J pdbbind
#SBATCH -o pdbbind.out
#SBATCH -e pdbbind.err
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --mem=10G
#SBATCH -p CPU
#SBATCH -t 20160

source /opt/apps/anaconda3-2023.03-1.sh

conda activate apscore

python dataset2inter_feats.py \
    --input_dirs /path/to/PDBind/general_set /path/to/PDBind/refined_set \
    --output_prefix /path/to/output/dataset/ \
    --affinity_file /path/to/INDEX_general_PL_data.2020 \
    --coreset_path /path/to/coreset/ \
    --batch_size 500
