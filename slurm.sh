#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --nodelist=tenant-ac-nowak-h100-reserved-237-02


source ~/anaconda3/etc/profile.d/conda.sh
conda init bash
cd /mnt/vesuv/home/nowak/MT3
export PYTHONPATH="$PWD:$PYTHONPATH"
conda activate vesuv

python3 train.py configs/segmentation/unetr_sf/config_b5.py
