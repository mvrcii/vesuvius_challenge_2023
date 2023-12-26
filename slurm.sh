#!/bin/bash
#SBATCH --job-name=train

source ~/anaconda3/etc/profile.d/conda.sh
conda init bash
cd /mnt/vesuv/home/nowak/kaggle1stReimp 
export PYTHONPATH="$PWD:$PYTHONPATH"
conda activate vesuv

python3 train.py configs/segmentation/unet3d_sf/config_b5.py
