#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -D /fhome/nqing/TFG_project  # Use your actual working dir
#SBATCH -t 1-00:00
#SBATCH -p tfg
#SBATCH --mem 2048
#SBATCH --gres gpu:1
#SBATCH -o classifier_model/outputs/output%j.out
#SBATCH -e classifier_model/outputs/error%j.err

source .venv/bin/activate

python -u classifier_model/main.py \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --hidden_features 256 \
    --dropout 0.5\
    --weighted_loss