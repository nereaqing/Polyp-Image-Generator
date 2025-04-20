#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -D /fhome/nqing/TFG_project  # Use your actual working dir
#SBATCH -t 1-00:00
#SBATCH -p tfg
#SBATCH --mem 2048
#SBATCH --gres gpu:1
#SBATCH -o results/outputs/output%j.out
#SBATCH -e results/errors/error%j.err

# Activate your environment
source .venv/bin/activate

# Run your Python script with any desired arguments
python -u main.py \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --hidden_features 512 \
    --dropout 0.5
