#!/bin/bash
#SBATCH -n 2
#SBATCH -N 1
#SBATCH -D /fhome/nqing/TFG_project  # Use your actual working dir
#SBATCH -t 1-00:00
#SBATCH -p tfg
#SBATCH --mem 2048
#SBATCH --gres gpu:1
#SBATCH -o generator_model/outputs/output%j.out
#SBATCH -e generator_model/outputs/error%j.err

source .venv/bin/activate

python -u generator_model/test_diffusion_model.py \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --hidden_features 256 \
    --image_size 128 \
    --dropout 0.5 \
    --experiment_name diffusion_from_scratch \
    --run_id 18aef7b165de44608aa8eeffaf4d46db \
    --path_model_to_test models/generator_model/diffusion_scratch_20250429_165321