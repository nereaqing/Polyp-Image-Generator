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
    --image_size 224 \
    --dropout 0.5 \
    --weighted_sampling \
    --ad_vs_rest \
    --experiment_name diffusion_from_scratch \
    --run_id ea8994f1f593409da7c6a4d874f96a75 \
    --path_model_to_test models/generator_model/diffusion_20250517_211218