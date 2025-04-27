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

python -u generator_model/training_stable_diffusion.py 