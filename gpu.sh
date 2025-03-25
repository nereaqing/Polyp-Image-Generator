#!/bin/bash
#SBATCH -n 2                 # Number of CPU cores
#SBATCH -N 1                 # All cores on one machine
#SBATCH -D /tmp              # Working directory
#SBATCH -t 0-00:05           # Runtime in D-HH:MM
#SBATCH -p tfg               # Partition to submit to
#SBATCH --mem 2048           # 2GB RAM
#SBATCH -o output1.out      # Output log file
#SBATCH -e error1.err      # Error log file
#SBATCH --gres gpu:1         # Request 1 GPU

# Manually set CUDA 12.4 paths
# export PATH=/usr/local/cuda-12.4/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Activate virtual environment
source /fhome/nqing/TFG_project/.venv/bin/activate  

# Ensure correct Python is used
export PYTHONPATH=/fhome/nqing/TFG_project/.venv/bin/python
python --version 

# Check GPU availability
nvidia-smi

# Run Python script on the GPU
python /fhome/nqing/TFG_project/script_classification_model.py
