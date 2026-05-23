#!/bin/bash
#SBATCH --job-name=train_radio_unet_clean_DPM    # Job name
#SBATCH --partition=P100                         # Requested compute partition
#SBATCH --gres=gpu:1                             # Request 1 GPU
#SBATCH --time=2:30:00                           # Maximum task runtime
#SBATCH --output=%x_%j.out                       # Standard output log (%x: job name | %j: job ID)
#SBATCH --error=%x_%j.err                        # Error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenxin.luo@ip-paris.fr

# Force the working directory to be the project root directory
#SBATCH --chdir=/home/infres/cluo-25/radio-map-prediction 

# Load the Apptainer module
module load apptainer

# Training using Apptainer
# --nv：Enable NVIDIA graphics card support
# --bind：Mount the host machine's current directory (.) to the /workspace directory inside the container.
apptainer exec --nv --bind .:/workspace --pwd /workspace rmp_env.sif \
    python train_radio_unet_clean_DPM.py