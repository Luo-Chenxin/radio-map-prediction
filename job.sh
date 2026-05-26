#!/bin/bash
#SBATCH --job-name=train_radio_unet_clean_DPM    # Job name
#SBATCH --partition=3090                         # Requested compute partition
#SBATCH --gres=gpu:1                             # Request 1 GPU
#SBATCH --time=03:40:00                          # Maximum task runtime
#SBATCH --output=outputs/%x_%j_out.log           # Standard output log (%x: job name | %j: job ID)
#SBATCH --error=outputs/%x_%j_err.log            # Error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenxin.luo@ip-paris.fr

# Force the working directory to be the project root directory
#SBATCH --chdir=/home/infres/cluo-25/radio-map-prediction 

# Make outputs directory
mkdir -p outputs

# Load the Apptainer module
module load apptainer

# Training using Apptainer
# --nv：Enable NVIDIA graphics card support
# --bind：Mount the host machine's current directory (.) to the /workspace directory inside the container.
apptainer exec --nv --bind .:/workspace --pwd /workspace rmp_env.sif \
    python job.py