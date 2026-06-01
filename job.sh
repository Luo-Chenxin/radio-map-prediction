#!/bin/bash
#SBATCH --job-name=train_radio_unet_clean_DPM     # Job name
#SBATCH --partition=3090                          # Requested compute partition
#SBATCH --gres=gpu:1                              # Request 1 GPU
#SBATCH --time=03:40:00                           # Maximum task runtime
#SBATCH --output=outputs/%x_%j_out.log            # Standard output log (%x: job name | %j: job ID)
#SBATCH --error=outputs/%x_%j_err.log             # Error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenxin.luo@ip-paris.fr

# ==============================================================================
# PARAMETERS & PATHS CONFIGURATION
# ==============================================================================
# Define your project root directory on the cluster host machine
PROJ_DIR="/home/infres/cluo-25/radio-map-prediction"

# Define your work directory in the container
WORK_DIR="/workspace"

# Define the local directory inside the project to cache patch packages
TARGET_PKG_DIR=".temp/temp-packages"

# Define the filename of your Apptainer image
IMAGE_NAME="rmp_env.sif"

# Force Slurm to change the working directory to the project root before execution
#SBATCH --chdir=${PROJ_DIR} 

# ==============================================================================
# ENVIRONMENT & PRE-FLIGHT CHECKS
# ==============================================================================
# Load the Apptainer module on the cluster node
module load apptainer

# Create required directories if they don't exist
mkdir -p "${PROJ_DIR}/outputs"
mkdir -p "${PROJ_DIR}/${TARGET_PKG_DIR}"

# --- Dynamic Package Synchronization ---
echo "Checking and synchronizing packages from environment.yml..."

# Execute pip inside the active container environment
# It parses environment.yml and safely installs new/missing upper-level libraries
# note: core base packages (python, pip, pytorch, etc.) are strictly excluded to protect container stability
PACKAGES=$(grep -E '^\s*-\s+' "${PROJ_DIR}/environment.yml" | \
           sed 's/^[[:space:]]*-[[:space:]]*//' | \
           sed 's/=/==/g' | \
           grep -vE '^(python|pip|pytorch|torchvision|cuda|cudnn|conda)([=>=<[:space:]]|$)')

echo "Installing verified packages: $PACKAGES"

apptainer exec --nv --bind "${PROJ_DIR}":"${WORK_DIR}" --pwd "${WORK_DIR}" "${PROJ_DIR}/${IMAGE_NAME}" \
    pip install $PACKAGES --target="${WORK_DIR}/${TARGET_PKG_DIR}" --quiet --no-cache-dir

# --- Inject Container Environment Variables ---
# Append the temp-packages directory to PYTHONPATH inside the container
export APPTAINERENV_PYTHONPATH="$WORK_DIR/$TARGET_PKG_DIR:$PYTHONPATH"

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
echo "Starting the deep learning training..."
apptainer exec --nv --bind "${PROJ_DIR}":"${WORK_DIR}" --pwd "${WORK_DIR}" "${PROJ_DIR}/${IMAGE_NAME}" \
    python job.py