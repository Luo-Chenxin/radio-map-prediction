#!/bin/bash
#SBATCH --job-name=build_sif                     # Job name
#SBATCH --partition=CPU                          # Requested compute partition
#SBATCH --cpus-per-task=4                        # Allocate CPU
#SBATCH --mem=32G                                # Allocate Memory
#SBATCH --time=1:20:00                           # Maximum task runtime
#SBATCH --output=%x_%j.out                       # Standard output log (%x: job name | %j: job ID)
#SBATCH --error=%x_%j.err                        # Error log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chenxin.luo@ip-paris.fr

# Force the working directory to be the project root directory
#SBATCH --chdir=/home/infres/cluo-25/radio-map-prediction 

# Load the Apptainer module
module load apptainer

# Start building Apptainer mirror
apptainer build --fakeroot rmp_env.sif rmp_env.def