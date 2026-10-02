#!/bin/bash
#SBATCH --job-name=2M0624_ultra  # Job name
#SBATCH --output=2M0624_SiO2NUC_outputs/output.txt
#SBATCH --error=2M0624_SiO2NUC_outputs/error.txt
#SBATCH --account=b1094           # allocation/account
#SBATCH --partition=ciera-std	          # Partition or queue name, debug, shared, wholenode
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=52         # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --time=30:00             # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=ALL               # Send email at job completion
#SBATCH --mail-user=balmer@northwestern.edu  # Email address for notifications

# Activate the virtual environment
mamba activate prt3

# Run your Python script

mpiexec -n 52 python weird_clouds_bd_2M0624_sionuc.py

# Deactivate the virtual environment
conda deactivate