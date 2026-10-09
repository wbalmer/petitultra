#!/bin/bash
#SBATCH --job-name=2M0624_ultra  # Job name
#SBATCH --output=2M0624_SiO2NUC_homog_outputs/output.txt
#SBATCH --error=2M0624_SiO2NUC_homog_outputs/error.txt
#SBATCH --account=b1094           # allocation/account
#SBATCH --partition=ciera-std	          # Partition or queue name, debug, shared, wholenode
#SBATCH --nodes=12                     # Number of nodes
#SBATCH --ntasks-per-node=52         # Number of tasks per node
#SBATCH --cpus-per-task=1             # Number of CPU cores per task
#SBATCH --time=2-00:00:00             # Maximum runtime (D-HH:MM:SS)
#SBATCH --mail-type=ALL               # Send email at job completion
#SBATCH --mail-user=balmer@northwestern.edu  # Email address for notifications

# get bashrc
source /home/efn6832/.bashrc

# Activate the virtual environment
mamba activate prt3

# create arviz cache to avoid race condition error
mkdir -p /home/efn6832/.cache/arviz/

# Run your Python script

mpiexec -n 624 python weird_clouds_bd_2M0624_sionuc_homog.py

# Deactivate the virtual environment
conda deactivate