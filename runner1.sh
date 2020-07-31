#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -J umap
#SBATCH -q regular
#SBATCH -t 10:00:00

conda activate umap
module load python
export HDF5_USE_FILE_LOCKING=FALSE 
srun -n 1 -c64 --cpu-bind=cores python ./umapL2.py