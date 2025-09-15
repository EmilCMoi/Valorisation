#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=71
#SBATCH --gpus-per-task=1
#SBATCH -A lp96
#SBATCH --uenv=quantumespresso/v7.4:v2
#SBATCH --view=default
#SBATCH -o 'relax.out'
#SBATCH --time=00:30:00
#SBATCH --mail-user='emiliano.cruzaranda@epfl.ch'
#SBATCH --mail-type='all'

export OMP_NUM_THREADS=20
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_PLACES=cores

srun -u --cpu-bind=socket /user-environment/env/default/bin/pw.x < relax.in
