#!/bin/bash

#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q premium
#SBATCH -t 2:00:00
#SBATCH -J train_rpv
#SBATCH -o batch_logs/%x-%j.out

. setup.sh
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 64 \
    python ./train_rpv.py
