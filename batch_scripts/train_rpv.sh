#!/bin/bash

#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q premium
#SBATCH -t 2:00:00
#SBATCH -J train_rpv
#SBATCH -o batch_logs/%x-%j.out

export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
export NUM_INTER_THREADS=2
export NUM_INTRA_THREADS=16 #66 knl
export OMP_NUM_THREADS=$NUM_INTRA_THREADS

. setup.sh
srun -N ${SLURM_NNODES} -n ${SLURM_NNODES} -c 64 \
    python ./train_rpv.py
