#!/bin/bash

# Run this on a batch node to start some MPI engines.
nEngines=4
if [ $# -gt 0 ]; then
    nEngines=$1
fi

export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY="granularity=fine,compact,1,0"
export NUM_INTER_THREADS=2
export NUM_INTRA_THREADS=16 #66 knl
export OMP_NUM_THREADS=$NUM_INTRA_THREADS

. setup.sh
srun -n $nEngines ipengine
