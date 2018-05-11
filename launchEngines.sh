#!/bin/bash

nNodes=32
nEngines=32
if [ $# -gt 0 ]; then
    nNodes=$1
    nEngines=$1
fi

# Submit engines to batch
echo "Submitting $nEngines engines on $nNodes nodes"
salloc -C haswell -q interactive -t 4:00:00 -N $nNodes \
    ./startEngines.sh $nEngines
