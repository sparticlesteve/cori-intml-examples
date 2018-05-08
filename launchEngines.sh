#!/bin/bash

nNodes=4
nEngines=8

# Submit engines to batch
salloc -C haswell -q interactive -t 2:00:00 -N $nNodes \
    ./startEngines.sh $nEngines
