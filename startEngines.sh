#!/bin/bash

# Run this on a batch node to start some MPI engines.

nEngine=4
srun -n $nEngine ipengine
