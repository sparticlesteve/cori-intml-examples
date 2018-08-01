#!/bin/bash

. setup.sh

# Get the IP address of our head node
headIP=$(ip addr show ipogif0 | grep '10\.' | awk '{print $2}' | awk -F'/' '{print $1}')

# Use a unique cluster ID for this job
clusterID=cori_${SLURM_JOB_ID}
 
echo "Launching controller"
ipcontroller --ip="$headIP" --cluster-id=$clusterID &
sleep 30
 
echo "Launching engines"
srun ipengine --cluster-id=$clusterID
