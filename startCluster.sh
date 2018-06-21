#!/bin/bash

. setup.sh

# Get the IP address of our head node
headIP=$(ip addr show ipogif0 | grep '10\.' | awk '{print $2}' | awk -F'/' '{print $1}')

# Construct the IPython profile named by job ID
#profile=cori_${SLURM_JOB_ID}
#echo "Creating profile ${profile} on IP $headIP"
#ipython profile create ${profile}

# Use a unique cluster ID for this job
clusterID=cori_${SLURM_JOB_ID}
 
echo "Launching controller"
ipcontroller --ip="$headIP" --cluster-id=$clusterID &
#--profile="${profile}"
#--log-to-file
#--nodb
sleep 20
 
echo "Launching engines"
srun ipengine --cluster-id=$clusterID
#--profile="$profile"
#--location=$(hostname)
#--log-to-file
