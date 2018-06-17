#!/bin/bash

. setup.sh

# Get the IP address of our head node
headIP=$(ip addr show ipogif0 | grep '10\.' | awk '{print $2}' | awk -F'/' '{print $1}')

# Programatically name this profile
#profile=job_${SLURM_JOB_ID}_$(hostname)
#echo "Creating profile ${profile} on IP $headIP"
#ipython profile create ${profile}
 
echo "Launching controller"
ipcontroller --ip="$headIP" & #--log-to-file &
#ipcontroller --ip="$headIP" --profile=${profile} --log-to-file
#--nodb
sleep 10
 
echo "Launching engines"
srun ipengine #--log-to-file
#srun ipengine --profile=${profile} --location=$(hostname) #--log-to-file &
