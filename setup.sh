# Example environment setup script for jupyter deep learning on Cori

# Setup the Cori software
module load tensorflow/intel-1.9.0-py36

# Set some threading environment variables
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export NUM_INTER_THREADS=2
export NUM_INTRA_THREADS=32
export OMP_NUM_THREADS=$NUM_INTRA_THREADS
