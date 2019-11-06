# Example environment setup script for jupyter deep learning on Cori

# Setup the Cori software
module load tensorflow/intel-1.13.1-py36

# Cray HPO library
module load cray-hpo

# Set some threading environment variables
export KMP_AFFINITY="granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export NUM_INTER_THREADS=2
export NUM_INTRA_THREADS=32
export OMP_NUM_THREADS=$NUM_INTRA_THREADS
