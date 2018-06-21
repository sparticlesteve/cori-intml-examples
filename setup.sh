# Using a conda env
module load python
. activate /global/cscratch1/sd/sfarrell/conda/isc-ihpc

# Set some threading variables
export NUM_INTER_THREADS=2
export NUM_INTRA_THREADS=32
export OMP_NUM_THREADS=32
