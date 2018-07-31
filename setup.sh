# Example environment setup script for jupyter deep learning on Cori

# Using our conda environment which has TF, keras, horovod, IPyParallel, etc.
. /usr/common/software/python/3.6-anaconda-4.4/etc/profile.d/conda.sh
conda activate /global/common/software/dasrepo/JupyterDL

# Set some threading environment variables
export NUM_INTER_THREADS=2
export NUM_INTRA_THREADS=32
export OMP_NUM_THREADS=32
