# For the environment on cori, we need the tensorflow module
# with horovod plus my own installations of ipyparallel, jupyter, etc.

# Now using a conda env
module load python
. activate /global/cscratch1/sd/sfarrell/conda/isc-ihpc

#module load tensorflow/intel-1.8.0-py27
#export PATH=/global/homes/s/sfarrell/.local/cori/intel-tensorflow1.8.0-py27/bin:$PATH

#module load tensorflow/intel-horovod-mpi-1.6
#export PYTHONPATH=/global/cscratch1/sd/sfarrell/tf-hvd-ipp/lib/python2.7/site-packages:$PYTHONPATH
#export PATH=/global/cscratch1/sd/sfarrell/tf-hvd-ipp/bin:$PATH
