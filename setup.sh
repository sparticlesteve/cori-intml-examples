# For the environment on cori, we need the python3 module
# plus my own installation of ipyparallel (for now).
module load python/3.6-anaconda-4.4
export PYTHONPATH=/global/cscratch1/sd/sfarrell/ipyparallel/lib/python3.6/site-packages:$PYTHONPATH
export PATH=/global/cscratch1/sd/sfarrell/ipyparallel/bin:$PATH
