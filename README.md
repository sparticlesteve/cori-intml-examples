# Jupyter Deep Learning examples for the NERSC Cori supercomputer

This repository contains examples for interactive distributed deep learning
on the Cori supercomputer at NERSC with Jupyter notebooks.
The code accompanies the slides and paper on
_Interactive Distributed Deep Learning with Jupyter Notebooks_
shown at the Interactive HPC workshop at ISC 2018.

## Environment setup

The examples here utilize a custom conda environment on Cori SCRATCH
which should be fine for you to use for now. If you want to tweak things you
can build your own conda environment (you can start by cloning ours) and
update the setup.sh script accordingly. Soon we'll put all the dependencies in
our standard deep learning installations so you can just use those.

You'll have to install the kernel specification for whatever installation you
use. This should follow the instructions on this page:

http://www.nersc.gov/users/data-analytics/data-analytics-2/jupyter-and-rstudio/

First setup the environment (e.g. by sourcing the setup.sh script) and do

```bash
# Give the kernel a unique name
python -m ipykernel install --user --name jupyter-dl
```

If you want to try out the IPyWidgets examples, you'll have to also install
and enable the notebook extensions:

```bash
# Install extensions in your jupyter folder
jupyter nbextension install --user --py qgrid
jupyter nbextension install --user --py bqplot
# Enable the extensions
jupyter nbextension enable --user --py qgrid
jupyter nbextension enable --user --py bqplot
# Confirm the extensions are enabled
jupyter nbextension list
```

## Starting the cluster

Start an IPyParallel cluster on Cori batch nodes via either command line or
via IPython magic commands in the notebook. For now we recommend the command
line option.

### Command line

You can do this completely from your browser by launching a terminal in jupyter.

Get an allocation on Cori. If possible, the interactive queue is a good option
to get tens of nodes for (currently) up to 4 hours:

```bash
# Request 8 nodes for 1 hour
salloc -C haswell -N 8 -q interactive -t 1:00:00
```

You can use the provided script to launch the controller and engines on your
head and worker nodes:

```bash
./startCluster.sh
```

You should see log messages showing the engines connecting to the controller.
The script uses your default ipython profile and creates the cluster with an
ID containing your slurm job ID.

## Distributed training

We have two example notebooks for running data-parallel synchronous training
via Keras and Horovod:

- DistTrain_mnist.ipynb
- DistTrain_rpv.ipynb

These are for a simple MNIST classifier and an ATLAS RPV image classifier
(see https://arxiv.org/abs/1711.03573), respectively.

## Distributed random-search hyper-parameter optimization

We have two example notebooks that run single-node training tasks in
random-search hyper-parameter optimization, corresponding to the same
problems and models in the Distributed training section above:

- DistHPO_mnist.ipynb
- DistHPO_rpv.ipynb

These examples leverage the load-balanced scheduler of IPyParallel to farm
the single-node training tasks out to your cluster. One can then interactively
query and monitor the status of the tasks via the AsyncResult objects as shown
in the notebooks.

## Interactive widgets for distributed HPO

Finally, we have two examples that show how one can incorporate interactive
widgets for live monitoring of hyper-parameter training tasks:

- DistWidgetHPO_mnist.ipynb
- DistWidgetHPO_rpv.ipynb

Note these are still quite experimental and under development.
