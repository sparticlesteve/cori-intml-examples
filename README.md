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
# Enable the extensions
# Confirm the extensions are enabled
```

## Starting the cluster

## Distributed training

## Distributed random-search hyper-parameter optimization
