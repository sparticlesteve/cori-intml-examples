#!/bin/bash

# Run this script on a login node to launch the controller.
. setup.sh
ipcontroller --enginessh=$HOST
