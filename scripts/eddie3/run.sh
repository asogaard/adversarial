#!/bin/bash

# Main job running the training, 
# evaluation, etc. of the adversarial 
# neural network.
# --------------------------------------

# @TODO: 
# - Add commandline argument specifying config file
# - Add necessary checks
# -

# Settings
# ------------------
# Set name of job
#$ -N run
#
# Set to use current working directory
#$ -cwd
#
# Wait for staging job to finish
#$ -hold_jid stagein

# Specify requested amount of memory per core
#$ -l h_vmem=8G
#
# Specify hard runtime limit
#$ -l h_rt=10:00:00

# @NOTE Will this work? NO.
# IF GPU: Request 'gpu' parallel environment with 2 CPU cores (necessary to have
# enough memory ca. 11GB < 8 x 8GB) and N GPUs
#---$ -pe gpu 8

# IF CPU: Specify requested number of compute cores
#$ -pe sharedmem 8
#


# Send mail to these users
#$ -M andreas.sogaard@ed.ac.uk
#
# Mail at beginning/end/on suspension
#$ -m bes

# Script
# ------------------
# Set up correct environment
echo "Setting up python environment"
. /etc/profile.d/modules.sh
# @TODO Implement flag
if [ "$GPU" == true ]; then
    . ./setup.sh gpu
else
    . ./setup.sh cpu
fi
  

# Run python program
echo "Reading data from $INPUTDIR/*.root"
echo "Writing to $OUTPUTDIR/"
mkdir -p $OUTPUTDIR
echo ""

GROUPPATH=/exports/csce/eddie/ph/groups/PPE/asogaard
echo "Adding '$GROUPPATH' to PYTHONPATH"
export PYTHONPATH=$PYTHONPATH:$GROUPPATH
echo ""

# Set up command-line arguments to the python script
FLAGS="-i $INPUTDIR -o $OUTPUTDIR"
if [ "$GPU" == true ]; then
    FLAGS="$FLAGS --gpu"
fi
if [ "$TRAIN" == true ]; then
    FLAGS="$FLAGS --train"
fi
if [ "$TENSORFLOW" == true ]; then
    FLAGS="$FLAGS --tensorflow"
fi
if [ ! -z "$DEVICES" ]; then 
    FLAGS="$FLAGS --devices $DEVICES"
fi
if [ ! -z "$FOLDS" ]; then 
    FLAGS="$FLAGS --folds $FOLDS"
fi
if [ ! -z "$CONFIG" ]; then 
    FLAGS="$FLAGS --config $CONFIG"
fi
if [ ! -z "$PATCH" ]; then 
    read -r -a PATCH_ARRAY <<< "$PATCH"
    for THIS_PATCH in "${PATCH_ARRAY[@]}"; do
	FLAGS="$FLAGS --patch $THIS_PATCH"
    done
fi

echo "Running program with flags:"
echo "  $FLAGS"
./run.py $FLAGS 2>&1 | tee $OUTPUTDIR/log.txt
echo "Done"
