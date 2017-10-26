#!/bin/bash
# Small script to setup the environment 
# necessary for running the adversarial 
# neural network training an evaluation.

if   [[ "$HOSTNAME" == *"lxplus"* ]]; then
    source scripts/lxplus/setup.sh "$@"
elif [[ "$HOSTNAME" == *"ed.ac.uk"* ]]; then
    source scripts/eddie3/setup.sh "$@"
else
    echo "Host not recognised; unable to setup environment."
fi
