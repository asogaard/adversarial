#!/bin/bash
# Setup the environment necessary for running the adversarial neural network
# training and evaluation, depending on the current host.

# Host-specific setup
if   [[ "$HOSTNAME" == *"lxplus"* ]]; then
    source scripts/lxplus/setup.sh "$@"
elif [[ "$HOSTNAME" == *"ed.ac.uk"* ]]; then
    source scripts/eddie3/setup.sh "$@"
else
    echo "Host not recognised; unable to setup environment."
    return
fi


# Enable auto-complete for command-line arguments
source scripts/autocomplete.sh