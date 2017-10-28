#!/bin/bash

# Import utility methods
source scripts/utils.sh

if ! hash conda 2>/dev/null; then
    print "Installing Miniconda."
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    # Follow the screen prompts
    # ...
    rm Miniconda3-latest-Linux-x86_64.sh
else
    print "conda already installed!"    
fi
