#!/bin/bash
# Utility script to stage some data for testing.

# Import general utility methods
source scripts/utils.sh

# Stage data, depending on host
target="./input/"
source="/eos/atlas/user/a/asogaard/adversarial/data"
mkdir -p $target
if [[ "$(hostname)" == *"lxplus"* ]]; then
    ln -s $source/data.h5 $target
else
    print "Enter your lxplus username and press [ENTER]: "
    read -e -i "$(whoami)" -p ">> " username
    if [[ -z "$username" ]]; then
	warning "No username was specified. Exiting."
	return 1
    fi
    rsync $username@lxplus.cern.ch:$source/data.h5 $target
fi
