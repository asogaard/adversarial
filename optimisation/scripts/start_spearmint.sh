#!/bin/bash
# Start spearmint program

# Import utility methods
source scripts/utils.sh

# Define variable(s)
experiment="$1"
experimentpath="optimisation/experiments/$experiment/"

# Check(s)
if [ -z "$experiment" ]; then
    warning "start_spearmint: Please provide an experiment name, i.e."
    warning " $ source path/to/start_spearmint.sh <experiment>"
    return 1
fi

if [ ! -d "$experimentpath" ]; then
    warning "Experiment at $experimentpath doesn't exist."
    return 1
fi

if is_running spearmint.main; then
    warning "Main spearmint program is already running. To restart, please run stop.sh first."
    return 1
fi

# Make sure that directories exist
logpath="logs/"
mkdir -p $logpath

# Start Spearmint
nohup python -m spearmint.main $experimentpath >> $logpath/spearmint.log 2>&1 &
