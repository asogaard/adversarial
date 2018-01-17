#!/bin/bash
# Clean up optimisation directory

# Import utility methods
source scripts/utils.sh

# Define variable(s)
dir="optimisation"

# Check(s)
if [ ! -d "$dir" ]; then
    warning "Directory $dir/ doesn't exist."
    return 1
fi

# Platform-dependent flags
xargs_flags="-r"
if [[ "$(uname)" == *"Darwin"* ]]; then
    xargs_flags=""
fi

# Perform clean-up
find $dir -name '*.pyc' | xargs $xargs_flags rm
find $dir -name '*~'    | xargs $xargs_flags rm
