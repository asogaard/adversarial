#!/bin/bash
# Clean up optimisation directory

# Import utility methods
source scripts/utils.sh

# Platform-dependent flags
xargs_flags="-r"
if [[ "$(uname)" == *"Darwin"* ]]; then
    xargs_flags=""
fi

# Perform clean-up
find . -name '*.pyc' | xargs $xargs_flags rm
find . -name '*~'    | xargs $xargs_flags rm
