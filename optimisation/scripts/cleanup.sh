#!/bin/bash
# Clean up optimisation directory

# Import utility methods
source scripts/utils.sh

# Base directory
dir="optimisation"

if [ -d "$dir" ]; then

  if [ -d "$dir/experiments" ]; then

    # Platform-dependent flags
    xargs_flags="-r"
    if [[ "$(uname)" == *"Darwin"* ]]; then
      xargs_flags=""
    fi

    # Perform actual clean-up
    find $dir -name '*.pyc' | xargs $xargs_flags rm
    find $dir -name '*~'    | xargs $xargs_flags rm
  else
    warning "Directory $dir/ exists, but has no subdirectory $dir/experiment/."
    return 1
  fi

else
  warning "Directory $dir/ doesn't exist."
  return 1
fi
