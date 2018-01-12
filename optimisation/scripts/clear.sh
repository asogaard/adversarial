#!/bin/bash
# Clear optimisation directory

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

    # Check with user before proceeding
    question "This is going to clear all history in $dir. Do you want to proceed?" "n"
    response="$?"
    if (( "$response" )); then

      # Stop running processes
      source $dir/scripts/stop.sh
      ret="$?"
      if (( "$ret" )); then
        warning "Something went wrong in 'stop.sh' script. Exiting."
        return "$ret"
      fi

      # Perform regular clean-up
      source $dir/scripts/cleanup.sh
      ret="$?"
      if (( "$ret" )); then
        warning "Something went wrong in 'cleanup.sh' script. Exiting."
        return "$ret"
      fi

      # Perform actual cleaning
      rm -rf $dir/experiments/*/output
      rm -f $dir/logs/log.txt*
      rm -rf $dir/db/*
    else
      print "OK, exiting."
      return 0
    fi

  else
    warning "Directory $dir/ exists, but has no subdirectory $dir/experiment/."
    return 1
  fi

else
  warning "Directory $dir/ doesn't exist."
  return 1
fi
