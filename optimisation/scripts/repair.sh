#!/bin/bash
# Repair MongoDB database

# Import utility methods
source scripts/utils.sh

# Base directory
dir="optimisation"

if [ -d "$dir" ]; then

  # Check whether MongoDB server is running
  if [ -z "$(ps -u `whoami` -F | grep mongod | grep -v grep)" ]; then
    print "No MongoDB served is currently running. Please restart before repairing."
    return 1
  fi
  
  # Check whether Spearmint is running
  if [[ ! -z "$(ps -u `whoami` -F | grep spearmint.main | grep -v grep)" ]]; then
    print "Spearmint is already running. Please stop before repairing."
    return 1
  fi

  # Run JS repair script on MongoDB server
  mongo $dir/scripts/repair.js
  return "$?"

else
  warning "Directory $dir/ doesn't exist."
  return 1
fi
