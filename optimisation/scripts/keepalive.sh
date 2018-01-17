#!/bin/bash
# Keep MongoDB server and Spearmint script alive

# Import utility methods
source scripts/utils.sh

# Base directory
dir="optimisation"

if [ -d "$dir" ]; then

  # Check whether a MongoDB server is already running
  if [[ -z "$(ps -u `whoami` -F | grep mongod | grep -v grep)" ]]; then
    print "No MongoDB server is running. Please start it first."
    return 1
  fi

  if [[ -z "$(ps -u `whoami` -F | grep spearmint.main | grep -v grep)" ]]; then
    print "No Spearmint script is running. Please start it first."
    return 1
  fi

  # Starting commands
  cmd_mongo="source optimisation/scripts/start_mongo.sh"
  cmd_spearmint="source optimisation/scripts/start_spearmint.sh"

  while true; do
    if [[ -z "$(ps -u `whoami` -F | grep mongod | grep -v grep)" ]] && [[ -z "$(ps -u `whoami` -F | grep spearmint.main | grep -v grep)" ]]; then
      warning "MongoDB and Spearmint are both down. Restarting."
      echo "Calling: $cmd_mongo"
      echo "Calling: $cmd_spearmint"
      $cmd_mongo
      $cmd_spearmint
    fi
    sleep 1
  done

else
  warning "Directory $dir/ doesn't exist."
  return 1
fi
