#!/bin/bash
# Start MongoDB server

# Import utility methods
source scripts/utils.sh

# Base directory
dir="optimisation"

if [ -d "$dir" ]; then

  # Check whether a MongoDB server is already running
  if [[ ! -z "$(ps -u `whoami` | grep mongod | grep -v grep)" ]]; then
    print "A MongoDB server is already running. To restart, please run stop.sh first."
    return 1
  fi

  # Make sure that directories exist
  logpath="$dir/logs"
  dbpath="$dir/db"

  mkdir -p $logpath
  mkdir -p $dbpath

  # Start server
  mongod_arguments="--fork --logpath $logpath/mongo.log --dbpath $dbpath/"
  mongod $mongod_arguments

  # Check return code
  ret="$?"
  if (( "$ret" )); then

    # Try to repair
    warning "Recieved return code $ret. Trying to repair server."
    source $dir/scripts/repair.sh

    # Check if successful
    ret="$?"
    if (( "$ret" )); then
      return "$ret"
    else

      # Try to start server again
      print "Trying to start server again."
      mongod $mongod_arguments

      # Check if successful
      ret="$?"
      if (( "$ret" )); then
        warning "Restart was unsuccessful."
        return "$ret"
      fi

    fi

  fi

  # Perform cleanup
  source $dir/scripts/cleanup.sh

  print "Remember to run spearmint.main yourself!"
  return 0

else
  warning "Directory $dir/ doesn't exist."
  return 1
fi
