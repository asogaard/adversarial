#!/bin/bash
# Start spearmint program

# Import utility methods
source scripts/utils.sh

# Experiment directory
experiment="optimisation/experiments/classifier/"

if [ -d "$experiment" ]; then

  # Check whether Spearmint program(s) are already running
  if [[ ! -z "$(ps -u `whoami` | grep spearmint | grep -v grep)" ]]; then
    print "Spearmint program(s) are already running. To restart, please run stop.sh first."
    return 1
  fi

  # Make sure that directories exist
  logpath="logs/"
  mkdir -p $logpath

  # Start Spearmint
  # @TODO: Generalise experiment
  nohup python -m spearmint.main $experiment >> $logpath/spearmint.log 2>&1 &
  return 0

else
  warning "Experiment $experiment doesn't exist."
  return 1
fi
