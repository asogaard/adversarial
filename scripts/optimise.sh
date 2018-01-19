#!/bin/bash
# Script to submit optimisation job

# Import general utility methods
source scripts/utils.sh

# Define variable(s)
experiment="$1"

# Check(s)
if [ -z "$experiment" ]; then
    warning "start_spearmint: Please provide an experiment name, i.e."
    warning " $ source path/to/optimise.sh experiment"
    return 1
fi

# Check(s)
if is_running mongod; then
    warning "A MongoDB server is already running. Please stop it first."
    return 1
fi

if is_running spearmint; then
    warning "Spearmint progam(s) are already running. Please stop it first."
    return 1
fi

# Clear history
question "Clear all history before starting optimisation?" "n"
response="$?"
if (( "$response" )); then
    print "Clearing..."
    source optimisation/scripts/clear.sh
fi
echo -e -n "\033[0m"

# Start MongoDB server
source optimisation/scripts/start_mongo.sh

# Remove pending jobs
question "Remove pending jobs before proceeding?" "y"
response="$?"
if (( "$response" )); then
    print "Repairing..."
    source optimisation/scripts/repair.sh $experiment
else
    num_pending_jobs="$(mongo --eval "var experiment='$experiment'" optimisation/scripts/get_pending.js | grep 'Pending job' | wc -l)"
    print "Starting with $num_pending_jobs pending jobs."
fi

# Start Spearmint
source optimisation/scripts/start_spearmint.sh $experiment
if [[ ! "$?" ]]; then
    warning "Exiting."
    return 1
fi

#### # Keep both processes alive
#### nohup ./optimisation/scripts/keepalive.sh $experiment >> logs/keepalive.log 2>&1 &

print "Here we go!"
