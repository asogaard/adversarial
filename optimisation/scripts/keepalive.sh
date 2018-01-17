#!/bin/bash
# Keep MongoDB server and Spearmint script alive

# Import utility methods
source scripts/utils.sh

# Define variable(s)
experiment="$1"

# Check(s)
if [ -z "$experiment" ]; then
    warning "start_spearmint: Please provide an experiment name, i.e."
    warning " $ source path/to/keepalive.sh <experiment>"
    exit 1
fi

if ! is_running mongod; then
    warning "No MongoDB server is running. Please start it first."
    exit 1
fi

if ! is_running spearmint.main; then
    warning "No Spearmint script is running. Please start it first."
    exit 1
fi

# Starting commands
cmd_mongo="source optimisation/scripts/start_mongo.sh"
cmd_spearmint="source optimisation/scripts/start_spearmint.sh $experiment"

while true; do
    if ! is_running mongod && ! is_running spearmint.main; then
        warning "MongoDB and Spearmint are both down. Restarting."
        echo "Calling: $cmd_mongo"
        echo "Calling: $cmd_spearmint"
        $cmd_mongo
        $cmd_spearmint
    fi
    sleep 1
done
