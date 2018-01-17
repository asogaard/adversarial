#!/bin/bash
# Start MongoDB server and Spearmint instance

# Import utility methods
source scripts/utils.sh

# Utility script to check whether a named process is running
function is_running () {
    pgrep "$1" > /dev/null
}

programa="mongod"
programb="keepalive"

if is_running "$programa" && is_running "$programb"; then
    echo "$programa AND $programb are running"
else
    if is_running "$programa"; then
        echo "$programa is running"
    else
        echo "$programa is NOT running"
    fi
    if is_running "$programb"; then
        echo "$programb is running"
    else
        echo "$programb is NOT running"
    fi
fi
