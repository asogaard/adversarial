#!/bin/bash
# Start MongoDB server and Spearmint instance

# Import utility methods
source scripts/utils.sh

# Get user confirmation
if is_running mongod && is_running spearmint; then
    warning "MongoDB server and Spearmint instance(s) are currently running."
    question "Do you want to stop both?" "n"
    response="$?"
    if (( "$response" )); then
        print "OK, proceeding"
    else
        print "OK, exiting"
        return 0
    fi
fi

# Kill any running 'keepalive' processes, MongoDB servers, and Spearmint
# (-spawned) instances
try_kill keepalive
try_kill spearmint
try_kill mongod
return 0
