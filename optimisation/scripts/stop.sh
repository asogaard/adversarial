#!/bin/bash
# Start MongoDB server and Spearmint instance

# Import utility methods
source scripts/utils.sh

# Utility script to check whether a named process is running
function is_running () {
    [ ! -z "$(ps -u `whoami` -F | grep $1 | grep -v grep)" ]
}

# Base directory
dir="optimisation"

if [ -d "$dir" ]; then

    # Platform-dependent flags
    xargs_flags="-r"
    if [[ "$(uname)" == *"Darwin"* ]]; then
        xargs_flags=""
    fi

    IFS=' ' read -r -a array <<< "$(ps -u `whoami` -F | head -1)"
    pid_field=1
    counter=0
    for i in "${array[@]}"; do
        if [[ "$i" == "PID" ]]; then
            pid_field=$counter
            break
        fi
        let counter=counter+1
    done

    if [ ! -z "$(ps -u `whoami` -F | grep mongod | grep -v grep)" ] && [ ! -z "$(ps -u `whoami` -F | grep spearmint | grep -v grep)" ]; then
        warning "MongoDB server and Spearmint instance(s) are currently running."
        question "Do you want to stop both?" "n"
        response="$?"
        if (( "$response" )); then
            print "OK, proceeding"
        else
            print "OK, exiting"
            return 1
        fi
    fi

    # Kill 'keepalive' process
    if [[ -z "$(ps -u `whoami` -F | grep keepalive | grep -v grep)" ]]; then
        print "No running 'keepalive' script to stop."
    else
        IFS=' ' read -r -a array <<< "$(ps -u `whoami` -F | grep keepalive | grep -v grep)"
        kill "${array[$pid_field]}"
    fi

    # Kill running MongoDB server
    if [[ -z "$(ps -u `whoami` -F | grep mongod | grep -v grep)" ]]; then
        print "No running MongoDB servers to stop."
    else
        IFS=' ' read -r -a array <<< "$(ps -u `whoami` -F | grep mongod | grep -v grep)"
        kill "${array[$pid_field]}"
    fi

    # Kill running Spearmint instance
    if [[ -z "$(ps -u `whoami` -F | grep spearmint | grep -v grep)" ]]; then
        print "No running Spearmint instances to stop."
    else
        while [[ ! -z "$(ps -u `whoami` -F | grep spearmint | grep -v grep)" ]]; do
            IFS=' ' read -r -a array <<< "$(ps -u `whoami` -F | grep spearmint | grep -v grep)"
            kill "${array[$pid_field]}"
        done
    fi

else
    warning "Directory $dir/ doesn't exist."
    return 1
fi
