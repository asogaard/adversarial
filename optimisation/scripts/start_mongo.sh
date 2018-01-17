#!/bin/bash
# Start MongoDB server

# Import utility methods
source scripts/utils.sh

# Base directory
dir="optimisation"

if [ -d "$dir" ]; then

    # Check whether a MongoDB server is already running
    if [[ ! -z "$(ps -u `whoami` | grep mongod | grep -v grep)" ]]; then
        warning "A MongoDB server is already running. To restart, please run stop.sh first."
        return 1
    fi

    # Make sure that directories exist
    logpath="logs/"
    dbpath="$dir/db"

    mkdir -p $logpath
    mkdir -p $dbpath

    # Start server
    mongod_arguments="--fork --logpath $logpath/mongo.log --dbpath $dbpath/"
    mongod $mongod_arguments

    # Check return code
    ret="$?"
    if (( "$ret" )); then
        warning "Recieved return code $ret."

        if (( "$ret" == 100 )); then
            rm -f $dir/db/mongod.lock

            # Try to start server again
            print "Try to start server again."
            return 1
        fi

    fi

    # Perform repair
    #source $dir/scripts/repair.sh

    # Perform cleanup
    source $dir/scripts/cleanup.sh

    print "Remember to run spearmint.main yourself!"
    return 0

else
    warning "Directory $dir/ doesn't exist."
    return 1
fi
