#!/bin/bash
# Start MongoDB server

# Import utility methods
source scripts/utils.sh

# Define variable(s)
dir="optimisation"

# Check(s)
if [ ! -d "$dir" ]; then
    warning "Directory $dir/ doesn't exist."
    return 1
fi

if is_running mongod; then
    warning "A MongoDB server is already running. To restart, please run stop.sh first."
    return 1
fi

# Make sure that directories exist
logpath="logs"
dbpath="$dir/db"

mkdir -p $logpath
mkdir -p $dbpath

# Run standard database repair
mongod --repair --logpath $logpath/mongo.log --logappend --dbpath $dir/db/ --quiet 2>&1

# Start server with manual forking
nohup mongod --logpath $logpath/mongo.log --logappend --dbpath $dbpath/ >> $logpath/mongo.log 2>&1 & 

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

# Perform cleanup
source $dir/scripts/cleanup.sh
