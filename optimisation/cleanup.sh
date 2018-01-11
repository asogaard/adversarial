#!/bin/bash
# Clean up optimisation directory

dir="optimisation"

if [ -d "$dir" ]; then
    if [ -d "$dir/experiments" ]; then
	rm -rf $dir/experiments/*/output
	rm -f $dir/experiments/*/*.pyc
	rm -f $dir/log.txt*
	rm -rf $dir/db/*
  find $dir -name *~ | xargs rm
	ps -u asogaard | grep mongod | grep -v grep | sed 's/^ *//g' | cut -d" " -f2 | xargs kill
    else
	echo "Directory $dir/ exists, but has no subdirectory $dir/experiment/."
    fi
else
    echo "Directory $dir/ doesn't exist."
    return
fi
