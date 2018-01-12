#!/bin/bash
# Clean up optimisation directory

dir="optimisation"

if [ -d "$dir" ]; then
    if [ -d "$dir/experiments" ]; then
	rm -rf $dir/experiments/*/output
	rm -f $dir/experiments/*/*.pyc
	rm -f $dir/log.txt*
	rm -rf $dir/db/*
	find $dir -name *~ | xargs -r rm
	
	# Get index if `pid` field in `ps -u ...`
	pid_field=1
	if [ "$(uname)" == *"Darwin"* ]; then
	    pid_field=2
	fi
	
	ps -u `whoami` | grep mongod | grep -v grep | sed 's/^ *//g' | cut -d" " -f$pid_field | xargs -r kill
    else
	echo "Directory $dir/ exists, but has no subdirectory $dir/experiment/."
    fi
else
    echo "Directory $dir/ doesn't exist."
    return
fi
