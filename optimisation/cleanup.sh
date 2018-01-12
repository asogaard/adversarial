#!/bin/bash
# Clean up optimisation directory

dir="optimisation"

if [ -d "$dir" ]; then
    if [ -d "$dir/experiments" ]; then
	# Platform-dependent: Get index if `pid` field in `ps -u ...`, correct `xargs` flag(s).
	pid_field=1
	xargs_flags="-r"
	if [[ "$(uname)" == *"Darwin"* ]]; then
	    pid_field=2
	    xargs_flags=""
	fi

	# Perform actual clean-up
	rm -rf $dir/experiments/*/output
	rm -f $dir/experiments/*/*.pyc
	rm -f $dir/log.txt*
	rm -rf $dir/db/*
	find $dir -name *~ | xargs $xargs_flags rm
	ps -u `whoami` | grep mongod | grep -v grep | sed 's/^ *//g' | cut -d" " -f$pid_field | xargs $xargs_flags kill
    else
	echo "Directory $dir/ exists, but has no subdirectory $dir/experiment/."
    fi
else
    echo "Directory $dir/ doesn't exist."
    return
fi
