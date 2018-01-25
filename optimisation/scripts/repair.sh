#!/bin/bash
# Repair MongoDB database

# Import utility methods
source scripts/utils.sh

# Define variable(s)
dir="optimisation"
experiment="$1"

# Check(s)
if [ -z "$experiment" ]; then
    warning "start_spearmint: Please provide an experiment name, i.e."
    warning " $ source path/to/repair.sh <experiment>"
    return 1
fi

if [ ! -d "$dir" ]; then
    warning "Directory $dir/ doesn't exist."
    return 1
fi

if ! is_running mongod; then
    warning "No MongoDB served is currently running. Please restart before repairing."
    return 1
fi

if is_running spearmint.main; then
    warning "Spearmint is already running. Please stop before repairing."
    return 1
fi

# Remove TensorBoard logs corresponding to pending/stalled jobs
pending_ids="$(mongo --eval "var experiment='$experiment'" $dir/scripts/get_pending.js | grep 'Pending job' | sed 's/.* //g')"
for pending_id in ${pending_ids[@]}; do
    file="$(ls -d logs/tensorboard/$experiment-patch.*/ | grep -E \.[0]+$pending_id/)"
    num_files="$(echo $file | wc -l)"
    if (( $num_files == 0)); then
        warning "No TensorBoard logfile associated with stalled Spearmint job with ID $pending_id."
    elif (( $num_files > 1 )); then
        warning "More than one ($numfiles) TensorBoard logfiles associated with stalled Spearmint job with ID $pending_id. Not deleting any."
    else
        question "Delete TensorBoard logfile $file associated with stalled Spearmint job with ID $pending_id." "n"
	response="$?"
	if (( "$response" )); then
	    print "OK, deleting."
            rm -rf $file
	else
	    print "OK, proceeding."
	fi    
    fi
done

# Run JS repair script on MongoDB server
mongo --eval "var experiment='$experiment'" $dir/scripts/repair.js
ret="$?"

return $ret
