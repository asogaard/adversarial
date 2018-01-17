#!/bin/bash
# Repair MongoDB database

# Import utility methods
source scripts/utils.sh

# Base directory
dir="optimisation"
experiment="classifier"
# @TODO: Generalise experiment

if [ -d "$dir" ]; then

    # Check whether MongoDB server is running
    if [ -z "$(ps -u `whoami` -F | grep mongod | grep -v grep)" ]; then
        warning "No MongoDB served is currently running. Please restart before repairing."
        return 1
    fi

    # Check whether Spearmint is running
    if [[ ! -z "$(ps -u `whoami` -F | grep spearmint.main | grep -v grep)" ]]; then
        warning "Spearmint is already running. Please stop before repairing."
        return 1
    fi

    # Remove TensorBoard logs corresponding to pending/stalled jobs
    pending_ids="$(mongo $dir/scripts/get_pending.js | grep 'Pending job' | sed 's/.* //g')"
    for pending_id in ${pending_ids[@]}; do
        file="$(ls -d logs/tensorboard/$experiment-patch.*/ | grep -E \.[0]+$pending_id/)"
        num_files="$(echo $file | wc -l)"
        if (( $num_files == 0)); then
            warning "No TensorBoard logfile associated with stalled Spearmint job with ID $pending_id."
        elif (( $num_files > 1 )); then
            warning "More than one ($numfiles) TensorBoard logfiles associated with stalled Spearmint job with ID $pending_id. Not deleting any."
        else
            print "Deleting TensorBoard logfile $file associated with stalled Spearmint job with ID $pending_id."
            rm -rf $file
        fi

    done

    # Run JS repair script on MongoDB server
    mongo $dir/scripts/repair.js
    return "$?"

else
    warning "Directory $dir/ doesn't exist."
    return 1
fi
