#!/bin/bash   

# Data staging job, moving results of 
# run from (transient) Eddie scratch disk
# to (permantent) DataStore to 
# --------------------------------------

# Settings
# ------------------
# Set name of job
#$ -N stageout
#
# Set to use current working directory
#$ -cwd
#
# Wait for run job to finish
#$ -hold_jid run

# Requires working on staging nodes
#$ -q staging 

# Send mail to these users
#$ -M andreas.sogaard@ed.ac.uk
#
# Mail at beginning/end/on suspension
#$ -m bes
 
# Script
# ------------------
# Synchronise from Eddie scratch disc to DataStore
echo "Synchronising files from"
echo "  ${SOURCE}"
echo "to"
echo "  ${DESTINATION}"
rsync -rl -vv ${SOURCE} ${DESTINATION}

# Check that all went well
if [ "$?" -eq "0" ]; then
    echo "Done, all went well"
else
    echo "Something went wrong"
    #exit 100
fi
