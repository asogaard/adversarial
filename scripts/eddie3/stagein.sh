#!/bin/bash   

# Data staging job, synchronising data 
# from (permantent) DataStore to 
# (transient) Eddie scratch disk
# --------------------------------------

# Settings
# ------------------
# Set name of job
#$ -N stagein 
#
# Set to use current working directory
#$ -cwd

# Make the job resubmit itself if it runs out of time: rsync will start where it left off
#$ -r yes
#$ -notify
trap 'exit 99' sigusr1 sigusr2 sigterm

# Send mail to these users
#$ -M andreas.sogaard@ed.ac.uk
#
# Mail at beginning/end/on suspension
#$ -m bes

# Script
# ------------------
# Synchronise from EOS to Eddie scratch disc
echo "Synchronising files from"
echo "  ${SOURCE}"
echo "to"
echo "  ${DESTINATION}"
rsync -rl -vv -e ssh ${SOURCE} ${DESTINATION}

# Check that all went well (e.g. that lxplus node could be accessed)
if [ "$?" -eq "0" ]; then
    echo "Done, all went well"
else
    echo "+----------------------+"
    echo "| Something went wrong |"
    echo "+----------------------+"
    #exit 100
fi
