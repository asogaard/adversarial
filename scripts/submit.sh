#!/bin/bash

# Get command-line arguments
# ------------------------------------------------------------------------------

# Default value(s)
HELP=""
GPU=false
TRAIN=false
TENSORFLOW=false
DEVICES=""
FOLDS=""
CONFIG=""
PATCH=""
TAG=""
USERNAME=asogaard

# From [https://stackoverflow.com/a/14203146]
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -h|--help)
    HELP=true
    break
    ;;
    --gpu)
    GPU=true
    ;;
    --train)
    TRAIN=true
    ;;
    --tensorflow)
    TENSORFLOW=true
    ;;
    --config)
    CONFIG="$2"
    shift # past argument
    ;;
    --devices)
    DEVICES="$2"
    shift # past argument
    ;;
    --folds)
    FOLDS="$2"
    shift # past argument
    ;;
    --patch)
    # Allow for more than one '--patch <path>' argument; trim delimiters
    PATCH="$(echo "$PATCH $2" | sed 's/^ *//g;s/ *$//g')" 
    shift # past argument
    ;;
    --tag)
    TAG="$2"
    shift # past argument
    ;;
    --username)
    USERNAME="$2"
    shift # past argument
    ;;
    *)
        # unknown option
	echo "Option '$key: $2' was not recognised."
    ;;
esac
shift # past argument or value
done

if [ ! -z "$HELP" ]; then
    echo "usage: ./scripts/submit.sh [--gpu] [--train] [--tensorflow] [--devices <num>] [--folds <num>] [--config <path>] [--patch <path>] [--patch <path>] ... [--tag <name>] [--username <name>]"
    echo "Options and arguments:"
    echo "  -h, --help        : print this help message and exit"
    echo "  --gpu             : run on GPU(s)"
    echo "  --train           : perform training"
    echo "  --tensorflow      : whether to use Tensorflow backend"
    echo "  --devices <num>   : number of CPU/GPU devices to use"
    echo "  --folds <num>     : number of cross-validation folds"
    echo "  --config <path>   : configuration file"
    echo "  --patch <path>    : patch applied the default configuration. May be used multiple times"
    echo "  --tag <name>      : unique tag for the job being submitted"
    echo "  --username <name> : lxplus username for rsync"
    exit
fi


# Initialisation
# ------------------------------------------------------------------------------

# Variables
KFILE=.kerberos/krb5cc_ticket
SSHLOG=.sshlog

# Check if valid Kerberos ticket exists
KTICKET=`klist -c $KFILE | grep /CERN.CH`
if [ -z "$KTICKET" ]; then
    echo "No valid Kerberos ticket was found. Please create one:"
    kinit -f $USERNAME@CERN.CH -c $KFILE || { echo "Incorrect password. Exiting"; exit; }
else
    echo "Found a valid Kerberos ticket:"
    echo "  $KTICKET"
fi

# Get named lxplus node to use
rm -f $SSHLOG
ssh -v $USERNAME@lxplus.cern.ch > $SSHLOG 2>&1 3>&1 &
PID="$!"

# -- Wait for the right time during ssh to extract node name
QUIT=false
PATIENCE=5
start=`date +%s`
while [ ! -f $SSHLOG ] || [ -z "$(cat $SSHLOG | grep "Connecting to lxplus.cern.ch")" ]; do
    sleep 0.1
    now=`date +%s`
    if (( $((now - start)) > $PATIENCE )); then
	echo "No progress in $PATIENCE seconds. Exiting"
	QUIT=true
	break
    fi
done

# -- Clean up after ssh
kill $PID
wait $PID 2>/dev/null

# -- Quit if no node name was found
if [ $QUIT == true ]; then { exit; }; fi

# -- Extract lxplus node name from ssh output
LXPLUS=`cat $SSHLOG | grep "Connecting to lxplus.cern.ch" | cut -d "[" -f2 | cut -d "]" -f1 | xargs host | sed 's/.*\(lxplus[0-9]*\).*/\1/g'`

if [ -z "$LXPLUS" ]; then
    echo "Got empty LXPLUS. Something went wrong in parsing the output from SSH."
    exit
else
    echo "Using lxplus node '$LXPLUS'"
fi

# Define common unique indicators
#VERSION=2017-06-22              # AnalysisTools outputObjdef cache
#VERSION=2017-08-25-ANN
VERSION=2017-09-08-ANN

# Tag to uniquely distinguish runs
if [ -z "$TAG" ]; then
    TAG="$(date +%Y-%m-%d_%H%M%S)"
else
    TAG="$(date +%Y-%m-%d)_$TAG" # To uniquely distinguish runs
fi

# Define common patch
EOS=asogaard@$LXPLUS.cern.ch:/eos/atlas/user/a/asogaard/Analysis/2016/BoostedJetISR/outputObjdef
#EOS=asogaard@$LXPLUS.cern.ch:/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysis2016/FlatNTuples/20170530/MLTrainingTesting
DATASTORE=/exports/csce/datastore/ph/groups/PPE/atlas/users/$USER/adversarial
SCRATCH=/exports/eddie/scratch/$USER/adversarial


# Submission
# ------------------------------------------------------------------------------

# Submit all jobs in the correct order with the necessary environment variables
qsub -v SOURCE="$EOS/$VERSION" \
     -v DESTINATION="$SCRATCH/data" \
     -v KRB5CCNAME="FILE:$KFILE" \
     scripts/eddie3/stagein.sh

qsub -v INPUTDIR="$SCRATCH/data/$VERSION" \
     -v OUTPUTDIR="$SCRATCH/output/$TAG" \
     -v GPU="$GPU" \
     -v TRAIN="$TRAIN" \
     -v TENSORFLOW="$TENSORFLOW" \
     -v DEVICES="$DEVICES" \
     -v FOLDS="$FOLDS" \
     -v CONFIG="$CONFIG" \
     -v PATCH="$PATCH" \
     scripts/eddie3/run.sh

qsub -v SOURCE="$SCRATCH/output/$TAG" \
     -v DESTINATION="$DATASTORE/output" \
     scripts/eddie3/stageout.sh
