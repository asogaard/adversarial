#!/bin/bash
# Small script to setup the environment 
# necessary for running the adversarial 
# neural network training an evaluation.

# Extract flags
ARGUMENTS=("$@")
set -- # Unsetting positional arguments, to avoid error from "source deactivate"

CPU=false
GPU=false
TEST=false
UNSET=false
for ARG in "${ARGUMENTS[@]}"; do
    if   [ "$ARG" == "cpu" ];  then
	CPU=true
    elif [ "$ARG" == "gpu" ];  then
	GPU=true
    elif [ "$ARG" == "test" ]; then
	TEST=true
    elif [ "$ARG" == "unset" ]; then
	UNSET=true
    else
	echo "Argument '$ARG' was not understood"
    fi
done

# Deactivate any current environments
# -- Check if command 'conda' exists
# -- Check whether an environment is activated
if hash conda 2>/dev/null && [ "$(conda info --envs | grep \* | grep -v root)" ]
then
    source deactivate
fi

# Unload any existing modules on Eddie
module unload $(module list 2>&1 | tail -n +2 | sed 's/ *[0-9]*) \([a-z]*\)\/[0-9._a-z]*/\1 /g' | sed 's/sge *//g')

if [ "$UNSET" == true ]; then
    return
fi

# Set up appropriate environment
MODE="cpu"
if   [ "$CPU" == false ] && [ "$GPU" == true ]; then
    MODE="gpu"
    if ! hash nvidia-smi 2>/dev/null; then
	echo "WARNING: Requesting GPUs on a node that doesn't have any. Exiting."
	return 1
    fi
elif [ "$CPU" == "$GPU" ]; then
    echo "Using CPU by default"
fi
module load anaconda cuda/8.0.61 root/6.06.02
source activate adversarial-$MODE

# Set up paths for interactive testing
if [ "$TEST" == true ]; then
    echo "Setting I/O environment variables for interactive testing"
    export INPUTDIR="/exports/eddie/scratch/s1562020/adversarial/data/2017-08-25-ANN/"
    export OUTPUTDIR="/exports/eddie/scratch/s1562020/adversarial/output/test/"
fi

return
