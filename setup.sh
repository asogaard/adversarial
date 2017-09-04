#!/bin/bash
# Small script to setup the environment 
# necessary for running the adversarial 
# neural network training an evaluation.

# @NOTE: Improve; 'if "test" in args'-type thing?
MODE=$1
TEST=$2
if [ "$MODE" == "test" ]; then
    MODE=""
    TEST="test"
fi

# Deactivate any current environments
if [ ! -z "$(python -c "import sys; print sys.version" 2>/dev/null | grep "Continuum\|conda")" ]; then
    source deactivate
fi

# Unload any existing modules on Eddie
LOADED_MODULES="$(module list 2>&1 | grep -v "Currently Loaded" | sed 's/ //g' | sed 's/[0-9]*)/\n/g' | grep -v "^$" | sed 's/\/.*//g')"
read -r -a LOADED_MODULES <<< $LOADED_MODULES
for MODULE in "${LOADED_MODULES[@]}"; do
    module unload $MODULE
done

# Set up appropriate environment
if [ "$MODE" == "cpu" ] || [ "$MODE" == "gpu" ] || [ "$MODE" == "" ]; then
    if [ "$MODE" == "" ]; then
	echo "Using cpu by default."
	MODE="cpu"
    fi
    module load anaconda cuda/8.0.61 root/6.06.02
    source activate adversarial-$MODE
else
    echo "Command-line argument '$1' is not either 'cpu' or 'gpu'."
    return 1
fi

# Set up paths for interactive testing
if [ ! -z "$TEST" ]; then
    export INPUTDIR="/exports/eddie/scratch/s1562020/adversarial/data/2017-08-25-ANN/"
    export OUTPUTDIR="/exports/eddie/scratch/s1562020/adversarial/output/test/"
fi

return 0
