#!/bin/bash
# Small script to setup the environment 
# necessary for running the adversarial 
# neural network training an evaluation.
MODE=$1
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
return 0
