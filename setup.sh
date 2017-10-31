#!/bin/bash
# Setup the environment necessary for running the adversarial neural network
# training and evaluation, depending on the current host.

# Import utility methods
source scripts/utils.sh

# Extract flags
ARGUMENTS=("$@")
set -- # Unsetting positional arguments, to avoid error from "source deactivate"

CPU=true
GPU=false
TEST=false
UNSET=false
LCG=false
for ARG in "${ARGUMENTS[@]}"; do
    if   [ "$ARG" == "cpu" ];  then
        CPU=true
    elif [ "$ARG" == "gpu" ];  then
        GPU=true
    elif [ "$ARG" == "test" ]; then
        TEST=true
    elif [ "$ARG" == "unset" ]; then
        UNSET=true
    elif [ "$ARG" == "LCG" ] || [ "$ARG" == "lcg" ]; then
        LCG=true
    else
        print "Argument '$ARG' was not understood."
    fi
done

# Determine host
if   [[ "$HOSTNAME" == *"lxplus"* ]]; then
    HOST="lxplus"
elif [[ "$HOSTNAME" == *"ed.ac.uk"* ]]; then
    HOST="eddie3"
else
    HOST="local"
fi

# Actual setting up
if [[ "$LCG" == true ]]; then
    if [[ "$HOST" != "lxplus" ]]; then
	warning "Cannot setup LCG environment on $HOST platform. Exiting."
	return 1
    else
	print "Setting up LCG environment on $HOST"
	source scripts/lxplus/lcg.sh
	return 0
    fi
else
    # Check that installation was performed.
    if ! hash conda 2>/dev/null; then
	warning "conda was not installed. Please run the 'install.sh' script. Exiting."
	return 1
    fi

    # Deactivate conda environment
    if [[ "$UNSET" == true ]]; then
	if [[ "$(conda info --env | grep \* | sed 's/ .*//g')" != "root" ]]; then
	    source deactivate
	fi
	return 0
    fi

    # Determine running mode (CPU/GPU)
    MODE="cpu"
    if   [ "$CPU" == false ] && [ "$GPU" == true ]; then
	MODE="gpu"
	if ! hash nvidia-smi 2>/dev/null; then
            warning "Requesting GPUs on a node that doesn't have any. Exiting."
            return 1
	fi
    elif [ "$CPU" == "$GPU" ]; then
	print "Using CPU by default"
    fi

    # Activate appropriate conda environment
    ENV="adversarial-$MODE"
    print "Setting up conda environment '$ENV' on $HOST platform"
    ENV_EXISTS="$(conda info --env | sed 's/ .*//g;s/^#//g' | grep $ENV)"
    if [[ "$ENV_EXISTS" ]]; then
	source activate $ENV
    else
	warning "Conda environment '$ENV' does not exist. Please run the installation script."
    fi
fi

### # Host-specific setup
### if   [[ "$HOSTNAME" == *"lxplus"* ]]; then
###     source scripts/lxplus/setup.sh "$@"
### elif [[ "$HOSTNAME" == *"ed.ac.uk"* ]]; then
###     source scripts/eddie3/setup.sh "$@"
### else
###     echo "Host not recognised; unable to setup environment."
###     return
### fi

# Enable auto-complete for command-line arguments
source scripts/autocomplete.sh
