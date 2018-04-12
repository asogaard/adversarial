#!/bin/bash
# Activate the environment necessary for running the adversarial neural network
# training and evaluation, depending on the current host.

# Import utility methods
source scripts/utils.sh

# Extract flags
arguments=("$@")
set -- # Unsetting positional arguments, to avoid error from "source deactivate"

cpu=false
gpu=false
test=false
unset=false
lcg=false
for arg in "${arguments[@]}"; do
    if   [ "$arg" == "cpu" ];  then
        cpu=true
    elif [ "$arg" == "gpu" ];  then
        gpu=true
    elif [ "$arg" == "test" ]; then
        test=true
    elif [ "$arg" == "unset" ]; then
        unset=true
    elif [ "$arg" == "LCG" ] || [ "$arg" == "lcg" ]; then
        lcg=true
    else
        warning "Argument '$arg' was not understood."
	return 1
    fi
done

# Determine host
if   [[ "$(hostname)" == *"lxplus"* ]]; then
    host="lxplus"
elif [[ "$(hostname)" == *"ed.ac.uk"* ]]; then
    host="eddie3"
else
    host="local"
fi

# Actual setting up
if [[ "$lcg" == true ]]; then

    if [[ "$host" != "lxplus" ]]; then
	warning "Cannot activate LCG environment on $host platform. Exiting."
	return 1
    else
	print "Activating LCG environment on $host"
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
    if [[ "$(conda info --env | grep \* | sed 's/ .*//g')" != "root" ]]; then
	source deactivate > /dev/null 2>&1
    fi

    # Unset complete
    if [[ "$unset" == true ]]; then
	return 0
    fi

    # Determine running mode (CPU/GPU)
    mode="cpu"
    if   [ "$cpu" == false ] && [ "$gpu" == true ]; then

	mode="gpu"
	if ! hash nvidia-smi 2>/dev/null; then
            warning "Requesting GPUs on a node that doesn't have any. Exiting."
            return 1
	fi

	# Setup Cuda/CuDNN
	if [[ "$host" == "eddie3" ]]; then
	    print "Loading Cuda module"
	    module unload cuda  # Fails silently
	    module load cuda/8.0.61
	else
	    warning "GPU mode requested. Make sure you have Cuda/CuDNN installed"
	fi

    elif [ "$cpu" == "$gpu" ]; then
	print "Using CPU by default"
    fi

    # Set `MKL_THREADING_LAYER` environment variable, necessary for running
    # Theano 1.0.0rc1
    export MKL_THREADING_LAYER=GNU

    # Activate appropriate conda environment
    env="adversarial-$mode"
    print "Activating conda environment '$env' on $host platform"
    env_exists="$(conda info --env | sed 's/ .*//g;s/^#//g' | grep $env)"
    if [[ "$env_exists" ]]; then
	source activate $env > /dev/null 2>&1
    else
	warning "Conda environment '$env' does not exist. Please run the installation script."
    fi
fi

# Set necessary flag(s)
export CFLAGS=-D__USE_XOPEN2K8:$CFLAGS

# Enable auto-complete for command-line arguments
source scripts/autocomplete.sh
