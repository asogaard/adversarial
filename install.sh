#!/bin/bash
# Install conda environments

# Import general utility methods
source scripts/utils.sh

# Import installation-specific utility methods
source scripts/install_utils.sh

# Check whether conda is installed
if ! hash conda 2>/dev/null; then
    print "conda was not installed."
    question "Do you want to do it now?"
    response="$?"
    if (( "$response" )); then

	# Install miniconda
	install_conda

	# Check if installation succeeded.
	response="$?"
	if (( "$response" )); then
	    return 1
	fi
	
    else
	warning "Please install conda manually, see e.g. https://github.com/asogaard/adversarial. Exiting."
	return 1
    fi
fi

# Set necessary flags
# --
# Used when compiling scipy.weave (from Spearmint) -- possibly also Theano -- to
# avoid compilation error. See also StackOverflow question linked in
# scripts/install_utils.sh
CFLAGS=-D__USE_XOPEN2K8  

# Environment names
env_cpu="adversarial-cpu"
env_gpu="adversarial-gpu"

if   [[ "$(uname)" == *"Linux"* ]]; then
    envfolder="linux"
elif [[ "$(uname)" == *"Darwin"* ]]; then
    envfolder="macOS"
else
    warning "Uname '$(uname)' not recognised. Exiting."
    return 1
fi

# Install CPU environment
create_env $env_cpu envs/$envfolder/$env_cpu.yml
create_env $env_gpu envs/$envfolder/$env_gpu.yml
