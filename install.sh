#!/bin/bash
# Install conda environments

# Import general utility methods
source scripts/utils.sh

# Import installation-specific utility methods
source scripts/install_utils.sh

# Utility function to resolve libstdc++ link problems
#### function fix_link {
#### 
####     # Define environment variable
####     ENV="$1"
#### 
####     # Validate input
####     if [ -z "$ENV" ]; then
#### 	warning "fix_link: No environment was specified"
#### 	return
####     fi
#### 
####     # Try to activate environment
####     print "  Activating '$ENV'."
####     source activate "$ENV" 2>&1 1>/dev/null
#### 
####     # Check if environment was succesfully activated
####     ENV_ACTIVE="$(conda info --envs | grep \* | sed 's/ .*//g')"
####     if [ "$ENV_ACTIVE" == "$ENV" ]; then
#### 
#### 	# Fix libstdsc++ symlink problem on linux platforms
#### 	if [[ "$(uname)" == *"Linux"* ]]; then
#### 	
#### 	    # Base directory of active environment
#### 	    ENVDIR="$(conda info --env | grep \* | sed 's/.* //g')"
#### 
#### 	    # Problematic symlink
#### 	    LINKPATH="$ENVDIR/lib/libstdc++.so.6"
#### 	    
#### 	    # Latest available libstdc++ library
#### 	    LATESTLIB="$(find $ENVDIR/lib/ -name libstdc++.* ! -type l | grep -v .py | sort | tail -1)"
#### 	    
#### 	    # Check that link exists
#### 	    if [ -L "$LINKPATH" ]; then
#### 		# Check whether link target is most latest available library
#### 		if [ "$(readlink -f $LINKPATH)" == "$LATESTLIB" ]; then
#### 		    # $LINKPATH already links to $LATESTLIB
#### 		    :
#### 		else
#### 		    # Try to update symlink target
#### 		    print "  Changing target of"
#### 		    print "    $LINKPATH"
#### 		    print "  to be"
#### 		    print "    $LATESTLIB"
#### 		    question "  Is that OK?" "y"
#### 		    response="$?"
#### 		    if (( $response )); then
#### 			ln -s -f $LATESTLIB $LINKPATH
#### 		    else
#### 			warning "  OK, not doing it, but be warned that errors might occur. You can always run the installation script again if you change your mind."
#### 		    fi
#### 		fi
#### 	    else
#### 		warning "  Symlink '$LINKPATH' doesn't exist."
#### 	    fi
#### 	    
#### 	fi
#### 	
#### 	# Deactivate environment
#### 	print "  Deactivating '$ENV_CPU'."
#### 	source deactivate 2>&1 1>/dev/null
####     else
#### 	warning "Failed to activate '$ENV_CPU'."
####     fi
#### }

# Check whether conda is installed
if ! hash conda 2>/dev/null; then
    print "conda was not installed."
    question "Do you want to do it now?"
    response="$?"
    if (( "$response" )); then

	install_conda
	
#### 	print "Installing Miniconda."
#### 
#### 	# Download install file
#### 	REPO="https://repo.continuum.io/miniconda"
#### 	if   [[ "$(uname)" == *"Linux"* ]]; then
#### 	    INSTALLFILE="Miniconda2-latest-Linux-x86_64.sh"
#### 	    wget $REPO/$INSTALLFILE
#### 	elif [[ "$(uname)" == *"Darwin"* ]]; then
#### 	    INSTALLFILE="Miniconda2-latest-MacOSX-x86_64.sh"
#### 	    curl -o ./$INSTALLFILE -k $REPO/$INSTALLFILE
#### 	else
#### 	    warning "Uname '$(uname)' not recognised. Exiting."
#### 	    return 1
#### 	fi
#### 
#### 	# Run installation
#### 	bash $INSTALLFILE
#### 
#### 	# Clean-up
#### 	rm -f $INSTALLFILE
#### 
#### 	# Check whether installation worked
#### 	if ! hash conda 2>/dev/null; then
#### 	    warning "conda wasn't installed properly. Perhaps something went wrong in the installation, or 'PATH' was not set? Exiting."
#### 	    return 1
#### 	else
#### 	    print "conda was installed succesfully!"
	#### 	fi
	
    else
	warning "Please install conda manually, see e.g. https://github.com/asogaard/adversarial. Exiting."
	return 1
    fi
fi

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
#### ENVFILE=envs/$env_cpu.yml

create_env $env_cpu envs/$envfolder/$env_cpu.yml
#create_env $env_gpu envs/$envfolder/$env_gpu.yml

#### if [ "$(conda info --envs | grep $env_cpu)" ]; then
####     print "Environment '$env_cpu' already exists"
####     
####     # Check consistency with baseline env.
####     print "  Checking consistency"
#### 
####     # Silently activate environment
####     source activate $env_cpu 2>&1 1>/dev/null 
#### 
####     # Write the enviroment specifications to file
####     TMPFILE=".tmp.env.txt"
####     conda env export > $TMPFILE
####     
####     # Compare current enviroment with default
####     DIFFERENCES="$(diff -y --left-column $TMPFILE $ENVFILE | grep -v "prefix:" | grep -v "(" | sed $'s/\t/    /g' )"
####     if (( "${#DIFFERENCES}" )); then
#### 	warning "  The existing '$env_cpu' env. differs from the default one in '$ENVFILE':"
#### 	POSINDEX="$(echo "$DIFFERENCES" | grep -b -o "|" | cut -d: -f1)"
#### 	printf "%-${POSINDEX}s| %s\n" "ACTIVE ENVIRONMENT" "DEFAULT ENVIRONMENT"
#### 	printf "%0.s-" $(seq 1 $(( 2 * $POSINDEX + 1)) )
#### 	echo ""
#### 	echo "$DIFFERENCES"
#### 	warning "  Beware that this might lead to problems when running the code."
####     fi
#### 
####     # Clean-up
####     rm -f $TMPFILE
####     
####     # Silently deactivate environment
####     source deactivate 2>&1 1>/dev/null
#### else
####     # Fix ROOT setup problem on macOS (1/2)
####     if [[ "$(uname)" == *"Darwin"* ]]; then
#### 	# The `activateROOT.sh` scripts for Linux and macOS are different, the
#### 	# later being broken. Therefore, we (1) need to set the `CONDA_ENV_PATH`
#### 	# environment variable before installation and (2) need to update
#### 	# `activateROOT.sh` after installation so as to not have to do this
#### 	# every time.
#### 	CONDA_ENV_PATH="$(which conda)"
#### 	SLASHES="${CONDA_ENV_PATH//[^\/]}"
#### 	CONDA_ENV_PATH="$(echo "$CONDA_ENV_PATH" | cut -d '/' -f -$((${#SLASHES} - 2)))"
#### 	CONDA_ENV_PATH="$CONDA_ENV_PATH/envs/$env_cpu"
#### 	export CONDA_ENV_PATH
#### 	print "Setting CONDA_ENV_PATH=$CONDA_ENV_PATH" # @TEMP
####     fi
####     
####     # Create environment
####     print "Creating CPU environment '$env_cpu'."
####     conda env create -f $ENVFILE
#### 
####     # Fix ROOT setup problem on macOS (2/2)
####     if [[ "$(uname)" == *"Darwin"* ]]; then
#### 	# Silently activate environment
#### 	source activate $env_cpu 2>&1 1>/dev/null
#### 	
#### 	# Check if environment was succesfully activated
#### 	ENV_ACTIVE="$(conda info --envs | grep \* | sed 's/ .*//g')"
#### 	if [ "$ENV_ACTIVE" == "$env_cpu" ]; then
#### 	    
#### 	    # Base directory of active environment
####             ENVDIR="$(conda info --env | grep \* | sed 's/.* //g')"
#### 	    
#### 	    # Locate ROOT activation file
#### 	    ROOTINITFILE="$ENVDIR/etc/conda/activate.d/activateROOT.sh"
#### 	    
#### 	    # Replace whatever statement is used to source 'thisroot.sh' by 'source
#### 	    # $(which thisroot)' to remove dependence on the 'CONDA_ENV_PATH'.
#### 	    sed -i.bak 's/\(^.*thisroot.sh\)/#\1\'$'\nsource $(which thisroot.sh)/g' $ROOTINITFILE
#### 	fi
####     fi
####     
####     # Silently deactivate environment
####     source deactivate 2>&1 1>/dev/null
#### fi
#### 
#### # Fix libstdc++ symblink problem
#### fix_link $env_cpu
#### print "Done!"
