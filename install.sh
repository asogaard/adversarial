#!/bin/bash
# Install conda environments

# Import utility methods
source scripts/utils.sh

# Utility function to resolve libstdc++ link problems
function fix_link {

    # Define environment variable
    ENV="$1"

    # Validate input
    if [ -z "$ENV" ]; then
	warning "fix_link: No environment was specified"
	return
    fi

    # Try to activate environment
    print "  Activating '$ENV'."
    source activate "$ENV" 2>&1 1>/dev/null

    # Check if environment was succesfully activated
    ENV_ACTIVE="$(conda info --envs | grep \* | sed 's/ .*//g')"
    if [ "$ENV_ACTIVE" == "$ENV" ]; then

	# Base directory of active environment
	ENVDIR="$(conda info --env | grep \* | sed 's/.* //g')"

	# Problematic symlink
	LINKPATH="$ENVDIR/lib/libstdc++.so.6"

	# Latest available libstdc++ library
	LATESTLIB="$(find $ENVDIR/lib/ -name libstdc++.* ! -type l | grep -v .py | sort | tail -1)"

	# Check that link exists
	if [ -L "$LINKPATH" ]; then
	    # Check whether link target is most latest available library
	    if [ "$(readlink -f $LINKPATH)" == "$LATESTLIB" ]; then
		# $LINKPATH already links to $LATESTLIB
		:
            else
		# Try to update symlink target
		print "  Changing target of"
		print "    $LINKPATH"
		print "  to be"
		print "    $LATESTLIB"
		question "  Is that OK?" "y"
		RESPONSE="$?"
		if (( $RESPONSE )); then
                    ln -s -f $LATESTLIB $LINKPATH
		else
                    warning "  OK, not doing it, but be warned that errors might occur. You can always run the installation script again if you change your mind."
		fi
            fi
	else
            warning "  Symlink '$LINKPATH' doesn't exist."
	fi

	# Deactivate environment
	print "  Deactivating '$ENV_CPU'."
	source deactivate 2>&1 1>/dev/null
    else
	warning "Failed to activate '$ENV_CPU'."
    fi
}

# Check whether conda is installed
if ! hash conda 2>/dev/null; then
    print "conda was not installed."
    question "Do you want to do it now?"
    RESPONSE="$?"
    if (( "$RESPONSE" )); then
	print "Installing Miniconda."
	# @TODO: Generalise to different OS's (in particular, macOS)
	INSTALLFILE="Miniconda2-latest-Linux-x86_64.sh"
	wget https://repo.continuum.io/miniconda/$INSTALLFILE
	bash $INSTALLFILE
	rm -f $INSTALLFILE
	if ! hash conda 2>/dev/null; then
	    print "conda wasn't installed properly. Perhaps something went wrong in the installation, or 'PATH' was not set? Exiting."
	    return 1
	else
	    print "conda was installed succesfully!"
	fi
    else
	print "Please install conda manually, see e.g. https://github.com/asogaard/adversarial. Exiting."
	return 1
    fi
fi

# Environment names
ENV_CPU="adversarial-cpu"
ENV_GPU="adversarial-gpu"

# Install CPU environment
ENVFILE=envs/$ENV_CPU.yml
if [ "$(conda info --envs | grep $ENV_CPU)" ]; then
    print "Environment '$ENV_CPU' already exists"
    
    # Check consistency with baseline env.
    print "  Checking consistency"

    # -- Silently activate environment
    source activate $ENV_CPU 2>&1 1>/dev/null 

    # -- Write the enviroment specifications to file
    TMPFILE=".tmp.env.txt"
    conda env export > $TMPFILE
    
    # -- Compare current enviroment with default
    DIFFERENCES="$(diff -y --left-column $TMPFILE $ENVFILE | grep -v "prefix:" | grep -v "(" | sed $'s/\t/    /g' )"
    if (( "${#DIFFERENCES}" )); then
	warning "  The existing '$ENV_CPU' env. differs from the default one in '$ENVFILE':"
	POSINDEX="$(echo "$DIFFERENCES" | grep -b -o "|" | cut -d: -f1)"
	printf "%-${POSINDEX}s| %s\n" "ACTIVE ENVIRONMENT" "DEFAULT ENVIRONMENT"
	printf "%0.s-" $(seq 1 $(( 2 * $POSINDEX + 1)) )
	echo ""
	echo "$DIFFERENCES"
	warning "  Beware that this might lead to problems when running the code."
    fi

    # -- Clean-up
    rm -f $TMPFILE
    
    # -- Silently deactivate environment
    source deactivate 2>&1 1>/dev/null
else
    print "Creating CPU environment '$ENV_CPU'."
    conda env create -f $ENVFILE
fi

# -- Fix libstdc++ symblink problem
fix_link $ENV_CPU
print "Done!"
