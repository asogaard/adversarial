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
	print "fix_link: No environment was specified"
	return
    fi

    # Try to activate environment
    print "  Activating '$ENV'."
    source activate "$ENV"

    # Check if environment was succesfully activated
    ENV_ACTIVE="$(conda info --envs | grep \* | sed 's/ .*//g')"
    if [ "$ENV_ACTIVE" == "$ENV" ]; then
	print "  Fixing problem with libstdc++.so.6 symlink"

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
		print "  '$LINKPATH' already links to '$LATESTLIB'"
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
                    print "  OK, not doing it, but be warned that errors might occur. You can always run the installation script again if you change your mind."
		fi
            fi
	else
            print "  Symlink '$LINKPATH' doesn't exist."
	fi

	# Deactivate environment
	print "  Deactivating '$ENV_CPU'."
	source deactivate
    else
	print "Failed to activate '$ENV_CPU'."
    fi
}

# Check whether conda is installed
if [ -z `which conda` ]; then
    print "Please install `conda`, see e.g. https://github.com/asogaard/adversarial. Exiting installation."
    return
fi

# Environment names
ENV_CPU="adversarial-cpu"
ENV_GPU="adversarial-gpu"

# Install CPU environment
if [ "$(conda info --envs | grep $ENV_CPU)" ]; then
    print "Environment '$ENV_CPU' already exists"
else
    print "Creating CPU environment '$ENV_CPU'."
    conda env create -f envs/$ENV_CPU.yml
fi

# -- Fix libstdc++ symblink problem
fix_link $ENV_CPU
print "Done!"
