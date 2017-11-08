#!/bin/bash
# Utility functions to perform installation of conda environments

# Import general utility methods
source scripts/utils.sh

# ------------------------------------------------------------------------------
# Utility function to resolve libstdc++ link problems
function fix_link () {

    # Define variable
    env="$1"

    # Validate input
    if [ -z "$env" ]; then
        warning "fix_link: No environment was specified"
        return
    fi

    # Silently try to activate environment
    source activate "$env" 2>&1 1>/dev/null

    # Check if environment was succesfully activated
    env_active="$(conda info --envs | grep \* | sed 's/ .*//g')"
    if [ "$env_active" == "$env" ]; then

        # Fix libstdsc++ symlink problem on linux platforms
        if [[ "$(uname)" == *"Linux"* ]]; then

            # Base directory of active environment
            envdir="$(conda info --env | grep \* | sed 's/.* //g')"

            # Problematic symlink
            linkpath="$envdir/lib/libstdc++.so.6"

            # Latest available libstdc++ library
            latestlib="$(find $envdir/lib/ -name libstdc++.* ! -type l | grep -v .py | sort | tail -1)"

            # Check that link exists
            if [ -L "$linkpath" ]; then
                # Check whether link target is most latest available library
                if [ "$(readlink -f $linkpath)" == "$(readlink -f $latestlib)" ]; then
                    # $linkpath already links to $latestlib
                    :
                else
                    # Try to update symlink target
                    print "  Changing target of"
                    print "    $linkpath"
                    print "  to be"
                    print "    $latestlib"
                    question "  Is that OK?" "y"
                    response="$?"
                    if (( $response )); then
                        ln -s -f $latestlib $linkpath
                    else
                        warning "  OK, not doing it, but be warned that errors might occur. You can always run the installation script again if you change your mind."
                    fi
                fi
            else
                warning "  Symlink '$linkpath' doesn't exist."
            fi

        fi

        # Silently deactivate environment
        source deactivate 2>&1 1>/dev/null
    else
        warning "Failed to activate '$env'."
    fi
}


# ------------------------------------------------------------------------------
# Utility function to resolve Theano/gcc compiling problem
function fix_theano_gcc () {

    # gcc on CentOS* has a know problem [https://stackoverflow.com/q/13879302]
    # which means that theano doesn't compile and therefore cannot be
    # used. Manually add a cxxflag to .theanorc to permanently avoid this
    # problem.

    # Check if anything needs to be done
    if [[ "$(uname)" != *"Linux"* ]]; then
	# Not on linux; no problems.
	return 0
    fi

    theanofile=~/.theanorc
    cxxflag="-D__USE_XOPEN2K8"
    
    if [[ -f "$theanofile" ]] && [[ ! -z "$(cat $theanofile | grep "\\$cxxflag")" ]]; then
	# Fix has been applied; no problems.
	return 0
    fi
    
    # Action required
    warning "You seem to be running on an OS (Linux) where a know problem with gcc prevents Theano from compiling."
    
    # Check if ~/.theanorc file alfready exists
    if [[ ! -f "$theanofile" ]]; then
	
	question "Is it OK to create the file '$theanofile' and add 'gcc.cxxflags = $cxxflag'?" "y"
	response="$?"
	if (( "$response" )); then
	    # Create new ~/.theanorc file
	    echo -e "[gcc]\ncxxflags = $cxxflag" >> $theanofile
	    print "Created '$theanofile'. Should be good to go."
	    return 0
	else
	    warning "OK, not creating '$theanofile', but you will need to use the TensorFlow backend for Keras or manually set the '$cxxflag' flag yourself."
	    return 1
	fi
	
    else
	
	if [[ -z "$(cat $theanofile | grep "\\$cxxflag")" ]]; then
	    question "Is it OK to add 'gcc.cxxflags = $cxxflag' to the existing '$theanofile'?" "y"
	    response="$?"
	    if (( "$response" )); then
		if   [[ "$(cat $theanofile | grep "cxxflags")" ]]; then
		    # Add new item to existing cxxflags list
		    sed -i.bak "s/cxxflags *= *\(.*\)/cxxflags = \1 $cxxflag/g" $theanofile
		elif [[ "$(cat $theanofile | grep "[gcc]")" ]]; then
		    # Add new cxxflags list to existing gcc field
		    sed -i.bak "s/\[gcc\]/\[gcc\]\ncxxflags = $cxxflag/g" $theanofile
		else
		    # Add new gcc field
		    echo -e "\n[gcc]\ncxxflags = $cxxflag" >> $theanofile
		fi
		print "Appended '$theanofile'. Should be good to go."
		return 0
	    else
		warning "OK, not appending '$theanofile', but you will need to use the TensorFlow backend for Keras or manually set the '$cxxflag' flag yourself."
		return 1
	    fi
	fi

    fi
}


# ------------------------------------------------------------------------------
# Install Miniconda2 
function install_conda () {
    
    print "Installing Miniconda."
    
    # Download install file
    REPO="https://repo.continuum.io/miniconda"
    if   [[ "$(uname)" == *"Linux"* ]]; then
        INSTALLFILE="Miniconda2-latest-Linux-x86_64.sh"
        wget $REPO/$INSTALLFILE
    elif [[ "$(uname)" == *"Darwin"* ]]; then
        INSTALLFILE="Miniconda2-latest-MacOSX-x86_64.sh"
        curl -o ./$INSTALLFILE -k $REPO/$INSTALLFILE
    else
        warning "Uname '$(uname)' not recognised. Exiting."
        return 1
    fi
    
    # Run installation
    bash $INSTALLFILE
    
    # Clean-up
    rm -f $INSTALLFILE
    
    # Re-source the bash initialisation script, to update PATH which might have
    # been set.
    if [[ -f ~/.bashrc ]]; then
        source ~/.bashrc
    fi    

    # Check whether installation worked
    if ! hash conda 2>/dev/null; then
        warning "conda wasn't installed properly. Perhaps something went wrong in the installation, or 'PATH' was not set? If you suspect the latter, update the 'PATH' environment variable and re-run the installation script. Exiting."
        return 1
    else
        print "conda was installed succesfully!"
    fi
}

# ------------------------------------------------------------------------------
# Check consistency of existing conda environment with .yml file from which it
# should be created
function check_env_consistency () {

    # Define variable(s)
    env="$1"
    envfile="$2"

    # Validate input
    if [[ -z "$env" ]]; then
        warning "Please specify the environment name as first argument to 'check_env_consistency'."
        return 1
    fi
    if [[ -z "$envfile" ]] || [[ "$envfile" != *".yml" ]]; then
        warning "Please specify the environment file (.yml) as second argument to 'check_env_consistency'."
        return 1
    fi
    
    # Check consistency with baseline env.
    print "Checking consistency of conda environment '$env' with file '$envfile'."
    
    # Silently activate environment
    source activate $env 2>&1 1>/dev/null

    # (1) Conda packages
    # Write the enviroment specifications to file
    tmpfile=".tmp.env.txt"
    conda env export > $tmpfile

    # Get as list of all of the required package versions, and check for each of
    # these whether the active environment contains a matching package.
    conda_packages=$"$(sed -e '1,/dependencies/d' $envfile  | grep -E "^-.*[^\:]$" | sed 's/^- //g')"
    old_IFS=$IFS
    IFS=$'\n' read -rd '' -a conda_packages <<<"$conda_packages"
    IFS=$old_IFS
    for conda_package in "${conda_packages[@]}"; do
	if [[ -z "$(cat "$tmpfile" | grep "$conda_package")" ]]; then
	    warning "No conda_package matching '$conda_package' was found in the active '$env' environment."
	    conda_package_name="$(echo $conda_package | cut -d= -f1)"
	    if [[ -z "$(cat "$tmpfile" | grep "$conda_package_name=")" ]]; then
		warning "Conda package '$conda_package_name' was not installed at all."
	    else
		warning "Conda package '$conda_package_name' was installed, but with possibly incompatible version:"
		warning "  $(cat "$tmpfile" | grep "$conda_package_name=" | cut -d= -f1-2 | sed 's/^- //g') (intalled) vs. $conda_package (required)"
	    fi
	fi
    done

    # (2) pip packages
    pip_packages=$"$(sed -e '1,/^- pip:/d' $envfile  | grep -E "^  -.*[^\:]$" | sed 's/^ *- //g')"
    old_IFS=$IFS
    IFS=$'\n' read -rd '' -a pip_packages <<<"$pip_packages"
    IFS=$old_IFS

    pip list --format=legacy > $tmpfile
    for pip_package in "${pip_packages[@]}"; do
	pip_package_name="$(echo $pip_package | cut -d= -f1)"
	pip_package_version="$(echo $pip_package | cut -d= -f3)"

	if [[ "$pip_package_name" == *"git"*":"* ]]; then
	    pip_package_actualname="$(echo $pip_package_name | sed 's/.*\/\(.*\)\.git.*/\1/g')"
	    print "  - It seems like pip package '$pip_package_name$' ($pip_package_actualname) is taken straight from git."
	    print "    Please make sure that the installed version is consistent with this."
	    continue
	fi
	
	pattern="$(sed 's/\./\\\./g' <<< "$pip_package_name ($pip_package_version)" | sed 's/)/.*)/g')"
	if [[ -z "$(cat $tmpfile | grep -i "$pattern")" ]]; then
	    warning "No pip package matching '$pip_package' was found in the active '$env' environment"
	    if [[ -z "$(cat $tmpfile | grep -i $pip_package_name)" ]]; then
		warning "Pip package '$pip_package_name' was not installed at all."
	    else
		warning "Pip package '$pip_package_name' was installed, but with possibly incompatible version:"
		warning "  $(cat $tmpfile | grep -i $pip_package_name | sed 's/ *(\(.*\))/==\1/g' | tr '[:upper:]' '[:lower:]') (installed) vs. $pip_package (required)"
	    fi
	fi
    done
    
    # Clean-up
    rm -f $tmpfile
    
    # Silently deactivate environment
    source deactivate 2>&1 1>/dev/null
    
}


# ------------------------------------------------------------------------------
# Create conda environment from .yml file
function create_env () {

    # Define variable(s)
    env="$1"
    envfile="$2"

    # Validate input
    if [[ -z "$env" ]]; then
	warning "Please specify the environment name as first argument to 'create_env'."
	return 1
    fi
    if [[ -z "$envfile" ]] || [[ "$envfile" != *".yml" ]]; then
	warning "Please specify the environment file (.yml) as second argument to 'create_env'."
	return 1
    elif [ ! -f "$envfile" ]; then
	warning "Wanted to create '$env', but specified environment file '$envfile' doesn't exist."
	return 1
    fi

    print "Setting up conda environment '$env'."
    
    # Check if environment already exists
    if [ "$(conda info --envs | grep $env)" ]; then

	print "Environment '$env' already exists"

	check_env_consistency "$env" "$envfile"
	
    else
	
	# Fix ROOT setup problem on macOS (1/2)
	if [[ "$(uname)" == *"Darwin"* ]]; then
            # The `activateROOT.sh` scripts for Linux and macOS are different, the
            # later being broken. Therefore, we (1) need to set the `CONDA_ENV_PATH`
            # environment variable before installation and (2) need to update
            # `activateROOT.sh` after installation so as to not have to do this
            # every time.
            CONDA_ENV_PATH="$(which conda)"
            SLASHES="${CONDA_ENV_PATH//[^\/]}"
            CONDA_ENV_PATH="$(echo "$CONDA_ENV_PATH" | cut -d '/' -f -$((${#SLASHES} - 2)))"
            CONDA_ENV_PATH="$CONDA_ENV_PATH/envs/$env"
            export CONDA_ENV_PATH
	fi
	
	# Create environment
	print "Creating new environment '$env' from file '$envfile'."
	conda env create -f $envfile
	
        # Silently activate environment
        source activate $env 2>&1 1>/dev/null

	# Fix ROOT setup problem on macOS (2/2)
	if [[ "$(uname)" == *"Darwin"* ]]; then
	    
	    unset CONDA_ENV_PATH

            # Check if environment was succesfully activated
            env_active="$(conda info --envs | grep \* | sed 's/ .*//g')"
            if [ "$env_active" == "$env" ]; then
		
		# Base directory of active environment
		envdir="$(conda info --env | grep \* | sed 's/.* //g')"
		
		# Locate ROOT activation file
		rootinitfile="$envdir/etc/conda/activate.d/activateROOT.sh"
		
		if [[ -f "$rootinitfile" ]] && [[ -z "$(cat $rootinitfile | grep "^ *source \$(which thisroot.sh)")" ]]; then
		    
		    # Replace whatever statement is used to source 'thisroot.sh' by 'source
		    # $(which thisroot)' to remove dependence on the 'CONDA_ENV_PATH'.
		    sed -i.bak 's/\(^.*thisroot.sh\)/#\1\'$'\nsource $(which thisroot.sh)/g' $rootinitfile
		fi
            fi
	fi
	
	# Silently deactivate environment
	source deactivate 2>&1 1>/dev/null
    fi

    # Fix known issues
    fix_link $env
    fix_theano_gcc

    print "Done!"    
}
