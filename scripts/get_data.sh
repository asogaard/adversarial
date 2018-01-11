#!/bin/bash
# Utility script to stage some data for testing.

# Import general utility methods
source scripts/utils.sh

# Variable definitions
usertarget=""  # Default
username="" # Default
target=input/
source=/eos/atlas/user/a/asogaard/adversarial/data
filename=data.h5

# Make sure target directory exists
mkdir -p $target

# Check host
if [[ "$(hostname)" == *"lxplus"* ]]; then

    # Check if file exists
    if [ -f "$target/$filename" ] && [ -L "$target/$filename" ]; then
	warning "Regular file $filename already exists in $target/. Not overwriting. Make sure it is the right one."
	return 1
    fi

    # Create a symbolic link
    if [ ! -f "$target/$filename" ] || [[ "$(readlink -f $source/$filename)" != "$(readlink -f $target/$filename)" ]]; then
	print "Creating symlink to $source/$filename in $target/"
	question "Is that OK?" "y"
	response="$?"
	if (( "$response" )); then
	    ln -s -f $source/$filename $target/
	else
	    warning "Exiting."
	    return 1
	fi
    fi

else

    # Get lxplus username
    print "Please enter your lxplus username and press [ENTER]:"
    if [[ -z "$username" ]]; then
	username="$(whoami)"
    fi
    read -e -i "$username" -p ">> " username
    if [[ -z "$username" ]]; then
	warning "No username was specified. Exiting."
	return 1
    fi

    # Update the default lxplus username
    thisfile="$(pwd)/${BASH_SOURCE[0]}"
    sed -i.bak "s/^username=.*# *Default *$/username=$username # Default/g" $thisfile

    # Get target directory
    print "Please specify target directory for data file download and press [ENTER]:"
    print "(pwd = $(pwd))"
    if [[ -z "$usertarget" ]]; then
	usertarget="$target"
    fi
    read -e -i "$usertarget/" -p ">> " usertarget
    usertarget="${usertarget%/}"  # Remove trailing slash
    usertarget="${usertarget/#\~/$HOME}"  # Expand possible tilde
    if   [[ -z "$usertarget" ]]; then 
	warning "No download target was specified. Exiting."
	return 1
    elif [ ! -d "$usertarget" ]; then
	warning "Directory $usertarget doesn't exist. Exiting."
	return 1
    fi

    # Update the default user target path
    thisfile="$(pwd)/${BASH_SOURCE[0]}"
    sed -i.bak "s/^usertarget=.*# *Default *$/usertarget=$(echo "$usertarget" | sed 's/\//\\\//g') # Default/g" $thisfile

    # Download the data file
    print "Downloading"
    print "  lxplus.cern.ch:$source/$filename (1.4GB size)"
    print "to"
    print "  $usertarget/"
    question "Is that OK?" "y"
    response="$?"
    if (( "$response" )); then
	scp $username@lxplus.cern.ch:$source/$filename $usertarget/
    else
	warning "Exiting."
	return 1
    fi

    # Check whether symlink is needed
    if [[ "$(readlink -f $usertarget)" == "$(readlink -f $target)" ]]; then
	# Downloaded to default target; nothing more to do
	:
    else
	if [ -f "$target/$filename" ] && [ ! -L "$target/$filename" ]; then
	    warning "Regular file $filename already exists in $target/. Not overwriting. Make sure it is the right one."
	elif [ ! -f "$target/$filename" ] || [[ "$(readlink -f $usertarget/$filename)" != "$(readlink -f $target/$filename)" ]]; then
	    # Create symlink to data file in default input directory
	    print "Creating symlink in $target/ to downloaded file $filename."
	    question "Is that OK?" "y"
	    response="$?"
	    if (( "$response" )); then
		ln -s -f $usertarget/$filename $target/
	    else
		warning "Exiting."
		return 1
	    fi
	fi
    fi

fi

print "Done!"
