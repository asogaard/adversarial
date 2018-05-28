#!/bin/bash
# Utility script to stage some data for testing.

# Import general utility methods
source scripts/utils.sh

# Variable definitions
usertarget="" # Default
username="" # Default
target=data
source=/eos/atlas/user/a/asogaard/adversarial/data/2018-04-20
filename=data_1M_10M.h5

# Make sure target directory exists
mkdir -p $target

# Get bash version
bash_version="$(bash --version | head -1 | sed 's/.*version //g;s/\..*//g')"

# Check host
if [[ "$(hostname)" == *"lxplus"* ]]; then

  # Check if file exists
  if [ -f "$target/$filename" ] && [ -L "$target/$filename" ]; then
  	warning "Regular file $filename already exists in $target/. Not overwriting. Make sure it is the right one."
  	return 1
  fi

  # Create a symbolic link
  if [ ! -f "$target/$filename" ] || [ "$source/$filename" ! -ef "$target/$filename" ]; then
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

  if (( "$bash_version" > 3 )); then
    # Bash version support readline default value (Linux platform)
    read -e -i "$username" -p ">> " username
  else
    # Bash version does not support readline default value (macOS platform)
    read -e -p ">> [$username] " response
    if [[ ! -z "$response" ]]; then
      username=$response
    fi
  fi

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

  if (( "$bash_version" > 3 )); then
    # Bash version support readline default value (Linux platform)
    read -e -i "$usertarget/" -p ">> " usertarget
  else
    # Bash version does not support readline default value (macOS platform)
    read -e -p ">> [$usertarget] " response
    if [[ ! -z "$response" ]]; then
      usertarget=$response
    fi
  fi

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

  # Check if file exists
  if [ -f "$usertarget/$filename" ] && [ ! -L "$usertarget/$filename" ]; then
  	warning "Regular file $filename already exists in $usertarget/. Not overwriting. Make sure it is the right one."
  	return 1
  fi

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
  if [ "$usertarget" -ef "$target" ]; then
    # Downloaded to default target; nothing more to do
    :
  else
  	if [ -f "$target/$filename" ] && [ ! -L "$target/$filename" ]; then
	    warning "Regular file $filename already exists in $target/. Not overwriting. Make sure it is the right one."
  	elif [ ! -f "$target/$filename" ] || [ "$usertarget/$filename" ! -ef "$target/$filename" ]; then
	    # Create symlink to data file in default data directory
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
