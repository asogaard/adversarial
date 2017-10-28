#!/bin/bash
# Bash utility methods

# Function to print fancy text
function print {
    echo -e "🎃  \033[1;38;5;208m$1\033[0m"
}

function question {
    # Define variables
    QUESTION="$1"
    DEFAULT="${2:-n}"
    OPTIONS="$(echo "y/n" | sed "s/$DEFAULT/[$DEFAULT]/g")"

    # Prompt response
    print "$QUESTION ($OPTIONS) \c"
    read RAW_RESPONSE

    # Format responce as single lower-case character
    RESPONSE=${RAW_RESPONSE:-$DEFAULT}
    RESPONSE=${RESPONSE:0:1}
    RESPONSE=`echo "${RESPONSE,,}"`

    # Return numeric value
    if   [ "$RESPONSE" == "y" ]; then
	return 1
    elif [ "$RESPONSE" == "n" ]; then
	return 0
    else
	print "Response '$RAW_RESPONSE' not recognised. Using default value '$DEFAULT'."
	if   [ "$DEFAULT" == "y" ]; then
	    return 1
	else
	    return 0
	fi
    fi
}
