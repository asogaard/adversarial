#!/bin/bash

# Clean up the current directory
rm core.[0-9]* 2> /dev/null
rm *~ 2> /dev/null
rm */*~ 2> /dev/null
rm */*/*~ 2> /dev/null
rm *.o[0-9]* 2> /dev/null
rm *.e[0-9]* 2> /dev/null
rm *.po[0-9]* 2> /dev/null
rm *.pe[0-9]* 2> /dev/null
