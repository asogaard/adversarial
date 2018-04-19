#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for slimming HDF5 files from W/top tagging ntuples.

  1. Convert
> 2. Slim
  3. Reweight
"""

# Basic import(s)
import re
import glob
import multiprocessing

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy

# Scientific import(s)
import numpy as np
from numpy.lib.recfunctions import append_fields

# Project import(s)
from adversarial.utils import garbage_collect
from adversarial.profile import Profile, profile
from .common import load_hdf5, save_hdf5, get_parser, run_batched

# Command-line arguments parser
parser = get_parser(dir=True, max_processes=True)
parser.description = "Slim, decorate HDF5 file."

# Global variable definition(s)
BRANCHES = [

    # Kinematics
    'm',
    'pt',

    # Truth variables
    'truth_pt',

    # Event-level info
    'npv',

    # Substructure variables.
    'Tau21',        # Wtop
    'C2',           # Wtop
    'D2',           # Wtop
    'Angularity',   # Wtop
    'Aplanarity',   # Wtop
    'FoxWolfram20', # Wtop
    'KtDR',         # Wtop
    'N2',
    'PlanarFlow',   # Wtop
    'Split12',      # Wtop
    'ZCut12',       # Wtop

    # Weights
    'weight_test',
    'signal',
    ]


# Main function definition
@profile
def main ():

    # Parse command-line argument
    args = parser.parse_args()

    # Modify directory name to conform to convention
    if not args.dir.endswith('/'): args.dir += '/'

    print "Reading and slimming, decorating files in:\n  {}".format(args.dir)

    paths = sorted(glob.glob(args.dir + '*/*_full.h5'))
    print "Found {} files.".format(len(paths))

    # Run batched slimming in parallel
    run_batched(FileSlimmer, paths, max_processes=args.max_processes)
    return


class FileSlimmer (multiprocessing.Process):

    def __init__ (self, path):
        """
        Process for slimming, decorating W/top HDF5 files.

        Arguments:
            path: Path to the HDF5 file to be slimmed, decorated.
        """

        # Base class constructor
        super(FileSlimmer, self).__init__()

        # Member variable(s)
        self.__path = path
        return


    @garbage_collect
    def run (self):

        # Get unique file identifier
        identifier = re.search("(WZqqqq_m[\d]+|JZ[\d]+W)\.", self.__path).groups(1)[0]

        # Load data
        data = load_hdf5(self.__path)
        print "     Read {:8d} samples ({}).".format(data.shape[0], identifier)

        # Perform slimming
        missing = [branch for branch in BRANCHES if branch not in data.dtype.names]
        if missing:
            print "ERROR: The following {} branches were not found in the input file:".format(len(missing))
            for name in missing:
                print "  {}".format(name)
                pass
            raise IOError()

        data = data[BRANCHES].copy()

        # @FIXME: Filter out NaN N2's
        data = data[~np.isnan(data['N2'])]

        # Add new, necessary fields
        data = append_fields(data, 'rho',    np.log(np.square(data['m']) / np.square(data['pt'])))
        data = append_fields(data, 'rhoDDT', np.log(np.square(data['m']) / data['pt'] / 1.))

        # Writing output HDF5 file
        save_hdf5(data, self.__path.replace('_full', '_slim'))
        return
    pass


# Main function call
if __name__ == '__main__':
    main()
    pass
