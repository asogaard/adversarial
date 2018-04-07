#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for slimming HDF5 files from W/top tagging ntuples."""

# Basic import(s)
import glob

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy

# Scientific import(s)
import numpy as np
from numpy.lib.recfunctions import append_fields
from hep_ml.reweight import BinsReweighter
import matplotlib.pyplot as plt

# Project import(s)
from adversarial.utils import mkdir
from adversarial.profile import Profile, profile
from .common import load_hdf5, save_hdf5

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Slim HDF5 file.")

parser.add_argument('--dir', action='store', type=str,
                    default='/eos/atlas/user/a/asogaard/adversarial/data/2018-04-07/',
                    help='Directory in which to read and write HDF5 files.')


# Global variable definition(s)
BRANCHES = [

    # Kinematics
    'm',
    'pt',

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

    print "Reading and slimming, adding to files in:\n  {}".format(args.dir)

    paths = sorted(glob.glob(args.dir + '*/*_full.h5'))
    print "Found {} files.".format(len(paths))

    for ipath, path in enumerate(paths):
        
        # Reading input HDF5 file
        print "[{}/{}] {}".format(ipath + 1, len(paths), path)
        data = load_hdf5(path)

        print "  Read {} samples.".format(data.shape[0])        
        
        # Perform slimming
        missing = [branch for branch in BRANCHES if branch not in data.dtype.names]
        if missing:
            print "ERROR: The following {} branches were not found in the input file:".format(len(missing))
            for name in missing:
                print "  {}".format(name)
                pass
            return 1
        
        data = data[BRANCHES].copy()
        
        # Add new, necessary fields
        data = append_fields(data, 'rho',    np.log(np.square(data['m']) / np.square(data['pt'])))
        data = append_fields(data, 'rhoDDT', np.log(np.square(data['m']) / data['pt'] / 1.))
        
        # Writing output HDF5 file
        save_hdf5(data, path.replace('_full', '_slim'))
        pass
    
    return


# Main function call
if __name__ == '__main__':
    main()
    pass
