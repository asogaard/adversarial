#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for slimming HDF5 files from W/top tagging ntuples."""

# Basic import(s)
import os
import h5py
import glob

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy

# Scientific import(s)
import numpy as np
from numpy.lib import recfunctions
from hep_ml.reweight import BinsReweighter
import matplotlib.pyplot as plt

# Project import(s)
from adversarial.utils import mkdir
from adversarial.profile import Profile, profile

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Slim HDF5 file.")

parser.add_argument('--input', action='store', type=str,
                    default='/eos/atlas/user/a/asogaard/adversarial/data/2018-04-04/data_full.h5',
                    help='Input HDF5 file.')
parser.add_argument('--output', action='store', type=str,
                    default='/eos/atlas/user/a/asogaard/adversarial/data/2018-04-04/',
                    help='Output directory, to which to write slimmed output HDF5 file.')
parser.add_argument('--tag', action='store', type=str,
                    default='data_slim',
                    help='Unique tag, used for output file name.')

# Global variable definition(s)
BRANCHES = [

    # Kinematics
    'm',
    'pt',

    # Event-level info
    'npv',

    # Truth variables
    'truth_pt',

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

    # For reproducibility
    np.random.seed(21)

    # Parse command-line argument
    args = parser.parse_args()
    
    # Modify output directory name to conform to convention
    if not args.output.endswith('/'): args.output += '/'

    print "Reading from input file:\n  {}\nand writing to output file:\n  {}".format(args.input, args.output + '{}.h5'.format(args.tag))

    # Reading input HDF5 file
    with Profile("Reading input HDF5 file"):
        with h5py.File(args.input, 'r') as hf:
            data = hf['dataset'][:]
            pass
        print "Read {} samples.".format(data.shape[0])
        pass

    # Perform slimming
    with Profile("Performing slimming"):
        missing = [branch for branch in BRANCHES if branch not in data.dtype.names]
        if missing:
            print "ERROR: The following {} branches were not found in the input file:".format(len(missing))
            for name in missing:
                print "  {}".format(name)
                pass
            return 1
        
        data = data[BRANCHES].copy()
        pass

    # Add new, necessary fields
    with Profile("Adding new fields"):
        data = recfunctions.append_fields(data, 'rho',    np.log(np.square(data['m']) / np.square(data['pt'])))
        data = recfunctions.append_fields(data, 'rhoDDT', np.log(np.square(data['m']) / data['pt'] / 1.))
        pass

    # Train/test split
    with Profile("Performing train/test split"):
        frac_train = 0.8
        msk_sig = data['signal'] == 1
        num_sig =   msk_sig .sum()
        num_bkg = (~msk_sig).sum()
        num_train = int(frac_train * min(num_sig, num_bkg))
        print "Found {:.1e} signal and {:.1e} background samples.".format(num_sig, num_bkg)
        print "Using {:.1e} samples for training, leaving {:.1e} signal and {:.1e} background samples for testing.".format(num_train, num_sig - num_train, num_bkg - num_train)
        
        idx_sig = np.where( msk_sig)[0]
        idx_bkg = np.where(~msk_sig)[0]
        idx_sig_train = np.random.choice(idx_sig, num_train, replace=False)
        idx_bkg_train = np.random.choice(idx_bkg, num_train, replace=False)
        
        data = recfunctions.append_fields(data, 'train', np.zeros_like(data['signal']).astype(int))
        data['train'][idx_sig_train] = 1
        data['train'][idx_bkg_train] = 1
        pass

    # Shuffle
    with Profile("Shuffling samples"):
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        data = data[idx]
        pass

    # Writing output HDF5 file
    with Profile("Writing output HDF5 file"):
        mkdir(args.output)
        with h5py.File(args.output + '{}.h5'.format(args.tag), 'w') as hf:
            hf.create_dataset('dataset',  data=data, compression="gzip")
            pass
        pass
    
    return


# Main function call
if __name__ == '__main__':
    print ""
    main()
    pass
