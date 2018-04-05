#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for conveniently formatting and storing the W/top-tagging ntuples."""

# Basic import(s)
import h5py

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

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

parser = argparse.ArgumentParser(description="Re-weight HDF5 file to flat pT-spectrum.")

parser.add_argument('--input', action='store', type=str,
                    default='/eos/atlas/user/a/asogaard/adversarial/data/2018-04-04/data_slim.h5',
                    help='Input HDF5 file.')
parser.add_argument('--output', action='store', type=str,
                    default='/eos/atlas/user/a/asogaard/adversarial/data/2018-04-04/',
                    help='Output directory, to which to write re-weighted HDF5 file.')
parser.add_argument('--tag', action='store', type=str,
                    default='data',
                    help='Unique tag, used for output file name.')

# @TODO Derive `output` from `args.input`


# Main function definition
@profile
def main ():

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

    # Re-weighting
    with Profile("Re-weighting"):

        # Add new data columns
        data = recfunctions.append_fields(data, 'weight_train', np.ones_like(data['weight_test']))

        # Reweight signal and background separately
        for sig in [0,1]:

            # Prepare data arrays
            msk = data['signal'] == sig
            original = data['pt'][msk]
            xmin, xmax = original.min(), original.max()
            target = np.random.rand(original.size) * (xmax - xmin) + xmin

            # Fit bins-reweighter
            reweighter = BinsReweighter(n_bins=100, n_neighs=1)
            reweighter.fit(original, target=target)

            # Predict new, flat-pT weights
            weight_train  = reweighter.predict_weights(original)
            weight_train /= weight_train.mean()

            # Store new, flat-pT weights 
            data['weight_train'][msk] = weight_train
            print "weight_train | min, 1-per., mean, 99-perc, max: {}, {}, {}, {}, {}".format(weight_train.min(), np.percentile(weight_train, 1), weight_train.mean(), np.percentile(weight_train, 99), weight_train.max())
            pass
        pass


    # Writing output HDF5 file
    with Profile("Writing output HDF5 file"):
        mkdir(args.output)
        with h5py.File(args.output + '{}.h5'.format(args.tag), 'w') as hf:
            hf.create_dataset('dataset',  data=data, compression="gzip")
            pass
        pass


    # Plotting
    with Profile("Plotting"):
        for sig in [0,1]:

            # Define common variables
            msk = data['signal'] == sig
            bins = np.linspace(200, 2000, (2000 - 200) // 50 + 1, endpoint=True)
            opts = dict(bins=bins, alpha=0.4, normed=True)
            
            # Compare un-, train-, and test-weighted
            fig, ax = plt.subplots()
            plt.suptitle('Signal' if sig else 'Background')
            plt.hist(data['pt'][msk], weights=data['weight_train'][msk], label='Train', **opts)
            plt.hist(data['pt'][msk], weights=data['weight_test'] [msk], label='Test',  **opts)
            plt.hist(data['pt'][msk], label='Unweighted', **opts)
            ax.set_yscale('log')
            plt.legend()
            plt.show()
            
            # Compare dropping weights
            fig, ax = plt.subplots()
            msk_good = data['weight_train'][msk] < np.percentile(data['weight_train'][msk], 99)
            plt.suptitle('Signal' if sig else 'Background')
            plt.hist(data['pt'][msk], weights=data['weight_train'][msk], label='Train', **opts)
            plt.hist(data['pt'][msk][msk_good], weights=data['weight_train'][msk][msk_good], label='Train (clip)',  **opts)
            plt.legend()
            plt.show()
            pass
        pass
    
    return


# Main function call
if __name__ == '__main__':
    print ""
    main()
    pass
