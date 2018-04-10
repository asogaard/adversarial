#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for reweighting HDF5 files from W/top tagging ntuples.

  1. Convert
  2. Slim
> 3. Reweight
"""

# Basic import(s)
import glob
import datetime

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
import numpy as np
from numpy.lib.recfunctions import append_fields
from hep_ml.reweight import BinsReweighter

# Project import(s)
from adversarial.profile import Profile, profile
from .common import load_hdf5, save_hdf5, get_parser

# Command-line arguments parser
parser = get_parser(size=True, dir=True)
parser.description = "Re-weight HDF5 file to flat pT-spectrum."

# Main function definition
@profile
def main ():

    # For reproducibility
    np.random.seed(21)

    # Parse command-line argument
    args = parser.parse_args()

    # Modify directory name to conform to convention
    if not args.dir.endswith('/'): args.dir += '/'

    print "Reading and reweighting, splitting files in:\n  {}".format(args.dir)

    paths = sorted(glob.glob(args.dir + '*/*_slim.h5'))

    print "Found {} files.".format(len(paths))

    # Reading input HDF5 file(s)
    data = None
    with Profile("Reading input HDF5 file(s)"):
        for ipath, path in enumerate(paths):
            print "[{}/{}] {}".format(ipath + 1, len(paths), path)
            part = load_hdf5(path)

            print "  Read {} samples.".format(part.shape[0])
            data = part if data is None else np.concatenate((data, part))
            pass
        pass


    # Subsample
    with Profile("Subsample"):
        for sig in [0,1]:

            # Select samples belonging to current category
            msk = data['signal'] == sig

            # Store reference of samples belonging to other category
            other = np.array(~msk).astype(bool)

            # Subsample current category
            idx = np.random.choice(np.where(msk)[0], int(2 * args.size * 1E+06), replace=False)
            sample = np.zeros_like(msk).astype(bool)
            sample[idx] = True

            # Select subsample, and all samples from other categories
            data = data[sample | other]
            pass
        pass


    # Re-weighting
    with Profile("Re-weighting"):

        # Add new data columns
        data = append_fields(data, 'weight_train', np.ones_like(data['weight_test']))

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


    # Train/test split
    with Profile("Performing train/test split"):
        frac_train = 0.5
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

        data = append_fields(data, 'train', np.zeros_like(data['signal']).astype(int))
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
        save_hdf5(data, args.dir + 'data_{}M.h5'.format(args.size))
        pass

    return


# Main function call
if __name__ == '__main__':
    main()
    pass
