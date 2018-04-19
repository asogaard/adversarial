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
import multiprocessing

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
import math
import numpy as np
from numpy.lib.recfunctions import append_fields
from hep_ml.reweight import BinsReweighter

# Project import(s)
from adversarial.utils import garbage_collect
from adversarial.profile import Profile, profile
from .common import load_hdf5, save_hdf5, get_parser, run_batched

# Command-line arguments parser
parser = get_parser(train=True, test=True, dir=True, max_processes=True)
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

        # Run batched conversion in parallel
        queue = multiprocessing.Queue()
        parts = run_batched(FileLoader, list(enumerate(paths)), queue=queue, max_processes=args.max_processes)

        # Concatenate data in sorted order, for reproducibility
        data = np.concatenate(zip(*sorted(parts, key=lambda t: t[0]))[1])
        pass
    
    print "Found {} samples.".format(data.shape[0])

    # Subsample
    with Profile("Subsample"):
        for sig in [0,1]:

            # Select samples belonging to current category
            msk = data['signal'] == sig

            # Store reference of samples belonging to other category
            other = np.array(~msk).astype(bool)

            # Subsample current category
            num_sample = int((args.train + args.test) * 1E+06)
            if num_sample <= msk.sum():
                idx = np.random.choice(np.where(msk)[0], num_sample, replace=False)
                sample = np.zeros_like(msk).astype(bool)
                sample[idx] = True
            else:
                print "[WARNING] Requested {:.1e} samples, but only {:.1e} are availabe in current mask. Using all available samples.".format(num_sample, msk.sum())
                sample = np.ones_like(msk).astype(bool)
                pass

            # Select subsample, and all samples from other categories
            data = data[sample | other]
            pass
        pass


    # Re-weighting
    with Profile("Re-weighting"):

        # Add new data columns
        data = append_fields(data, 'weight_train', np.ones_like(data['weight_test']))
        data = append_fields(data, 'weight_adv',   np.ones_like(data['weight_test']))

        # Reweight signal and background separately
        for sig in [0,1]:

            # Prepare data arrays
            msk = data['signal'] == sig

            # Flat pT
            # ------------------------------------------------------------------
            original = data['pt'][msk]
            xmin, xmax = original.min(), original.max()
            target = np.random.rand(original.size) * (xmax - xmin) + xmin

            # Fit bins-reweighter
            reweighter = BinsReweighter(n_bins=100, n_neighs=1)
            reweighter.fit(original, target=target)
            

            # Predict new, flat-pT weight
            original = data['pt'][msk]
            data['weight_train'][msk] = reweighter.predict_weights(original)
            data['weight_train'][msk] /= data['weight_train'][msk].mean()

            print "data['weight_train'][msk] | min, 1-perc., mean, 99-perc., max: {}, {}, {}, {}, {}".format(
                data['weight_train'][msk].min(),
                np.percentile(data['weight_train'][msk], 1), 
                data['weight_train'][msk].mean(), 
                np.percentile(data['weight_train'][msk], 99), 
                data['weight_train'][msk].max())


            # (Flat-pT, physical-m) reweighted
            # ------------------------------------------------------------------
            original_m  = data['m'] [msk]
            original_pt = data['pt'][msk]
            original = np.vstack((original_m, original_pt)).T

            p  = data['weight_test'][msk]
            p /= math.fsum(p)

            ptmin, ptmax = data['pt'].min(), data['pt'].max()

            target_m  = np.random.choice(data['m'][msk], msk.sum(), p=p, replace=True)
            target_pt = np.random.rand(msk.sum()) * (ptmax - ptmin) + ptmin
            target = np.vstack((target_m, target_pt)).T

            # Fit bins-reweighter
            reweighter = BinsReweighter(n_bins=100, n_neighs=1)
            reweighter.fit(original, target=target)

            # Compute new weights
            data['weight_adv'][msk]  = reweighter.predict_weights(original)
            data['weight_adv'][msk] /= data['weight_adv'][msk].mean()


            # Physical testing weights
            # ------------------------------------------------------------------
            data['weight_test'][msk] /= data['weight_test'][msk].mean()

            pass
        pass


    # Train/test split
    with Profile("Performing train/test split"):
        msk_sig = data['signal'] == 1
        num_sig =   msk_sig .sum()
        num_bkg = (~msk_sig).sum()
        num_train = int(args.train * 1E+06)
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
        save_hdf5(data, args.dir + 'data_{}M_{}M.h5'.format(args.train, args.test))
        pass

    return


class FileLoader (multiprocessing.Process):

    def __init__ (self, vargs):
        """
        Process for loading W/top HDF5 files.

        Arguments:
            path: Path to the HDF5 file to be loaded.
            idx: Index of the process; for reproducibility.
        """

        # Unpack input arguments
        idx, path = vargs

        # Base class constructor
        super(FileLoader, self).__init__()

        # Member variable(s)
        self.__idx   = idx
        self.__path  = path
        self.queue   = None  # Set by the runner script.
        return

    @garbage_collect
    def run (self):

        # Load data
        data = load_hdf5(self.__path)

        # Store in return dict
        self.queue.put((self.__idx, data))
        return

    pass


# Main function call
if __name__ == '__main__':
    main()
    pass
