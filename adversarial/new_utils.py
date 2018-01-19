#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Commong utility methods for de-correlated jet tagging."""

# Basic import(s)
import os
import sys
import json
import logging as log
import argparse

# Scientific import(s)
import numpy as np
import pandas as pd

# Project import(s)
import adversarial
from adversarial.profile import profile

# Global variables
PROJECTDIR='/'.join(os.path.realpath(adversarial.__file__).split('/')[:-2] + [''])


def parse_args (cmdline_args=sys.argv[1:]):
    """General script to query command-line arguments from the user, commen to
    all run scripts.

    Arguments:
        cmdline_args: List of arguments, either from the command-line (default)
            or specified programatically as a list of strings.

    Returns:
        args: argparse namespace containing all common `argparse` arguments,
            optionally with values specified by the user.
    """

    parser = argparse.ArgumentParser(description="Training uBoost classifierfor de-correlated jet tagging.")

    # -- Inputs
    parser.add_argument('-i', '--input',  dest='input',   action='store', type=str,
                        default=PROJECTDIR + 'data/', help='Input directory, from which to read HDF5 data file.')
    parser.add_argument('-o', '--output', dest='output',  action='store', type=str,
                        default=PROJECTDIR + 'output/', help='Output directory, to which to write results.')
    parser.add_argument('-c', '--config', dest='config',  action='store', type=str,
                        default=PROJECTDIR + 'configs/default.json', help='Configuration file.')
    parser.add_argument('-p', '--patch', dest='patches', action='append', type=str,
                        help='Patch file(s) with which to update configuration file.')

    # -- Flags
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_const',
                        const=True, default=False, help='Print verbose')

    return parser.parse_args(cmdline_args)


@profile
def initialise (args):
    """General script to perform any initialisation common to all run scripts.
    Assumes the existence of keys in the namespace corresponding to the common
    `argparse` arguments defined in the common `parse_args` script.

    Arguments:
        args: argparse namespace containing all arguments specified by the user.

    Returns:
        Tuple of `args` (possibly modified) and `cfg`, the configuration
        dictionary to be used in the run script.

    Raises:
        IOError: If any of the arguments are not valid, or any of the specified
            files don't exist.
    """

    # Set print level
    log.basicConfig(format="%(levelname)s: %(message)s",
                    level=log.DEBUG if args.verbose else log.INFO)

    #  Modify input/output directory names to conform to convention
    if not args.input .endswith('/'): args.input  += '/'
    if not args.output.endswith('/'): args.output += '/'

    # Make sure output directory exists
    if not os.path.exists(args.output):
        print "Creating output directory:\n  {}".format(args.output)
        os.makedirs(args.output)
        pass

    # Load configuration file
    with open(args.config, 'r') as f:
        cfg = json.load(f)
        pass

    # Apply patches
    args.patches = args.patches or []
    for patch_file in args.patches:
        log.info("Applying patch '{}'".format(patch_file))
        with open(patch_file, 'r') as f:
            patch = json.load(f)
            pass
        apply_patch(cfg, patch)
        pass

    # Return
    return args, cfg


@profile
def load_data (path, name='dataset', train_fraction=0.8, seed=21):
    """General script to load data, common to all run scripts.
    The loaded data is shuffled, re-indexed, and augmented with additional
    column(s), e.g. `train`. The weights for the signal and background sets are
    normalised to the same sum, such that the average signal sample weight is 1.
    A pre-selection, common to all run scripts, is applied to the set of samples.

    Arguments:
        path: The path to the HDF5 file, from which data should be loaded.
        name: Name of the dataset, as stored in the HDF5 file.
        train_fraction: The fraction of loaded, valid samples to be used for
            training, the remained used for testing.
        seed: Random state set before shuffling dataset _and_ before selecting
            training sample.

    Returns:
        Tuple of pandas.DataFrame containing the loaded, augmented dataset; list
        of loaded features to be used for training; and list of features to be
        used for de-correlation.

    Raises:
        AssertionError: If any manual checks of the function arguments fails.
        IOError: If no HDF5 file exists at the specified `path`.
        KeyError: If the HDF5 does not contained a dataset named `name`.
        KeyError: If any of the necessary features are not present in the loaded
            dataset.
    """

    # Check(s)
    assert train_fraction > 0 and train_fraction <= 1, \
        "Training fraction {} is not valid".format(train_fraction)

    # Read data from HDF5 file
    data = pd.read_hdf(path, name)

    # Perform selection
    msk  = (data['m']  >  40.) & (data['m']  <  300.)
    msk &= (data['pt'] > 200.) & (data['pt'] < 2000.)
    data = data[msk]

    # Select featues
    features_input         = ['Tau21', 'C2', 'D2', 'Angularity', 'Aplanarity', 'FoxWolfram20', 'KtDR', 'PlanarFlow', 'Split12', 'ZCut12']
    features_decorrelation = ['m']
    features_auxiliary     = ['signal', 'weight']
    data = data[features_input + features_decorrelation + features_auxiliary]

    # Re-scale weights
    msk = data['signal'].astype(bool)
    num_signal = float(msk.sum())
    data.loc[ msk,'weight'] *= num_signal / data['weight'][ msk].sum()
    data.loc[~msk,'weight'] *= num_signal / data['weight'][~msk].sum()

    # Shuffle and re-index to allow better indexing
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)  # For reproducibility

    # Split
    np.random.seed(seed)  # For reproducibility
    train = np.random.rand(data.shape[0]) < train_fraction
    data['train'] = pd.Series(train, index=data.index)

    # Return
    return data, features_input, features_decorrelation
