#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common utility methods for de-correlated jet tagging."""

# Basic import(s)
import os
import sys
import json
import logging as log
import argparse
import subprocess

# Scientific import(s)
import numpy as np
rng = np.random.RandomState(21)  # For reproducibility
import pandas as pd

# Project import(s)
import adversarial
from adversarial.profile import profile

# Global variables  # @TODO: Necessary?
PROJECTDIR='/'.join(os.path.realpath(adversarial.__file__).split('/')[:-2] + [''])


def parse_args (cmdline_args=sys.argv[1:], backend=False, adversarial=False, plots=False):
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

    # Inputs
    parser.add_argument('--input',  action='store', type=str,
                        default=PROJECTDIR + 'data/', help='Input directory, from which to read HDF5 data file.')
    parser.add_argument('--output', action='store', type=str,
                        default=PROJECTDIR + 'output/', help='Output directory, to which to write results.')
    parser.add_argument('--config', action='store', type=str,
                        default=PROJECTDIR + 'configs/default.json', help='Configuration file.')
    parser.add_argument('--patch', dest='patches', action='append', type=str,
                        help='Patch file(s) with which to update configuration file.')

    # Flags
    parser.add_argument('--verbose', action='store_true', help='Print verbose')

    # Conditional arguments
    if backend or adversarial:
        # Inputs
        parser.add_argument('--devices', action='store', type=int,
                            default=1, help='Number of CPU/GPU devices to use with TensorFlow.')
        parser.add_argument('--folds',   action='store', type=int,
                            default=3, help='Number of folds to use for stratified cross-validation.')

        # Flags
        parser.add_argument('--gpu',    action='store_true', help='Run on GPU')
        parser.add_argument('--theano', action='store_true', help='Use Theano backend')
        pass

    if adversarial:
        # Inputs
        parser.add_argument('--jobname', action='store', type=str,
                            default="", help='Name of job, used for TensorBoard output.')

        # Flags
        parser.add_argument('--tensorboard', action='store_true',
                            help='Use TensorBoard for monitoring')
        parser.add_argument('--reweight',    action='store_true',
                            help='Reweight background to flatness in adversarial training')
        parser.add_argument('--train',       action='store_true',
                            help='Perform training')
        parser.add_argument('--train-classifier', dest='train_classifier', action='store_true',
                            help='Perform classifier pre-training')
        parser.add_argument('--train-adversarial', dest='train_adversarial', action='store_true',
                            help='Perform adversarial training')

        group_optimise = parser.add_mutually_exclusive_group()
        group_optimise.add_argument('--optimise-classifier',  dest='optimise_classifier',  action='store_true',
                            help='Optimise stand-alone classifier')
        group_optimise.add_argument('--optimise-adversarial', dest='optimise_adversarial', action='store_true',
                            help='Optimise adversarial network')
        pass

    if plots:
        # Flags
        parser.add_argument('--save', action='store_true', help='Save plots to file')
        parser.add_argument('--show', action='store_true', help='Show plots')
        pass

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

    # Try adding `mode` field manually
    try:
        args = argparse.Namespace(mode='gpu' if args.gpu else 'cpu', **vars(args))
    except AttributeError:
        # No field `gpu`
        pass

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

    # Set adversary learning rate (LR) ratio from ratio of loss_weights
    try:
        cfg['combined']['model']['lr_ratio'] = cfg['combined']['compile']['loss_weights'][0] / \
                                               cfg['combined']['compile']['loss_weights'][1]
    except KeyError:
        # ...
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
    features_auxiliary     = ['signal', 'weight', 'pt']
    data = data[features_input + features_decorrelation + features_auxiliary]

    # Re-scale weights
    msk = data['signal'].astype(bool)
    num_signal = float(msk.sum())
    data.loc[ msk,'weight'] *= num_signal / data['weight'][ msk].sum()
    data.loc[~msk,'weight'] *= num_signal / data['weight'][~msk].sum()

    # Shuffle and re-index to allow better indexing
    data = data.sample(frac=1, random_state=rng).reset_index(drop=True)  # For reproducibility

    # Split
    train = rng.rand(data.shape[0]) < train_fraction
    data['train'] = pd.Series(train, index=data.index)

    # Return
    return data, features_input, features_decorrelation


def mkdir (path):
    """Script to ensure that the directory at `path` exists.

    Arguments:
        path: String specifying path to directory to be created.
    """

    # Check mether  output directory exists
    if not os.path.exists(path):
        print "mdkir: Creating output directory:\n  {}".format(path)
        try:
            os.makedirs(path)
        except OSError:
            # Apparently, `path` already exists.
            pass
        pass
    return


def kill (pid, name=None):
    """Query user to kill system process `pid`.

    Arguments:
        pid: ID of process to be killed assumed to be owned by user.
        name: Name of process.
    """

    # Get username
    username = subprocess.check_output(['whoami']).strip()

    # Get processes belonging to user
    lines = subprocess.check_output(["ps", "-u", username]).split('\n')[1:-1]

    # Get PIDs of processes belonging to user
    pids = [int(line.strip().split()[0]) for line in lines]

    # Check PID is suitable for deletion
    if int(pid) not in pids:
        print "[WARN]  No process with PID {} belonging to user {} was found running. Exiting.".format(pid, username)
        return

    # Query user
    print "[INFO]  {1} process ({0}) is running in background. Enter `q` to close it. Enter anything else to quit the program while leaving {1} running.".format(pid, name)
    response = raw_input(">> ")
    if response.strip() == 'q':
        subprocess.call(["kill", str(pid)])
    else:
        print "[INFO]  {} process is left running. To manually kill it later, do:".format(name)
        print "[INFO]    $ kill {}".format(pid)
        pass
    return


def save (basedir, name, model, history=None):
    """Standardised method to save Keras models to file.

    Arguments:
        basedir: Directory in which models should be saved. Is created if it
            doesn't already exist.
        name: Name of model to be saved, used in filenames.
        model: Keras model to be saved.
        history: Container with logged training history.
    """

    # Make sure output directory exists
    mkdir(basedir)

    # Save full model and model weights
    model.save        (basedir + '{}.h5'        .format(name))
    model.save_weights(basedir + '{}_weights.h5'.format(name))

    # Save training history
    if history is not None:
        with open(basedir + 'history__{}.json'.format(name), 'wb') as f:
            json.dump(history, f)
            pass
        pass
    return


def load (basedir, name, model=None):
    """Standardised method to load Keras models from file.
    If a pre-existing model is specified only weights are loaded into the model.

    Arguments:
        basedir: Directory from which models should be loaded.
        name: Name of model to be loaded, used in filenames.
        model: Pre-existing model.

    Returns:
        model: Keras model to be saved.
        history: Container with logged training history.

    Raises:
        IOError: If any of the attempted files do not exist.
    """

    # Import(s)
    from keras.models import load_model

    # Load full pre-trained model or model weights
    if model is None:
        model = load_model(basedir + '{}.h5'.format(name))
    else:
        model.load_weights(basedir + '{}_weights.h5'.format(name))
        pass

    # Load associated training histories
    try:
        history_file = basedir + 'history__{}.json'.format(name)
        with open(history_file, 'r') as f:
            history = json.load(f)
            pass
    except:
        print "[WARN] Could not find history file {}."
        history = None
        pass

    return model, history


def lwtnn_save(model, name, basedir='models/adversarial/lwtnn/'):
    """Method for saving classifier in lwtnn-friendly format.
    See [https://github.com/lwtnn/lwtnn/wiki/Keras-Converter]
    """
    # Check(s).
    if not basedir.endswith('/'):
        basedir += '/'
        pass

    # Make sure output directory exists
    mkdir(basedir)

    # Get the architecture as a json string
    arch = model.to_json()

    # Save the architecture string to a file
    with open(basedir + name + '_architecture.json', 'w') as arch_file:
        arch_file.write(arch)
        pass

    # Now save the weights as an HDF5 file
    model.save_weights(basedir + name + '_weights.h5')

    # Save full model to HDF5 file
    model.save(basedir + name + '.h5')
    return
