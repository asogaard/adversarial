#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common, setup-related utilities."""

# Basic import(s)
import os
import sys
import json
import argparse
import subprocess
import logging as log

# Scientific import(s)
import numpy as np
import pandas as pd

# Project import(s)
from .management import mkdir
from ..profile import profile

# Global variable definition(s)
RNG = np.random.RandomState(21)  # For reproducibility


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
                        default='./data/', help='Input directory, from which to read HDF5 data file.')
    parser.add_argument('--output', action='store', type=str,
                        default='./output/', help='Output directory, to which to write results.')
    parser.add_argument('--config', action='store', type=str,
                        default='./configs/default.json', help='Configuration file.')
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
        parser.add_argument('--train',       action='store_true',
                            help='Perform training')
        parser.add_argument('--train-classifier', action='store_true',
                            help='Perform classifier pre-training')
        parser.add_argument('--train-adversarial', action='store_true',
                            help='Perform adversarial training')
        parser.add_argument('--train-aux', action='store_true',
                            help='Train auxiliary mass-decorrelation methods')

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


def flatten (container):
    """Unravel nested lists and tuples.

    From [https://stackoverflow.com/a/10824420]
    """
    if isinstance(container, (list,tuple)):
        for i in container:
            if isinstance(i, (list,tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i
            pass
    else:
        yield container


def apply_patch (d, u):
    """Update nested dictionary without overwriting previous levels.

    From [https://stackoverflow.com/a/3233356]
    """
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = apply_patch(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
            pass
        pass
    return d


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
    mkdir(args.output)

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

    # @TODO: Scale loss_weights[0] by 1./(1. + lambda_reg)?
    cfg['combined']['compile']['loss_weights'][0] *= 1./(1. + cfg['combined']['model']['lambda_reg'])
    
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
def configure_theano (args, num_cores):
    """
    Backend-specific method to configure Theano.

    Arguments:
        args: Namespace containing command-line arguments from argparse. These
            settings specify which back-end should be configured, and how.
        num_cores: Number of CPU cores available for parallelism.
    """

    # Check(s)
    if args.devices > 1:
        log.warning("Currently it is not possible to specify more than one devices for Theano backend.")
        pass

    if not args.gpu:
        # Set number of OpenMP threads to use; even if 1, set to force use of
        # OpenMP which doesn't happen otherwise, for some reason. Gives speed-up
        # of factor of ca. 6-7. (60 sec./epoch -> 9 sec./epoch)
        os.environ['OMP_NUM_THREADS'] = str(num_cores * 2)
        pass

    # Switch: CPU/GPU
    cuda_version = '8.0.61'
    standard_flags = [
        'device={}'.format('cuda' if args.gpu else 'cpu'),
        'openmp=True',
        ]
    dnn_flags = [
        'dnn.enabled=True',
        'dnn.include_path=/exports/applications/apps/SL7/cuda/{}/include/'.format(cuda_version),
        'dnn.library_path=/exports/applications/apps/SL7/cuda/{}/lib64/'  .format(cuda_version),
        ]
    os.environ["THEANO_FLAGS"] = ','.join(standard_flags + (dnn_flags if args.gpu else []))

    return None


@profile
def configure_tensorflow (args, num_cores):
    """
    Backend-specific method to configure Theano.

    Arguments:
        args: Namespace containing command-line arguments from argparse. These
            settings specify which back-end should be configured, and how.
        num_cores: Number of CPU cores available for parallelism.
    """

    # Set print level to avoid unecessary warnings, e.g.
    #  $ The TensorFlow library wasn't compiled to use <SSE4.1, ...>
    #  $ instructions, but these are available on your machine and could
    #  $ speed up CPU computations.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Load the tensorflow module here to make sure only the correct
    # GPU devices are set up
    import tensorflow as tf

    # Manually configure Tensorflow session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1,
                                allow_growth=True)

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores * 2,
                            inter_op_parallelism_threads=num_cores * 2,
                            allow_soft_placement=True,
                            device_count={'GPU': args.devices if args.gpu else 0},
                            gpu_options=gpu_options if args.gpu else None)

    session = tf.Session(config=config)

    return session


@profile
def initialise_backend (args):
    """Initialise the Keras backend.

    Args:
        args: Namespace containing command-line arguments from argparse. These
            settings specify which back-end should be configured, and how.
    """

    # Check(s)
    assert 'keras' not in sys.modules, \
        "initialise_backend: Keras was imported before initialisation."

    if args.gpu and args.theano and args.devices > 1:
        raise NotImplementedError("Distributed training on GPUs is current not enabled.")

    # Specify Keras backend and import module
    os.environ['KERAS_BACKEND'] = "theano" if args.theano else "tensorflow"

    # Get number of cores on CPU(s), name of CPU devices, and number of physical
    # cores in each device.
    try:
        cat_output = subprocess.check_output(["cat", "/proc/cpuinfo"]).split('\n')
        num_cpus  = len(filter(lambda line: line.startswith('cpu cores'),  cat_output))
        name_cpu  =     filter(lambda line: line.startswith('model name'), cat_output)[0] \
                        .split(':')[-1].strip()
        num_cores = int(filter(lambda line: line.startswith('cpu cores'),  cat_output)[0] \
                        .split(':')[-1].strip())
        log.info("Found {} {} devices with {} cores each.".format(num_cpus, name_cpu, num_cores))
    except subprocess.CalledProcessError:
        # @TODO: Implement CPU information for macOS
        num_cores = 1
        log.warning("Could not retrieve CPU information -- probably running on macOS. Therefore, multi-core running is disabled.")
        pass

    # Configure backend
    if args.theano:
        _       = configure_theano(args, num_cores)
    else:
        session = configure_tensorflow(args, num_cores)
        pass

    # Import Keras backend
    import keras.backend as K
    K.set_floatx('float32')

    if not args.theano:
        # Set global Tensorflow session
        K.set_session(session)
        pass

    return


@profile
def load_data (path, name='dataset', train_fraction=0.8):
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
    data = data.sample(frac=1, random_state=RNG).reset_index(drop=True)  # For reproducibility

    # Split
    train = RNG.rand(data.shape[0]) < train_fraction
    data['train'] = pd.Series(train, index=data.index)

    # Return
    return data, features_input, features_decorrelation
