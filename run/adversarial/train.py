#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing training adversarial neural network for de-correlated jet tagging."""

# Basic import(s)
import os
import sys
import gzip
import glob
import json
import pickle
import datetime
import subprocess
from pprint import pprint
import logging as log
import itertools

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
import numpy as np
import pandas as pd
import root_numpy
from sklearn.model_selection import StratifiedKFold

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Project import(s)
from adversarial.utils     import *
from adversarial.profile   import *
from adversarial.constants import *
from adversarial.new_utils import parse_args, initialise, load_data, mkdir, kill, save, load, lwtnn_save


# Global variable(s)
rng = np.random.RandomState(21)  # For reproducibility


# Main function definition
@profile
def main (args):

    # Initialisation
    # --------------------------------------------------------------------------
    with Profile("Initialisation"):

        # Initialising
        # --------------------------------------------------------------------------
        args, cfg = initialise(args)

        # Validate train/optimise flags
        if args.optimise_classifier:

            # Stand-alone classifier optimisation
            args.train_classifier  = True
            args.train_adversarial = False
            args.train = False
            cfg['classifier']['fit']['verbose'] = 2

        elif args.optimise_adversarial:

            # Adversarial network optimisation
            args.train_classifier  = False
            args.train_adversarial = True
            args.train = False
            cfg['combined']['fit']['verbose'] = 2

            pass

        # Initialise Keras backend
        initialise_backend(args)

        import keras
        import keras.backend as K
        from keras.utils import multi_gpu_model
        from keras.models import load_model
        from keras.callbacks import Callback, TensorBoard, EarlyStopping
        from keras.utils.vis_utils import plot_model

        # Print setup information
        log.info("Running '%s'" % __file__)
        log.info("Command-line arguments:")
        pprint(vars(args))

        log.info("Configuration file contents:")
        pprint(cfg)

        log.info("Python version: {}".format(sys.version.split()[0]))
        log.info("Numpy  version: {}".format(np.__version__))
        try:
            log.info("Keras  version: {}".format(keras.__version__))
            log.info("Using keras backend: '{}'".format(K.backend()))
            if K.backend() == 'tensorflow':
                import tensorflow
                print "  TensorFlow version: {}".format(tensorflow.__version__)
            else:
                import theano
                print "  Theano version: {}".format(theano.__version__)
                pass
        except NameError: log.info("Keras not imported")

        # Save command-line argument configuration in output directory
        with open(args.output + 'args.json', 'wb') as f:
            json.dump(vars(args), f, indent=4, sort_keys=True)
            pass

        # Save configuration dict in output directory
        with open(args.output + 'config.json', 'wb') as f:
            json.dump(cfg, f, indent=4, sort_keys=True)
            pass

        # Evaluate the 'optimizer' fields for each model, once and for all
        for model in ['classifier', 'combined']:
            opts = cfg[model]['compile']
            opts['optimizer'] = eval("keras.optimizers.{}(lr={}, decay={})" \
                                     .format(opts['optimizer'],
                                             opts.pop('lr'),
                                             opts.pop('decay')))
            pass

        # Multiply batch size by number of devices, to ensure equal splits.
        for model in ['classifier', 'combined']:
            cfg[model]['fit']['batch_size'] *= args.devices
            pass

        # If the `model/architecture` parameter is provided as an int, convert
        # to list of empty dicts
        for network in ['classifier', 'adversary']:
            if isinstance(cfg[network]['model']['architecture'], int):
                cfg[network]['model']['architecture'] = [{} for _ in range(cfg[network]['model']['architecture'])]
                pass
            pass

        # Start TensorBoard instance
        if not args.theano:
            tensorboard_dir = 'logs/tensorboard/{}/'.format('-'.join(re.split('-|:| ', str(datetime.datetime.now()).replace('.', 'T'))) if args.jobname == "" else args.jobname)
            log.info("Writing TensorBoard logs to '{}'".format(tensorboard_dir))
            if args.tensorboard:
                assert not args.theano, "TensorBoard requires TensorFlow backend."

                log.info("Starting TensorBoard instance in background.")
                log.info("The output will be available at:")
                log.info("  http://localhost:6006")
                tensorboard_pid = subprocess.Popen(["tensorboard", "--logdir", tensorboard_dir]).pid
                log.info("TensorBoard has PID {}.".format(tensorboard_pid))
                pass
            pass

        pass


    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data(args.input + 'data.h5')
    data = data[data['train'] == 1]
    num_features = len(features)

    # Get standard-formatted inputs for reweighting regressor
    from run.reweight.common import get_input as reweighter_input
    decorrelation, decorrelation_weight = reweighter_input(data)



    # Re-weighting to flatness
    # --------------------------------------------------------------------------
    if args.reweight:
        with Profile("Re-weighting"):

            # Load reweighting regressor
            with gzip.open('models/reweight/gbreweighter.pkl.gz', 'r') as f:
                reweighter = pickle.load(f)
                pass


            # Compute flatness-boosted weights
            uniform  = reweighter.predict_weights(decorrelation, original_weight=decorrelation_weight)
            uniform *= np.sum(data['weight']) / np.sum(uniform)

            # Store flatness-boosted weights
            data['uniform'] = pd.Series(uniform, index=data.index)
            pass
        pass


    # Classifier-only fit
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, cross-validation"):
        # @TODO:
        # - Implement data generator looping over all of the background and
        # randomly sampling signal events to have equal fractions in each
        # batch. Use DataFlow from Tensorpack?

        # Define variable(s)
        basename = 'crossval_classifier'

        # Get indices for each fold in stratified k-fold training
        # @NOTE: No shuffling is performed -- assuming that's already done above.
        skf = StratifiedKFold(n_splits=args.folds).split(data[features].values,
                                                         data['signal'].values)

        # Import module creator methods and optimiser options
        from adversarial.models import classifier_model, adversary_model, combined_model

        # Import custom callbacks
        from adversarial.callbacks import LossCallback

        # Collection of classifiers and their associated training histories
        classifiers = list()
        histories   = list()

        # Train or load classifiers
        if args.train or args.train_classifier:
            log.info("Training cross-validation classifiers")

            # Loop `k` folds
            for fold, (train, validation) in enumerate(skf):
                with Profile("Fold {}/{}".format(fold + 1, args.folds)):

                    # Map to DataFrame indices
                    train      = data.index[train]
                    validation = data.index[validation]

                    # Define unique name for current classifier
                    name = '{}__{}of{}'.format(basename, fold + 1, args.folds)

                    # Get classifier
                    classifier = classifier_model(num_features, **cfg['classifier']['model'])

                    # Parallelise on GPUs
                    # @NOTE: Store reference to base model to allow for saving.
                    #        Cf. [https://github.com/keras-team/keras/issues/8446#issuecomment-343559454]
                    if (not args.theano) and args.gpu and args.devices > 1:
                        parallelised = multi_gpu_model(classifier, args.devices)
                    else:
                        parallelised = classifier
                        pass

                    # Compile model (necessary to save properly)
                    parallelised.compile(**cfg['classifier']['compile'])

                    # Prepare arrays
                    X = data.loc[train, features].values
                    Y = data.loc[train, 'signal'].values
                    W = data.loc[train, 'weight'].values
                    validation_data = (
                        data.loc[validation, features].values,
                        data.loc[validation, 'signal'].values,
                        data.loc[validation, 'weight'].values
                    )

                    # Create callbacks
                    callbacks = []

                    # -- Loss logging, for debugging
                    #callbacks += [LossCallback(train=(X,Y,W), validation=validation_data)]

                    # -- TensorBoard
                    if not args.theano:
                        callbacks += [TensorBoard(log_dir=tensorboard_dir + 'classifier/fold{}/'.format(fold))]
                        pass


                    # Fit classifier model
                    ret = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, callbacks=callbacks, **cfg['classifier']['fit'])

                    # Add to list of cost histories
                    histories.append(ret.history)

                    # Add to list of classifiers
                    classifiers.append(classifier)

                    # Save classifier model and training history to file, both
                    # in unique output directory and in the directory for pre-
                    # trained classifiers
                    for destination in [args.output, 'models/adversarial/classifier/crossval/']:
                        save(destination, name, classifier, ret.history)
                        pass

                    pass
                pass # end: k-fold cross-validation
            pass
        else:

            # Load pre-trained classifiers
            log.info("Loading cross-validation classifiers from file")
            try:
                basedir = 'models/adversarial/classifier/crossval/'
                for fold in range(args.folds):
                    name = '{}__{}of{}'.format(basename, fold + 1, args.folds)
                    classifier, history = load(basedir, name)
                    classifiers.append(classifier)
                    histories.append(history)
                    pass
            except IOError as err:
                log.error("[ERROR] {}".format(err))
                log.error("[ERROR] Not all files were loaded. Exiting.")
                return 1

            pass # end: train/load
        pass


    # Early stopping in case of stand-alone classifier optimisation
    # --------------------------------------------------------------------------
    if args.optimise_classifier:
        # Kill TensorBoard
        if args.tensorboard:
            kill(tensorboard_pid, "TensorBoard")
            pass

        # Compute average validation loss
        val_avg = np.mean([hist['val_loss'] for hist in histories], axis=0)
        return np.min(val_avg)


    # Classifier-only fit, full
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, full"):

        # Define variable(s)
        name = 'full_classifier'

        if args.train or args.train_classifier:
            log.info("Training full classifier")

            # Get classifier
            classifier = classifier_model(num_features, **cfg['classifier']['model'])

            # Save classifier model diagram to file
            plot_model(classifier, to_file=args.output + 'model_classifier.png', show_shapes=True)

            # Parallelise on GPUs
            if (not args.theano) and args.gpu and args.devices > 1:
                parallelised = multi_gpu_model(classifier, args.devices)
            else:
                parallelised = classifier
                pass

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['classifier']['compile'])

            # Create callbacks
            callbacks = []

            # -- TensorBoard
            if not args.theano:
                callbacks += [TensorBoard(log_dir=tensorboard_dir + 'classifier/')]
                pass

            # Prepare arrays
            X = data[features].values
            Y = data['signal'].values
            W = data['weight'].values

            # Fit classifier model
            ret = parallelised.fit(X, Y, sample_weight=W, callbacks=callbacks, **cfg['classifier']['fit'])

            # Save classifier model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            for destination in [args.output, 'models/adversarial/classifier/full/']:
                save(destination, name, classifier, ret.history)
                pass

        else:

            # Load pre-trained classifier
            log.info("Loading full classifier from file")
            basedir = 'models/adversarial/classifier/full/'
            classifier, history = load(basedir, name)

            pass # end: train/load
        pass


    # Saving classifier in lwtnn-friendly format.
    # --------------------------------------------------------------------------
    lwtnn_save(classifier, 'nn')


    # Combined fit, full (@TODO: Cross-val?)
    # --------------------------------------------------------------------------
    with Profile("Combined fit, full"):
        # @TODO:
        # - Checkpointing

        # Define variables
        name = 'full_combined'

        # Set up adversary
        adversary = adversary_model(gmm_dimensions=decorrelation.shape[1],
                                    **cfg['adversary']['model'])

        # Save adversarial model diagram
        plot_model(adversary, to_file=args.output + 'model_adversary.png', show_shapes=True)

        # Create callback array
        callbacks = list()

        # Create callback logging the adversary p.d.f.'s during training
        #callback_posterior = PosteriorCallback(data, args, adversary)

        # Create callback logging the adversary p.d.f.'s during training
        #callback_profiles  = ProfilesCallback(data, args, classifier)

        # (opt.) List all callbacks to be used
        #if args.plot:
        #    callbacks += [callback_posterior, callback_profiles]
        #    pass

        # (opt.) Add TensorBoard callback
        if not args.theano:
            callbacks += [TensorBoard(log_dir=tensorboard_dir + 'adversarial/')]
            pass

        # Set up combined, adversarial model
        combined = combined_model(classifier, adversary, **cfg['combined']['model'])

        # Save combined model diagram
        plot_model(combined, to_file=args.output + 'model_combined.png', show_shapes=True)

        if args.train or args.train_adversarial:
            log.info("Training full, combined model")

            # Create custom Kullback-Leibler (KL) divergence cost.
            eps = np.finfo(float).eps
            def kullback_leibler (p_true, p_pred):
                return - K.log(p_pred)  # + eps)

            cfg['combined']['compile']['loss'][1] = kullback_leibler

            # Parallelise on GPUs
            if (not args.theano) and args.gpu and args.devices > 1:
                parallelised = multi_gpu_model(combined, args.devices)
            else:
                parallelised = combined
                pass

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['combined']['compile'])

            # Prepare arrays
            # @TODO:
            # - Properly handle non-reweighted case
            X = [data[features].values, decorrelation]
            Y = [data['signal'].values.astype(K.floatx()), np.ones_like(data['signal'].values).astype(K.floatx())]
            #W = [data['weight'].values, np.multiply(data['uniform'].values, 1. - data['signal'].values)]
            W = [data['weight'].values, np.multiply(data['weight'].values, 1. - data['signal'].values)]

            # Fit classifier model
            ret = parallelised.fit(X, Y, sample_weight=W, callbacks=callbacks, **cfg['combined']['fit'])

            # Save combined model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            for destination in [args.output, 'models/adversarial/combined/full/']:
                save(destination, name, combined, ret.history)
                """
                combined.save        (destination + '{}.h5'        .format(name))
                combined.save_weights(destination + '{}_weights.h5'.format(name))
                with open(destination + 'history__{}.json'.format(name), 'wb') as f:
                    json.dump(ret.history, f)
                    pass
                    """
                pass

        else:

            # Load pre-trained combined _weights_ from file, in order to
            # simultaneously load the embedded classifier so as to not have to
            # extract it manually afterwards.
            log.info("Loading full, combined model from file")
            basedir = 'models/adversarial/combined/full/'
            model, history = load(basedir, name, model=combined)
            """
            mkdir('models/adversarial/full/combined/')
            combined_weights_file = 'models/adversarial/combined/full/{}_weights.h5'.format(name)
            combined.load_weights(combined_weights_file)

            # Load associated training histories
            history_file = 'models/adversarial/combined/full/history__{}.json'.format(name)
            with open(history_file, 'r') as f:
                history = json.load(f)
                pass
            """

            pass # end: train/load

        pass


    # Early stopping in case of adversarial network
    # --------------------------------------------------------------------------
    if args.optimise_adversarial:
        # Kill TensorBoard
        if args.tensorboard:
            kill(tensorboard_pid, "TensorBoard")
            pass

        # @TODO:
        # - Decide on proper metric!
        #   - clf_loss - lambda * <JSD( pass(m) || fail(m) )>?
        #   - Stratified k-fold cross-validation?
        return None


    # Saving adversarially trained classifier in lwtnn-friendly format.
    # --------------------------------------------------------------------------
    lwtnn_save(classifier, 'ann')


    # Clean-up
    # --------------------------------------------------------------------------
    if args.tensorboard:
        kill(tensorboard_pid, "TensorBoard")
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(adversarial=True)

    # Call main function
    main(args)
    pass
