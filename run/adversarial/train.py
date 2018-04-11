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


# Global variable(s)
RNG = np.random.RandomState(21)  # For reproducibility


# Main function definition
@profile
def main (args):

    # Initialisation
    # --------------------------------------------------------------------------
    with Profile("Initialisation"):

        # Initialising
        # ----------------------------------------------------------------------
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

        cfg['classifier']['fit']['verbose'] = 2  # @TEMP
        cfg['combined']  ['fit']['verbose'] = 2  # @TEMP

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
                                             opts.pop('decay', 0)))
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

    log.info("Found {:7.0f} training and {:7.0f} test samples for signal".format(
        sum((data['signal'] == 1) & (data['train'] == 1)),
        sum((data['signal'] == 1) & (data['train'] == 0))
        ))
    log.info("Found {:7.0f} training and {:7.0f} test samples for background".format(
        sum((data['signal'] == 0) & (data['train'] == 1)),
        sum((data['signal'] == 0) & (data['train'] == 0))
        ))

    data = data[data['train'] == 1]
    num_features = len(features)

    # Regulsarisation parameter
    lambda_reg = cfg['combined']['model']['lambda_reg']  # Use same `lambda` as the adversary
    digits = int(np.ceil(max(-np.log10(lambda_reg), 0)))
    lambda_str = '{l:.{d:d}f}'.format(d=digits,l=lambda_reg).replace('.', 'p')

    # Get standard-formatted decorrelation inputs
    decorrelation = get_decorrelation_variables(data)

    # Specify common weights
    # -- Classifier
    weight_var = 'weight_train'
    data['weight_clf'] = pd.Series(data[weight_var].values, index=data.index)

    # -- Adversary
    weight_var = 'weight_train'  # 'weight' / 'weight_flat'
    data['weight_adv'] = pd.Series(np.multiply(data[weight_var].values, 1. - data['signal'].values), index=data.index)
    data['weight_adv'] /= data['weight_adv'].mean()


    # Classifier-only fit, cross-validation
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, cross-validation"):
        # @TODO:
        # - Implement data generator looping over all of the background and
        # randomly sampling signal events to have equal fractions in each
        # batch. Use DataFlow from Tensorpack?

        # Define variable(s)
        basename = 'crossval_classifier'
        basedir  = 'models/adversarial/classifier/crossval/'

        # Get indices for each fold in stratified k-fold training
        # @NOTE: No shuffling is performed -- assuming that's already done above.
        skf = StratifiedKFold(n_splits=args.folds).split(data[features].values,
                                                         data['signal'].values)

        # Import module creator methods and optimiser options
        from adversarial.models import classifier_model, adversary_model, combined_model, decorrelation_model

        # Collection of classifiers and their associated training histories
        classifiers = list()
        histories   = list()

        # Train or load classifiers
        if args.train or args.train_classifier:
            log.info("Training cross-validation classifiers")

            # Loop `k` folds
            for fold, (train, validation) in enumerate(skf):
                with Profile("Fold {}/{}".format(fold + 1, args.folds)):

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
                    X = data[features].values[train]
                    Y = data['signal'].values[train]
                    W = data['weight_clf'].values[train]
                    validation_data = (
                        data[features].values[validation],
                        data['signal'].values[validation],
                        data['weight_clf'].values[validation]
                    )

                    # Create callbacks
                    callbacks = []

                    # -- TensorBoard
                    if not args.theano:
                        callbacks += [TensorBoard(log_dir=tensorboard_dir + 'classifier/fold{}/'.format(fold))]
                        pass

                    # Compute initial losses
                    X_val, Y_val, W_val = validation_data
                    eval_opts = dict(batch_size=cfg['classifier']['fit']['batch_size'], verbose=0)
                    #K.set_learning_phase(1)  # Manually set to training phase, for consistent comparison
                    initial_losses = [[parallelised.evaluate(X,     Y,     sample_weight=W,     **eval_opts)],
                                      [parallelised.evaluate(X_val, Y_val, sample_weight=W_val, **eval_opts)]]

                    # Fit classifier model
                    ret = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, callbacks=callbacks, **cfg['classifier']['fit'])

                    # Prepend initial losses
                    for metric, loss_train, loss_val in zip(parallelised.metrics_names, *initial_losses):
                        ret.history[metric]         .insert(0, loss_train)
                        ret.history['val_' + metric].insert(0, loss_val)
                        pass

                    # Add to list of cost histories
                    histories.append(ret.history)

                    # Add to list of classifiers
                    classifiers.append(classifier)

                    # Save classifier model and training history to file, both
                    # in unique output directory and in the directory for pre-
                    # trained classifiers
                    for destination in [args.output, basedir]:
                        save(destination, name, classifier, ret.history)
                        pass

                    pass
                pass # end: k-fold cross-validation
            pass
        else:

            # Load pre-trained classifiers
            log.info("Loading cross-validation classifiers from file")
            try:
                for fold in range(args.folds):
                    name = '{}__{}of{}'.format(basename, fold + 1, args.folds)
                    classifier, history = load(basedir, name)
                    classifiers.append(classifier)
                    histories.append(history)
                    pass
            except IOError as err:
                log.error("{}".format(err))
                log.error("Not all files were loaded. Exiting.")
                #return 1  # @TEMP
                pass

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
        name    = 'classifier'
        basedir = 'models/adversarial/classifier/full/'

        if args.train or args.train_classifier:
            log.info("Training full classifier")

            # Get classifier
            classifier = classifier_model(num_features, **cfg['classifier']['model'])

            # Save classifier model diagram to file
            plot_model(classifier, to_file=args.output + 'model_{}.png'.format(name), show_shapes=True)

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
                callbacks += [TensorBoard(log_dir=tensorboard_dir + name + '/')]
                pass

            # Prepare arrays
            X = data[features].values
            Y = data['signal'].values
            W = data['weight_clf'].values

            # Fit classifier model
            ret = parallelised.fit(X, Y, sample_weight=W, callbacks=callbacks, **cfg['classifier']['fit'])

            # Save classifier model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            for destination in [args.output, basedir]:
                save(destination, name, classifier, ret.history)
                pass

        else:

            # Load pre-trained classifier
            log.info("Loading full classifier from file")
            classifier, history = load(basedir, name)
            pass # end: train/load
        pass


    # Saving classifier in lwtnn-friendly format.
    # --------------------------------------------------------------------------
    lwtnn_save(classifier, 'nn')


    # Definitions for adversarial training
    # --------------------------------------------------------------------------
    # Create custom Kullback-Leibler (KL) divergence cost.
    def kullback_leibler (p_true, p_pred):
        return -K.log(p_pred)

    cfg['combined']['compile']['loss'][1] = kullback_leibler

    pretrain_epochs = 20  # @TODO: Make configurable  # `1` leads to good performance.


    # Combined adversarial fit, cross-validation
    # --------------------------------------------------------------------------
    """ @TEMP
    with Profile("Combined adversarial fit, cross-validation"):
        # @TODO:
        # - Checkpointing

        # Define variables
        basename = 'combined_lambda{}'.format(lambda_str)
        basedir  = 'models/adversarial/combined/crossval/'

        # Get indices for each fold in stratified k-fold training
        # @NOTE: No shuffling is performed -- assuming that's already done above.
        skf = StratifiedKFold(n_splits=args.folds).split(data[features].values,
                                                         data['signal'].values)

        if args.train or args.train_adversarial:
            log.info("Training combined model cross-validation")

            # Loop `k` folds
            for fold, (train, validation) in enumerate(skf):
                with Profile("Fold {}/{}".format(fold + 1, args.folds)):

                    # Define unique name for current classifier
                    name = '{}__{}of{}'.format(basename, fold + 1, args.folds)

                    # Load pre-trained classifier
                    classifier, _ = load('models/adversarial/classifier/full/', 'classifier')

                    # Set up adversary
                    adversary = adversary_model(gmm_dimensions=decorrelation.shape[1],
                                                **cfg['adversary']['model'])

                    # Set up combined, adversarial model
                    combined = combined_model(classifier, adversary, **cfg['combined']['model'])

                    # Parallelise on GPUs
                    if (not args.theano) and args.gpu and args.devices > 1:
                        parallelised = multi_gpu_model(combined, args.devices)
                    else:
                        parallelised = combined
                        pass

                    # Prepare arrays
                    # @TODO:
                    # - Properly handle non-reweighted case
                    W2_train      = np.multiply(data[weight_var].values[train],      1. - data['signal'].values[train])
                    W2_validation = np.multiply(data[weight_var].values[validation], 1. - data['signal'].values[validation])
                    W2_train      /= np.mean(W2_train)       # @TEMP
                    W2_validation /= np.mean(W2_validation)  # @TEMP
                    #### W2_train      *= 2.  # @TEMP
                    #### W2_validation *= 2.  # @TEMP

                    X = [data[features].values[train], decorrelation[train]]
                    Y = [data['signal'].values[train], np.ones_like(data['signal'].values[train]).astype(K.floatx())]
                    W = [data['weight_clf'].values[train], data['weight_adv'].values[train]]
                    validation_data = (
                        [data[features].values[validation], decorrelation[validation]],
                        [data['signal'].values[validation], np.ones_like(data['signal'].values[validation]).astype(K.floatx())],
                        [data['weight_clf'].values[validation], data['weight_adv'].values[validation]]
                    )
                    #W = [data['weight'].values, np.multiply(data['weight_flatness'].values, 1. - data['signal'].values)]
                    #W = [data['weight'].values, np.multiply(data['weight'].values, 1. - data['signal'].values)]

                    # Pre-training adversary
                    log.info("Pre-training")
                    classifier.trainable = False

                    # Compile model for pre-training
                    save_lr = K.get_value(cfg['combined']['compile']['optimizer'].lr)
                    K.set_value(cfg['combined']['compile']['optimizer'].lr, save_lr)
                    parallelised.compile(**cfg['combined']['compile'])

                    log.info("Learning rate before pre-training:  {}".format(K.get_value(cfg['combined']['compile']['optimizer'].lr)))

                    # Compute initial losses
                    log.info("Computing initial loss")
                    X_val, Y_val, W_val = validation_data
                    eval_opts = dict(batch_size=cfg['combined']['fit']['batch_size'], verbose=0)
                    K.set_learning_phase(1)  # Manually set to training phase, for consistent comparison
                    initial_losses = [parallelised.evaluate(X,     Y,     sample_weight=W,     **eval_opts),
                                      parallelised.evaluate(X_val, Y_val, sample_weight=W_val, **eval_opts)]

                    pretrain_fit_opts = dict(**cfg['combined']['fit'])
                    pretrain_fit_opts['epochs'] = pretrain_epochs
                    ret_pretrain = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, **pretrain_fit_opts)

                    # Fit classifier model
                    log.info("Actual training")
                    classifier.trainable = True

                    # Re-compile combined model for full training
                    K.set_value(cfg['combined']['compile']['optimizer'].lr, save_lr)
                    parallelised.compile(**cfg['combined']['compile'])
                    log.info("Learning rate before full training: {}".format(K.get_value(cfg['combined']['compile']['optimizer'].lr)))

                    ret = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, **cfg['combined']['fit'])

                    # Compute final losses (check) @TEMP
                    log.info("Computing final loss")
                    final_losses = [parallelised.evaluate(X,     Y,     sample_weight=W,     **eval_opts),
                                    parallelised.evaluate(X_val, Y_val, sample_weight=W_val, **eval_opts)]

                    # Prepend initial losses
                    for metric, loss_train, loss_val in zip(parallelised.metrics_names, *initial_losses):
                        ret_pretrain.history[metric]         .insert(0, loss_train)
                        ret_pretrain.history['val_' + metric].insert(0, loss_val)
                        pass

                    for metric in parallelised.metrics_names:
                        ret.history[metric]          = ret_pretrain.history[metric]          + ret.history[metric]
                        ret.history['val_' + metric] = ret_pretrain.history['val_' + metric] + ret.history['val_' + metric]
                        pass

                    #### for metric, loss_train, loss_val in zip(parallelised.metrics_names, *final_losses):
                    ####     ret.history[metric]         .append(loss_train)
                    ####     ret.history['val_' + metric].append(loss_val)
                    ####     pass

                    # Save combined model and training history to file, both in unique
                    # output directory and in the directory for pre-trained classifiers.
                    for destination in [args.output, basedir]:
                        print "Saving {} to {}".format(name, destination)
                        save(destination, name, combined, ret.history)
                        pass
                    pass
                pass
            pass
        pass
        """

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


    # Combined adversarial fit, full
    # --------------------------------------------------------------------------
    with Profile("Combined adversarial fit, full"):
        # @TODO:
        # - Checkpointing

        # Define variables
        name    = 'combined_lambda{}'.format(lambda_str)
        basedir = 'models/adversarial/combined/full/'

        # Load pre-trained classifier
        classifier, _ = load('models/adversarial/classifier/full/', 'classifier')

        # Set up adversary
        adversary = adversary_model(gmm_dimensions=decorrelation.shape[1],
                                    **cfg['adversary']['model'])

        # Save adversarial model diagram
        plot_model(adversary, to_file=args.output + 'model_adversary.png', show_shapes=True)

        # Create callback array
        callbacks = list()

        # (opt.) Add TensorBoard callback
        if not args.theano:
            callbacks += [TensorBoard(log_dir=tensorboard_dir + 'adversarial/')]
            pass

        # Set up combined, adversarial model
        combined = combined_model(classifier, adversary, **cfg['combined']['model'])

        # Save combined model diagram
        plot_model(combined, to_file=args.output + 'model_{}.png'.format(name), show_shapes=True)

        if args.train or args.train_adversarial:
            log.info("Training full, combined model")

            # Parallelise on GPUs
            if (not args.theano) and args.gpu and args.devices > 1:
                parallelised = multi_gpu_model(combined, args.devices)
            else:
                parallelised = combined
                pass

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['combined']['compile'])

            # Prepare arrays
            X = [data[features].values, decorrelation]
            Y = [data['signal'].values.astype(K.floatx()), np.ones_like(data['signal'].values).astype(K.floatx())]
            W = [data['weight_clf'].values, data['weight_adv'].values]

            # Pre-training adversary
            log.info("Pre-training")
            classifier.trainable = False

            # Compile model for pre-training
            save_lr = K.get_value(cfg['combined']['compile']['optimizer'].lr)
            K.set_value(cfg['combined']['compile']['optimizer'].lr, save_lr)
            parallelised.compile(**cfg['combined']['compile'])

            log.info("Learning rate before pre-training:  {}".format(K.get_value(cfg['combined']['compile']['optimizer'].lr)))

            pretrain_fit_opts = dict(**cfg['combined']['fit'])
            pretrain_fit_opts['epochs'] = pretrain_epochs
            ret_pretrain = parallelised.fit(X, Y, sample_weight=W, **pretrain_fit_opts)

            # Fit classifier model
            log.info("Actual training")
            classifier.trainable = True

            # Re-compile combined model for full training
            K.set_value(cfg['combined']['compile']['optimizer'].lr, save_lr)
            parallelised.compile(**cfg['combined']['compile'])
            log.info("Learning rate before full training: {}".format(K.get_value(cfg['combined']['compile']['optimizer'].lr)))

            # Fit classifier model
            ret = parallelised.fit(X, Y, sample_weight=W, callbacks=callbacks, **cfg['combined']['fit'])

            # Save combined model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            for destination in [args.output, basedir]:
                save(destination, name, combined, ret.history)
                pass

        else:

            # Load pre-trained combined _weights_ from file, in order to
            # simultaneously load the embedded classifier so as to not have to
            # extract it manually afterwards.
            log.info("Loading full, combined model from file")
            combined, history = load(basedir, name, model=combined)
            pass # end: train/load

        pass


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
