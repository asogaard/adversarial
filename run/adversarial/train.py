#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing training adversarial neural network for de-correlated jet tagging."""

# Basic import(s)
import glob
import pickle
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
from .common import *


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
        from keras.models import load_model
        from keras.callbacks import Callback, TensorBoard, EarlyStopping
        from keras.utils.vis_utils import plot_model

        # Neural network-specific initialisation of the configuration dict
        initialise_config(args, cfg)

        # Setup TensorBoard, if applicable
        tensorboard_dir = initialise_tensorboard(args, cfg)

        # Print the current environment setup
        print_env(args, cfg)
        pass


    # Loading data
    # --------------------------------------------------------------------------
    data, features, features_decorrelation = load_data(args.input + 'data.h5', train=True)
    num_features = len(features)

    # Regulsarisation parameter
    lambda_reg = cfg['combined']['model']['lambda_reg']  # Use same `lambda` as the adversary
    digits = int(np.ceil(max(-np.log10(lambda_reg), 0)))
    lambda_str = '{l:.{d:d}f}'.format(d=digits,l=lambda_reg).replace('.', 'p')

    # Get standard-formatted decorrelation inputs
    decorrelation = get_decorrelation_variables(data)
    aux_vars = ['logpt']
    data['logpt'] = pd.Series(np.log(data['pt'].values), index=data.index)
    
    # Specify common weights
    # -- Classifier
    weight_var = 'weight_adv'  # 'weight_adv' / 'weight_train'
    data['weight_clf'] = pd.Series(data[weight_var].values, index=data.index)

    # -- Adversary
    data['weight_adv'] = pd.Series(np.multiply(data['weight_adv'].values,  1. - data['signal'].values), index=data.index)


    # Classifier-only fit, cross-validation
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, cross-validation"):

        # Define variable(s)
        basename = 'crossval_classifier'
        basedir  = 'models/adversarial/classifier/crossval/'

        # Get indices for each fold in stratified k-fold training
        # @NOTE: No shuffling is performed -- assuming that's already done.
        skf = StratifiedKFold(n_splits=args.folds).split(data[features].values,
                                                         data['signal'].values)

        # Import module creator methods and optimiser options
        from adversarial.models import classifier_model, adversary_model, combined_model, decorrelation_model

        # Collection of classifiers and their associated training histories
        classifiers = list()
        histories   = list()

        # Train or load classifiers
        if args.optimise_classifier:  # args.train or args.train_classifier:
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
                    parallelised = parallelise_model(classifier, args)

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
                    if args.tensorboard:
                        callbacks += [TensorBoard(log_dir=tensorboard_dir + 'classifier/fold{}/'.format(fold))]
                        pass

                    # Compute initial losses
                    X_val, Y_val, W_val = validation_data
                    eval_opts = dict(batch_size=cfg['classifier']['fit']['batch_size'], verbose=0)
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
                    save([args.output, basedir], name, classifier, ret.history)
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

        # Compute average validation loss
        val_avg = np.mean([hist['val_loss'] for hist in histories], axis=0)
        val_std = np.std ([hist['val_loss'] for hist in histories], axis=0)
        return val_avg[-1] + val_std[-1]


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
            parallelised = parallelise_model(classifier, args)

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['classifier']['compile'])

            # Create callbacks
            callbacks = []

            # -- TensorBoard
            if args.tensorboard:
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
            save([args.output, basedir], name, classifier, ret.history)

            # Saving classifier in lwtnn-friendly format.
            lwtnn_save(classifier, 'nn')

        else:

            # Load pre-trained classifier
            log.info("Loading full classifier from file")
            classifier, history = load(basedir, name)
            pass # end: train/load
        pass



    # Definitions for adversarial training
    # --------------------------------------------------------------------------
    # Create custom Kullback-Leibler (KL) divergence cost.
    def kullback_leibler (p_true, p_pred):
        return -K.log(p_pred)

    cfg['combined']['compile']['loss'][1] = kullback_leibler


    # @TODO: Make `train_{classifier,adverarial}` methods for used with _both_
    #        cross-val.- and full trianing

    # Combined adversarial fit, cross-validation
    # --------------------------------------------------------------------------
    with Profile("Combined adversarial fit, cross-validation"):
        # @TODO:
        # - Checkpointing

        # Define variables
        results  = []  # Holding optimisation metrics
        basename = 'combined_lambda{}'.format(lambda_str)
        basedir  = 'models/adversarial/combined/crossval/'

        # Get indices for each fold in stratified k-fold training
        # @NOTE: No shuffling is performed -- assuming that's already done above.
        skf = StratifiedKFold(n_splits=args.folds).split(data[features].values,
                                                         data['signal'].values)

        if args.optimise_adversarial:  # args.train or args.train_adversarial:
            log.info("Training combined model cross-validation")

            # Loop `k` folds
            for fold, (train, validation) in enumerate(skf):
                with Profile("Fold {}/{}".format(fold + 1, args.folds)):

                    # Define unique name for current classifier
                    name = '{}__{}of{}'.format(basename, fold + 1, args.folds)

                    # Load pre-trained classifier
                    classifier, _ = load('models/adversarial/classifier/full/', 'classifier')

                    # Set up adversary
                    adversary = adversary_model(gmm_dimensions=len(DECORRELATION_VARIABLES),
                                                **cfg['adversary']['model'])
                    
                    # Set up combined, adversarial model
                    combined = combined_model(classifier, adversary, **cfg['combined']['model'])

                    # Parallelise on GPUs
                    parallelised = parallelise_model(combined, args)

                    # Prepare arrays
                    X = [data[features]    .values[train]] + [data[aux_vars].values[train], decorrelation[train]]
                    Y = [data['signal']    .values[train]] + [np.ones_like(data['signal'].values[train])]
                    W = [data['weight_clf'].values[train]] + [data['weight_adv'].values[train]]

                    validation_data = (
                        [data[features]    .values[validation]] + [data[aux_vars].values[validation], decorrelation[validation]],
                        [data['signal']    .values[validation]] + [np.ones_like(data['signal'].values[validation])],
                        [data['weight_clf'].values[validation]] + [data['weight_adv'].values[validation]]
                        )
                        
                    # Compile model for pre-training
                    classifier.trainable = False
                    parallelised.compile(**cfg['combined']['compile'])

                    # Compute initial losses
                    log.info("Computing initial loss")
                    X_val, Y_val, W_val = validation_data
                    eval_opts = dict(batch_size=cfg['combined']['fit']['batch_size'], verbose=0)
                    initial_losses = [parallelised.evaluate(X,     Y,     sample_weight=W,     **eval_opts),
                                      parallelised.evaluate(X_val, Y_val, sample_weight=W_val, **eval_opts)]

                    # Pre-training adversary
                    log.info("Pre-training")
                    pretrain_fit_opts = dict(**cfg['combined']['fit'])
                    pretrain_fit_opts['epochs'] = cfg['combined']['pretrain']
                    ret_pretrain = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, **pretrain_fit_opts)

                    # Re-compile combined model for full training
                    classifier.trainable = True
                    parallelised.compile(**cfg['combined']['compile'])

                    # Fit classifier model
                    log.info("Actual training")
                    ret = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, **cfg['combined']['fit'])

                    # Prepend initial losses
                    for metric, loss_train, loss_val in zip(parallelised.metrics_names, *initial_losses):
                        ret_pretrain.history[metric]         .insert(0, loss_train)
                        ret_pretrain.history['val_' + metric].insert(0, loss_val)
                        pass

                    for metric in parallelised.metrics_names:
                        ret.history[metric]          = ret_pretrain.history[metric]          + ret.history[metric]
                        ret.history['val_' + metric] = ret_pretrain.history['val_' + metric] + ret.history['val_' + metric]
                        pass

                    # Save combined model and training history to file, both in unique
                    # output directory and in the directory for pre-trained classifiers.
                    save([args.output, basedir], name, combined, ret.history)

                    # Add `ANN` variable
                    add_nn(data, classifier, 'ANN')

                    # Compute optimisation metric
                    try:
                        rej, jsd = metrics(data.iloc[validation], 'ANN')
                        print "Background rejection: {}".format(rej)
                        print "1/JSD:                {}".format(jsd)
                        if np.inf in [rej,jsd] or np.nan in [rej,jsd]:
                            return 0
                        results.append(rej + lambda_reg * jsd)
                    except ValueError:
                        print "Got a NaN. Returning 0"
                        return 0
                    pass
                pass
            pass
        pass


    # Early stopping in case of adversarial network
    # --------------------------------------------------------------------------
    if args.optimise_adversarial:

        # Return optimisation metric: - (rej + 1/jsd)
        print "rej + 1/jsd: {} ± {}".format(np.mean(results), np.std(results))
        return - (np.mean(results) - np.std(results))


    # Combined adversarial fit, full
    # --------------------------------------------------------------------------
    with Profile("Combined adversarial fit, full"):

        # Define variables
        name    = 'combined_lambda{}'.format(lambda_str)
        basedir = 'models/adversarial/combined/full/'

        # Load pre-trained classifier
        classifier, _ = load('models/adversarial/classifier/full/', 'classifier')

        # Set up adversary
        adversary = adversary_model(gmm_dimensions=len(DECORRELATION_VARIABLES),
                                    **cfg['adversary']['model'])
        
        # Save adversarial model diagram
        plot_model(adversary, to_file=args.output + 'model_adversary.png', show_shapes=True)

        # Create callback array
        callbacks = list()

        # (opt.) Add TensorBoard callback
        if args.tensorboard:
            callbacks += [TensorBoard(log_dir=tensorboard_dir + 'adversarial/')]
            pass

        # Set up combined, adversarial model
        combined = combined_model(classifier, adversary, **cfg['combined']['model'])

        # Save combined model diagram
        plot_model(combined, to_file=args.output + 'model_{}.png'.format(name), show_shapes=True)

        if args.train or args.train_adversarial:
            log.info("Training full, combined model")

            # Parallelise on GPUs
            parallelised = parallelise_model(combined, args)

            # Compile model (necessary to save properly)
            parallelised.compile(**cfg['combined']['compile'])

            # Prepare arrays
            X = [data[features]    .values] + [data[aux_vars].values, decorrelation]
            Y = [data['signal']    .values] + [np.ones_like(data['signal'].values)]
            W = [data['weight_clf'].values] + [data['weight_adv'].values]

            # Compile model for pre-training
            classifier.trainable = False
            parallelised.compile(**cfg['combined']['compile'])

            # Pre-training adversary
            log.info("Pre-training")
            pretrain_fit_opts = dict(**cfg['combined']['fit'])
            pretrain_fit_opts['epochs'] = cfg['combined']['pretrain']
            ret_pretrain = parallelised.fit(X, Y, sample_weight=W, **pretrain_fit_opts)

            # Re-compile combined model for full training
            classifier.trainable = True
            parallelised.compile(**cfg['combined']['compile'])

            # Fit classifier model
            log.info("Actual training")
            ret = parallelised.fit(X, Y, sample_weight=W, callbacks=callbacks, **cfg['combined']['fit'])

            # Prepend initial losses
            for metric in parallelised.metrics_names:
                ret.history[metric]          = ret_pretrain.history[metric]          + ret.history[metric]
                pass

            # Save combined model and training history to file, both in unique
            # output directory and in the directory for pre-trained classifiers.
            adv = lambda s: s.replace('combined', 'adversary')
            save([args.output,     basedir],      name,  combined, ret.history)
            save([args.output, adv(basedir)], adv(name), adversary)

            # Saving adversarially trained classifier in lwtnn-friendly format.
            lwtnn_save(classifier, 'ann')

        else:

            # Load pre-trained combined _weights_ from file, in order to
            # simultaneously load the embedded classifier so as to not have to
            # extract it manually afterwards.
            log.info("Loading full, combined model from file")
            combined, history = load(basedir, name, model=combined)
            pass # end: train/load

        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(adversarial=True)

    # Call main function
    main(args)
    pass
