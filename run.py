#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing training (and evaluation?) of adversarial neural networks for de-correlated jet tagging."""

# Basic import(s)
import os
import sys
import gzip
import glob
import json
import pickle
from pprint import pprint
import logging as log
import itertools

# Scientific import(s)
import numpy as np
from numpy.lib.recfunctions import append_fields
np.random.seed(21) # For reproducibility

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
plt.switch_backend('pdf')
plt.style.use('edinburgh')

# -- Explicitly ignore DeprecationWarning from scikit-learn, which we can't do 
#    anything about anyway.
stderr = sys.stderr
with open(os.devnull, 'w') as sys.stderr:
    from hep_ml.reweight import GBReweighter, BinsReweighter
    pass
sys.stderr = stderr

# Custom import(s)
from rootplotting import ap
from rootplotting.tools import loadData, loadXsec, scale_weights

# Project import(s)
from adversarial.utils    import *
from adversarial.profiler import *

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Perform training (and evaluation?) of adversarial neural networks for de-correlated jet tagging.")

# -- Inputs
parser.add_argument('-i', '--input',  dest='input',   action='store', type=str,
                    default='./', help='Input directory, from which to read ROOT files.')
parser.add_argument('-o', '--output', dest='output',  action='store', type=str,
                    default='./', help='Output directory, to which to write results.')
parser.add_argument('-c', '--config', dest='config',  action='store', type=str,
                    default='./configs/default.json', help='Configuration file.')
parser.add_argument('-p', '--patch', dest='patches', action='append', type=str,
                    help='Patch file(s) with which to update configuration file.')
parser.add_argument('--threads',     dest='threads', action='store', type=int,
                    default=1, help='Number of (CPU) threads to use with Theano.')
parser.add_argument('--folds',       dest='folds',    action='store', type=int,
                    default=2, help='Number of folds to use for stratified cross-validation.')

# -- Flags
parser.add_argument('-v', '--verbose', dest='verbose', action='store_const', 
                    const=True, default=False, help='Print verbose')
parser.add_argument('-g', '--gpu',  dest='gpu',        action='store_const',
                    const=True, default=False, help='Run on GPU')
parser.add_argument('--tensorflow', dest='tensorflow', action='store_const',
                    const=True, default=False, help='Use Tensorflow backend')
parser.add_argument('--train', dest='train', action='store_const',
                    const=True, default=False, help='Perform training')


# Main function definition
@profile
def main ():

    # Initialisation
    # --------------------------------------------------------------------------
    with Profiler("Initialisation"):

        # Parse command-line arguments
        with Profiler():
            args = parser.parse_args()
            pass

        # Add 'mode' field manually
        args = argparse.Namespace(mode='gpu' if args.gpu else 'cpu', **vars(args))

        # Set print level
        log.basicConfig(format="%(levelname)s: %(message)s", 
                        level=log.DEBUG if args.verbose else log.INFO)

        #  Modify input/output directory names to conform to convention
        if not args.input .endswith('/'): args.input  += '/'
        if not args.output.endswith('/'): args.output += '/'
        
        # Load configuration file
        with open(args.config, 'r') as f:
            cfg = json.load(f)
            pass

        # Apply patches
        if args.patches is not None:
            for patch_file in args.patches:
                log.info("Applying patch '{}'".format(patch_file))
                with open(patch_file, 'r') as f:
                    patch = json.load(f)
                    pass
                apply_patch(cfg, patch)
                pass
            pass

        # Initialise Keras backend
        initialise_backend(args)

        import keras
        import keras.backend as K
        from keras.models import Model
        from keras.models import load_model
        from keras.layers import Input
        from keras.layers.core import Lambda
        from keras.layers.merge import Concatenate
        from keras.callbacks import Callback
        from keras.utils.vis_utils import plot_model

        if args.tensorflow:
            import tensorflow as tf
            pass
        
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
        except NameError: log.info("Keras not imported")

        # Save command-line argument configuration in output directory
        with open(args.output + 'args.json', 'wb') as f:
            json.dump(vars(args), f, indent=4, sort_keys=True)
            pass

        # Save configuration dict in output directory
        with open(args.output + 'config.json', 'wb') as f:
            json.dump(cfg, f, indent=4, sort_keys=True)
            pass

        pass


    # @TODO: Turn 'Loading data', 'Re-weighting', 'Data preparation' into
    # utility function(s), for use with separate `evaluate.py` script
    
    # Loading data
    # --------------------------------------------------------------------------
    with Profiler("Loading data"):
        
        # Get paths for files to use
        with Profiler():
            log.debug("ROOT files in input directory:")
            all_paths = sorted(glob.glob(args.input + '*.root'))
            sig_paths = sorted(glob.glob(args.input + 'objdef_MC_30836*.root'))
            bkg_paths = sorted(glob.glob(args.input + 'objdef_MC_3610*.root'))
            for p in all_paths:
                cls = 'signal' if p in sig_paths else ('background' if p in bkg_paths else None)
                log.debug("  " + p + (" (%s)" % cls if cls else "" ))
                pass

            sig_paths = [sig_paths[0]]
            log.info("Using signal sample: '{}'".format(sig_paths[0]))
            pass
        
        # Get data- and info arrays
        datatreename = 'BoostedJet+ISRgamma/Nominal/EventSelection/Pass/NumLargeRadiusJets/Postcut'
        infotreename = 'BoostedJet+ISRgamma/Nominal/outputTree'
        prefix = 'Jet_'
        # @TODO: Is it easier/better to drop the prefix, list the names
        # explicitly, and then rename manually afterwards? Or should the
        # 'loadData' method support a 'rename' argument, using regex, with
        # support for multiple such operations? That last one is probably the
        # way to go, actually... In that case, I should probably also allow for
        # regex support for branches? Like branches=['Jet_.*',
        # 'leading_Photons_.*'], rename=[('Jet_', ''),]

        #branches = ['m', 'pt', 'C2', 'D2', 'ECF1', 'ECF2', 'ECF3', 'Split12',
        #'Split23', 'Split34', 'eta', 'leading_Photons_E',
        #'leading_Photons_eta', 'leading_Photons_phi', 'leading_Photons_pt',
        #'nTracks', 'phi', 'rho', 'tau21']

        log.info("Reading data from '%s' with prefix '%s'" % (datatreename, prefix))
        log.info("Reading info from '%s'" % (infotreename))
        with Profiler():
            sig_data = loadData(sig_paths, datatreename, prefix=prefix) # ..., branches=branches)
            sig_info = loadData(sig_paths, infotreename, stop=1)
            pass

        with Profiler():
            bkg_data = loadData(bkg_paths, datatreename, prefix=prefix) # ..., branches=branches)
            bkg_info = loadData(bkg_paths, infotreename, stop=1)
            pass
        
        log.info("Retrieved data columns: [%s]" % (', '.join(sig_data.dtype.names)))
        log.info("Retrieved %d signal and %d background events." % (sig_data.shape[0], bkg_data.shape[0]))
        
        # Scale by cross section
        with Profiler():
            log.debug("Scaling weights by cross-section and luminosity")
            xsec = loadXsec('share/sampleInfo.csv')
        
            sig = scale_weights(sig_data, sig_info, xsec=xsec, lumi=36.1)
            bkg = scale_weights(bkg_data, bkg_info, xsec=xsec, lumi=36.1)
            pass
        
        # Restricting phase space
        with Profiler():
            # - min(pT) of 200 GeV imposed in AnalysisTool code
            # - min(m)  of   0 GeV required by physics and log(·)
            # - otherwise, we're free to choose whatever phasespace we want

            # @TODO: Tune phase space selection and/or reweighter settings, such 
            # that there is not a drop-off at high mass/low pt of the re-weighted 
            # background spectrum
            log.debug("Restricting phase space")
            msk  = (sig['m']  >  10.) & (sig['m']  <  300.)
            msk &= (sig['pt'] > 200.) & (sig['pt'] < 2000.)
            sig  = sig[msk]
            
            msk  = (bkg['m']  >  10.) & (bkg['m']  <  300.)
            msk &= (bkg['pt'] > 200.) & (bkg['pt'] < 2000.)
            bkg  = bkg[msk]
            pass

        pass


    # Re-weighting to flatness
    # --------------------------------------------------------------------------
    with Profiler("Re-weighting"):
        # @NOTE: This is the crucial point: If the target is flat in (m,pt) the
        # re-weighted background _won't_ be flat in (log m, log pt), and vice 
        # versa. It should go without saying, but draw target samples from a 
        # uniform prior on the coordinates which are used for the decorrelation.

        # Performing pre-processing of de-correlation coordinates
        with Profiler():
            log.debug("Performing pre-processing")

            # Get number of background events and number of target events (arb.)
            N_sig = len(sig)
            N_bkg = len(bkg)
            N_tar = len(bkg)
            
            # Initialise and fill coordinate arrays
            P_sig = np.zeros((N_sig,2), dtype=float)
            P_bkg = np.zeros((N_bkg,2), dtype=float)
            P_sig[:,0] = np.log(sig['m'])
            P_sig[:,1] = np.log(sig['pt'])
            P_bkg[:,0] = np.log(bkg['m'])
            P_bkg[:,1] = np.log(bkg['pt'])
            P_tar = np.random.rand(N_tar, 2)
            
            # Scale coordinates to range [0,1]
            log.debug("Scaling background coordinates to range [0,1]")
            P_sig -= np.min(P_sig, axis=0)
            P_bkg -= np.min(P_bkg, axis=0)
            P_sig /= np.max(P_sig, axis=0)
            P_bkg /= np.max(P_bkg, axis=0)
            log.debug("  Min (sig):", np.min(P_sig, axis=0))
            log.debug("  Max (sig):", np.max(P_sig, axis=0))
            log.debug("  Min (bkg):", np.min(P_bkg, axis=0))
            log.debug("  Max (bkg):", np.max(P_bkg, axis=0))
            pass

        # Fit, or load, regressor to achieve flatness using hep_ml library
        with Profiler():
            log.debug("Performing re-weighting using GBReweighter")
            reweighter_filename = 'trained/reweighter.pkl.gz'
            if False: # @TODO: Make flag
                reweighter = GBReweighter(n_estimators=80, max_depth=7)
                reweighter.fit(P_bkg, target=P_tar, original_weight=bkg['weight'])
                log.debug("Saving re-weighting object to file '%s'" % reweighter_filename)
                with gzip.open(reweighter_filename, 'wb') as f:
                    pickle.dump(reweighter, f)
                    pass
            else:
                log.debug("Loading re-weighting object from file '%s'" % reweighter_filename)
                with gzip.open(reweighter_filename, 'r') as f:
                    reweighter = pickle.load(f)
                    pass
                pass
            pass

        # Re-weight for uniform prior(s)
        with Profiler():
            log.debug("Getting new weights for uniform prior(s)")
            new_weights  = reweighter.predict_weights(P_bkg, original_weight=bkg['weight'])
            new_weights *= np.sum(bkg['weight']) / np.sum(new_weights)
            bkg = append_fields(bkg, 'reweight', new_weights, dtypes=K.floatx())

            # Appending similary ("dummy") 'reweight' field to signal sample, for consistency
            sig = append_fields(sig, 'reweight', sig['weight'], dtypes=K.floatx())
            pass

        pass


    # Plotting: Re-weighting
    # --------------------------------------------------------------------------
    with Profiler("Plotting: Re-weighting"):
        

        fig, ax = plt.subplots(2, 4, figsize=(12,6))

        w_bkg  = bkg['weight']
        rw_bkg = bkg['reweight']
        w_tar  = np.ones((N_tar,)) * np.sum(bkg['weight']) / float(N_tar)

        for row, var in enumerate(['m', 'pt']):
            edges = np.linspace(0, np.max(bkg[var]), 60 + 1, endpoint=True)
            nbins  = len(edges) - 1

            v_bkg  = bkg[var]     # Background  mass/pt values for the background
            rv_bkg = P_bkg[:,row] # Transformed mass/pt values for the background
            rv_tar = P_tar[:,row] # Transformed mass/pt values for the targer

            ax[row,0].hist(v_bkg,  bins=edges, weights=w_bkg,  alpha=0.5, label='Background')
            ax[row,1].hist(v_bkg,  bins=edges, weights=rw_bkg, alpha=0.5, label='Background')
            ax[row,2].hist(rv_bkg, bins=nbins, weights=w_bkg,  alpha=0.5, label='Background') # =rw_bkg
            ax[row,2].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')
            ax[row,3].hist(rv_bkg, bins=nbins, weights=rw_bkg, alpha=0.5, label='Background')
            ax[row,3].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')

            for col in range(4):
                if col < 4: # 3
                    ax[row,col].set_yscale('log')
                    ax[row,col].set_ylim(1E+01, 1E+06)
                    if row == 1:
                        ax[row,col].set_ylim(1E-01, 1E+05)
                        pass
                    pass
                ax[row,col].set_xlabel("Jet %s%s%s" % (var, " (transformed)" if col > 1 else '', " (re-weighted)" if (col + 1) % 2 == 0 else ''))
                if col == 0:
                    ax[row,col].set_ylabel("Jets / {:.1f} GeV".format(np.diff(edges)[0]))
                    pass
                pass
            pass

        plt.legend()
        plt.savefig(args.output + 'priors_1d.pdf')

        # Plot 2D prior before and after re-weighting
        log.debug("Plotting 2D prior before and after re-weighting")
        fig, ax = plt.subplots(1,2,figsize=(11,5), sharex=True, sharey=True)
        h = ax[0].hist2d(P_bkg[:,0], P_bkg[:,1], bins=40, weights=bkg['weight'],   vmin=0, vmax=5, normed=True)
        h = ax[1].hist2d(P_bkg[:,0], P_bkg[:,1], bins=40, weights=bkg['reweight'], vmin=0, vmax=5, normed=True)
        ax[0].set_xlabel("Scaled log(m)")
        ax[1].set_xlabel("Scaled log(m)")
        ax[0].set_ylabel("Scaled log(pt)")
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
        fig.colorbar(h[3], cax=cbar_ax)
        plt.savefig(args.output + 'priors_2d.pdf')
        pass


    # Prepare arrays for training
    # --------------------------------------------------------------------------
    with Profiler("Data preparation"):

        # Remove unwanted fields from input array
        names = sig.dtype.names
        exclude = ['pt', 'm', 'isMC', 'DSID', 'weight', 'reweight', 'CutVariable', \
                   'id', 'q', 'x1', 'x2', 'tau21DDT', 'rhoDDT', 'tau21_ungroomed', \
                   'pt_ungroomed', 'pdgId1', 'pdgId2']
        names = sorted(list(set(names) - set(exclude)))
        
        log.info("Using The following variables as inputs to the neural network:\n[%s]" % ', '.join(names))
        
        # Weights
        W_sig = sig['weight']   / np.sum(sig['weight'])   * float(bkg['weight'].size) 
        W_bkg = bkg['weight']   / np.sum(bkg['weight'])   * float(bkg['weight'].size) 
        U_sig = sig['reweight'] / np.sum(sig['reweight']) * float(bkg['reweight'].size) 
        U_bkg = bkg['reweight'] / np.sum(bkg['reweight']) * float(bkg['reweight'].size) 
        W = np.hstack((W_sig, W_bkg)).astype(K.floatx())
        U = np.hstack((U_sig, U_bkg)).astype(K.floatx())
        
        # Labels
        #Y = np.hstack((np.ones(N_sig, dtype=int), np.zeros(N_bkg, dtype=int)))
        Y = np.hstack((np.ones(N_sig), np.zeros(N_bkg))).astype(K.floatx())
        
        # Input(s)
        X_sig = np.vstack(tuple(sig[var] for var in names)).T
        X_bkg = np.vstack(tuple(bkg[var] for var in names)).T
        
        # Data pre-processing
        # @NOTE: This is already done manually for P_{sig,bkg} above.
        substructure_scaler = StandardScaler().fit(X_bkg)
        X_sig = substructure_scaler.transform(X_sig)
        X_bkg = substructure_scaler.transform(X_bkg)
        
        # Concatenate signal and background samples
        X = np.vstack((X_sig, X_bkg)).astype(K.floatx())
        P = np.vstack((P_sig, P_bkg)).astype(K.floatx())

        # Convenient short-hands
        num_samples, num_features = X.shape

        # Manually shuffle input, once and for all
        #shuffle_indices = np.arange(num_samples, dtype=int)
        #np.random.shuffle(shuffle_indices)

        #X = X[shuffle_indices,:]
        #P = P[shuffle_indices,:]
        #Y = Y[shuffle_indices]
        #W = W[shuffle_indices]
        #U = U[shuffle_indices]
        pass


    # Classifier-only fit
    # --------------------------------------------------------------------------
    # Adapted from: https://github.com/asogaard/AdversarialSubstructure/blob/master/train.py
    with Profiler("Classifier-only fit, cross-validation"):
        # @TODO: - Implement checkpointing
        #        - Tons of stuf...
        
        # Fit non-adversarial neural network
        #if not retrain_classifier:
        #print "\n== Loading classifier model."
        #
        ## Load existing classifier model from file
        #classifier = load_model('classifier.h5')
        #
        #else:

        # Get indices for each fold in stratified k-fold training
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=True)

        # Get minimal number of samples for each fold, to ensure proper
        # batching. This will on average lead to a loss of around `args.folds /
        # 2` samples per classifier, which is something we can live with.
        #train_samples = min(len(train) for train, _ in skf.split(X,Y))
        #test_samples  = min(len(test)  for _, test  in skf.split(X,Y))

        # Importe module creator methods and optimiser options
        from adversarial.models import compiler_options, classifier_model, adversarial_model

        # @TODO: Implement _data-parallel_ training. In this way, the same model
        # is trained on different slices of data on different nodes. That way we
        # can speed up running for all training cases, not just for statified
        # k-fold training.
        # [https://www.tensorflow.org/deploy/distributed#create_a_cluster]
        #
        # @NOTE: Need to make sure that the parameres (W,b) for the classifier
        # live on the same parameter server (PS)/CPU, and that only the
        # computations are distributed across GPU.
        #
        # Resources:
        #  [https://github.com/fchollet/keras/issues/7515]
        #  [https://stackoverflow.com/questions/43821786/data-parallelism-in-keras]
        #  [https://stackoverflow.com/a/44771313]

        # Create unique set of random indices to use with stratification
        random_indices = np.arange(num_samples)
        np.random.shuffle(random_indices)

        # Collection of classifiers and their associated training histories
        classifiers = list()
        histories   = list()

        # Train or load classifiers
        if args.train:
            log.info("Training classifiers")
            
            # Loop `k` folds
            for fold, (train, validation) in enumerate(skf.split(X,Y)):
                with Profiler("Fold {}/{}".format(fold + 1, args.folds)):

                    # StratifiedKFold provides stratification, but since the
                    # input arrays are not randomised, neither will the
                    # folds. Therefore, the fold should be taken with respect to
                    # a set of randomised indices rather than range(N).
                    train      = random_indices[train]
                    validation = random_indices[validation]

                    # Define unique tag and name for current classifier
                    tag  = '{}of{}'.format(fold + 1, args.folds)
                    name = 'crossval_classifier__{}'.format(tag)

                    # Get classifier
                    classifier = classifier_model(num_features, **cfg['classifier']['model'])
                    
                    # Compile with optimiser configuration
                    opts = dict(**cfg['classifier']['compile'])
                    opts['optimizer'] = eval("keras.optimizers.{optimizer}(lr={lr}, decay={decay})" \
                                             .format(optimizer = opts['optimizer'],
                                                     lr        = opts.pop('lr'),
                                                     decay     = opts.pop('decay')))
                    classifier.compile(**opts)
                    
                    # Save classifier model diagram to file
                    if fold == 0:
                        plot_model(classifier, to_file=args.output + 'classifier.png', show_shapes=True)
                        pass                
                    
                    # Fit classifier model
                    hist = classifier.fit(X[train,:], Y[train], sample_weight=W[train],
                                          validation_data=(X[validation,:], Y[validation], W[validation]),
                                          **cfg['classifier']['fit'])
                    histories.append(hist.history)
                    
                    # Save classifier model to file
                    classifier.save('trained/{}.h5'.format(name))

                    # Save training history to file, both in unique output
                    # directory and in the directory for pre-trained classifiers
                    for destination in [args.output, 'trained/']:
                        with open(destination + 'history__{}.json'.format(name), 'wb') as f:
                            json.dump(hist.history, f)
                            pass
                        pass
                    
                    # Add to list of classifiers
                    classifiers.append(classifier)
                    pass
                pass
        else:
            log.info("Loading classifiers from file")
            
            # Load pre-trained classifiers
            classifier_files = sorted(glob.glob('trained/crossval_classifier__*of{}.h5'.format(args.folds)))
            assert len(classifier_files) == args.folds, "Number of pre-trained classifiers ({}) does not match number of requested folds ({})".format(len(classifier_files), args.folds)
            for classifier_file in classifier_files:
                classifiers.append(load_model(classifier_file))
                pass

            # Load associated training histories
            history_files = sorted(glob.glob('trained/history__crossval_classifier__*of{}.json'.format(args.folds)))
            assert len(history_files) == args.folds, "Number of training histories for pre-trained classifiers ({}) does not match number of requested folds ({})".format(len(history_files), args.folds)
            for history_file in history_files:
                with open(history_file, 'r') as f:
                    histories.append(json.load(f))
                    pass
                pass

            pass # end: train/load


    # Classifier-only fit
    # --------------------------------------------------------------------------
    # Optimal number of trainin epochs
    opt_epochs = None
    
    with Profiler("Plotting: Classifier-only fit, cross-val."):        
        # Log history
        fig, ax = plt.subplots()
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
        
        # @NOTE Assuming no early stopping
        epochs = 1 + np.arange(len(histories[0]['loss']))
        
        for fold, hist in enumerate(histories): 
            plt.plot(epochs, hist['val_loss'], color=colours[1], linewidth=0.6, alpha=0.3,
                     label='Validation (fold)' if fold == 0 else None)
            pass
        
        val_avg = np.mean([hist['val_loss'] for hist in histories], axis=0)
        plt.plot(epochs, val_avg,   color=colours[1], label='Validation (avg.)')

        # Store the optimal number of training epochs        
        opt_epochs = epochs[np.argmin(val_avg)]
        log.info("Using optimal number of {:d} training epochs".format(opt_epochs))
        
        for fold, hist in enumerate(histories):
            plt.plot(epochs, hist['loss'],     color=colours[0], linewidth=1.0, alpha=0.3,
                     label='Training (fold)'   if fold == 0 else None)
            pass
        
        train_avg = np.mean([hist['loss'] for hist in histories], axis=0)
        plt.plot(epochs, train_avg, color=colours[0], label='Train (avg.)')
        
        plt.title('Classifier-only, stratified {}-fold training'.format(args.folds), fontweight='medium')
        plt.xlabel("Epochs",        horizontalalignment='right', x=1.0)
        plt.ylabel("Logistic loss", horizontalalignment='right', y=1.0)
        
        epochs = [0] + list(epochs)
        step = max(int(np.floor(len(epochs) / 10.)), 1)
        
        plt.xticks(filter(lambda x: x % step == 0, epochs))
        plt.legend()
        plt.savefig(args.output + 'costlog.pdf')
        pass


    # Classifier-only fit, full
    # --------------------------------------------------------------------------
    with Profiler("Classifier-only fit, full"):

        # @TODO Train _final_ classifier on all data
        # Get classifier
        classifier = classifier_model(num_features, **cfg['classifier']['model'])
        
        # Compile with optimiser configuration
        opts = dict(**cfg['classifier']['compile'])
        opts['optimizer'] = eval("keras.optimizers.{optimizer}(lr={lr}, decay={decay})" \
                                 .format(optimizer = opts['optimizer'],
                                         lr        = opts.pop('lr'),
                                         decay     = opts.pop('decay')))
        classifier.compile(**opts)

        # Overwrite number of training epochs with optimal number found from
        # cross-validation
        cfg['classifier']['fit']['epochs'] = opt_epochs

        # Train final classifier
        #classifier.fit(X, Y, sample_weight=W, **cfg['classifier']['fit'])
        train_in_series(classifier, {'input': X, 'target': Y, 'weights': W},
                        config=cfg['classifier'])

        train_in_parallel(classifier, {'input': X, 'target': Y, 'weights': W},
                          config=cfg['classifier'], num_gpus=args.threads, seed=21)

        return
        
        # ...
        
        # Store classifier output as tagger variables. @NOTE This works only
        # _provided_ the input array X has the same ordering as sig/bkg.
        msk_sig = (Y == 1.)
        sig = append_fields(sig, 'NN', classifier.predict(X[ msk_sig], batch_size=1024).flatten(), dtypes=K.floatx())
        bkg = append_fields(bkg, 'NN', classifier.predict(X[~msk_sig], batch_size=1024).flatten(), dtypes=K.floatx())
        pass

    
    # Plotting: Distributions/ROC
    # --------------------------------------------------------------------------
    with Profiler("Plotting: Distributions/ROC"):

        # Tagger variables
        variables = ['tau21', 'tau21DDT', 'NN']

        # Plotted 1D tagger variable distributions
        fig, ax = plt.subplots(1, len(variables), figsize=(len(variables) * 4, 4))

        w_sig  = sig['weight']
        w_bkg  = bkg['weight']

        for ivar, var in enumerate(variables):
            nbins  = 50

            v_sig  = sig[var]
            v_bkg  = bkg[var]

            _, edges, _ = \
            ax[ivar].hist(v_bkg, bins=nbins, weights=w_bkg, alpha=0.5, normed=True, label='Background')
            ax[ivar].hist(v_sig, bins=edges, weights=w_sig, alpha=0.5, normed=True, label='Signal')

            ax[ivar].set_xlabel("Jet {}".format(var),                      horizontalalignment='right', x=1.0)
            ax[ivar].set_ylabel("Jets / {:.3f}".format(np.diff(edges)[0]), horizontalalignment='right',     y=1.0)
            pass

        plt.legend()
        plt.savefig(args.output + 'tagger_distributions.pdf')

        # Plotted ROCs
        fig, ax = plt.subplots(figsize=(5,5))

        ax.plot([0,1],[0,1], 'k--', linewidth=1.0, alpha=0.2)
        for ivar, var in enumerate(reversed(variables)):
            eff_sig, eff_bkg = roc_efficiencies(sig[var], bkg[var], sig['weight'], bkg['weight'])
            try:
                auc = roc_auc(eff_sig, eff_bkg)
            except: # Efficiencies not monotonically increasing
                auc = 0.
                pass
            ax.plot(eff_bkg, eff_sig, label='{} (AUC: {:.3f})'.format(var, auc))
            pass

        plt.xlabel("Background efficiency", horizontalalignment='right', x=1.0)
        plt.ylabel("Signal efficiency",     horizontalalignment='right', y=1.0)
        plt.legend()
        plt.savefig(args.output + 'tagger_ROCs.pdf')
        pass
    
    return

    # ==========================================================================
    '''
    
    # Set up combined, adversarial model
    adversarial = adversarial_model(classifier, architecture=[(64, 'tanh')] * 2, num_posterior_components=1, num_posterior_dimensions=P_train.shape[1])

    if resume: 
        load_checkpoint(adversarial)
        pass

    adversarial.compile(**opts['adversarial'])

    # Save adversarial model diagram
    plot(adversarial, to_file='adversarial.png', show_shapes=True)

    # Set fit options
    fit_opts = {
        'shuffle':          True,
        'validation_split': 0.2,
        'batch_size':       4 * 1024,
        'nb_epoch':         100,
        'sample_weight':    [W_train, np.multiply(W_train, 1. - Y_train)]
    }


    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.lossnames = ['loss', 'classifier_loss', 'adversary_loss']
            self.losses = {name: list() for name in self.lossnames}
            return

        def on_batch_end(self, batch, logs={}):
            for name in self.lossnames:
                self.losses[name].append(float(logs.get(name)))
                pass
            return
        pass

    history = LossHistory()

    # -- Callback for updating learning rate(s)
    damp = np.power(1.0E-04, 1./float(fit_opts['nb_epoch']))
    def schedule (epoch):
        "" " Update the learning rate of the two optimisers. "" "
        if 0 < damp and damp < 1:
            K_.set_value(adv_optim.lr, damp * K_.get_value(adv_optim.lr))
            pass
        return float(K_.eval(adv_optim.lr))

    change_lr = LearningRateScheduler(schedule)

    # -- Callback for saving model checkpoints
    from keras.callbacks import ModelCheckpoint
    checkpointer = ModelCheckpoint(filepath=".adversarial_checkpoint.h5", verbose=0, save_best_only=False)

    # -- Callback to reduce learning rate when validation loss plateaus
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1E-07)

    # Store callbacks in fit options
    fit_opts['callbacks'] = [history, change_lr, checkpointer]

    # Fit the combined, adversarial model
    adversarial.fit([X_train, P_train], [Y_train, np.ones_like(Y_train)], **fit_opts)
    hist = history.losses

    # Save cost log to file
    with open('cost.log', 'a' if resume else 'w') as cost_log:
        line  = "# "
        line += ", ".join(['%s' % name for name in history.lossnames])
        line += " \n"
        cost_log.write(line) 

        cost_array = np.squeeze(np.array(zip(hist.values())))
        for row in range(cost_array.shape[1]):
            costs = list(cost_array[:,row])
            line = ', '.join(['%.4e' % cost for cost in costs])
            line += " \n"
            cost_log.write(line)    
            pass
        pass

    '''
       
    # ...
        
    return 0


# Main function call
if __name__ == '__main__':
    print ""
    main()
    pass
