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
seed = 21 # For reproducibility
np.random.seed(seed)

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
from adversarial.utils   import *
from adversarial.profile import *
from adversarial.plots   import plot_posterior, plot_profiles

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
parser.add_argument('--devices',     dest='devices', action='store', type=int,
                    default=1, help='Number of CPU/GPU devices to use with Tensorflow.')
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
    with Profile("Initialisation"):

        # Parse command-line arguments
        args = parser.parse_args()

        # Add 'mode' field manually
        args = argparse.Namespace(mode='gpu' if args.gpu else 'cpu', **vars(args))

        # Set print level
        log.basicConfig(format="%(levelname)s: %(message)s", 
                        level=log.DEBUG if args.verbose else log.INFO)

        #  Modify input/output directory names to conform to convention
        if not args.input .endswith('/'): args.input  += '/'
        if not args.output.endswith('/'): args.output += '/'

        # @TODO:
        # - Make `args = prepare_args  (args)` method?
        # - Make `cfg  = prepare_config(args)` method?
        
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

        # Set adversary learning rate (LR) ratio from ratio of loss_weights
        cfg['combined']['model']['lr_ratio'] = cfg['combined']['compile']['loss_weights'][0] / \
                                               cfg['combined']['compile']['loss_weights'][1]

        # Initialise Keras backend
        initialise_backend(args)

        import keras
        import keras.backend as K
        from keras.models import load_model
        from keras.callbacks import Callback
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

        pass
    

    # @TODO: Turn 'Loading data', 'Re-weighting', 'Data preparation' into
    # utility function(s), for use with separate `evaluate.py` script
    
    # Loading data
    # --------------------------------------------------------------------------
    with Profile("Loading data"):
        
        # Get paths for files to use
        with Profile():
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
        with Profile():
            sig_data = loadData(sig_paths, datatreename, prefix=prefix) # ..., branches=branches)
            sig_info = loadData(sig_paths, infotreename, stop=1)
            pass

        with Profile():
            bkg_data = loadData(bkg_paths, datatreename, prefix=prefix) # ..., branches=branches)
            bkg_info = loadData(bkg_paths, infotreename, stop=1)
            pass
        
        log.info("Retrieved data columns: [%s]" % (', '.join(sig_data.dtype.names)))
        log.info("Retrieved %d signal and %d background events." % (sig_data.shape[0], bkg_data.shape[0]))
        
        # Scale by cross section
        with Profile():
            log.debug("Scaling weights by cross-section and luminosity")
            xsec = loadXsec('share/sampleInfo.csv')
        
            sig = scale_weights(sig_data, sig_info, xsec=xsec, lumi=36.1)
            bkg = scale_weights(bkg_data, bkg_info, xsec=xsec, lumi=36.1)
            pass
        
        # Restricting phase space
        with Profile():
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
    with Profile("Re-weighting"):
        # @NOTE: This is the crucial point: If the target is flat in (m,pt) the
        # re-weighted background _won't_ be flat in (log m, log pt), and vice 
        # versa. It should go without saying, but draw target samples from a 
        # uniform prior on the coordinates which are used for the decorrelation.

        decorrelation_variables = ['m']#, 'pt']
        
        # Performing pre-processing of de-correlation coordinates
        with Profile():
            log.debug("Performing pre-processing")

            # Get number of background events and number of target events (arb.)
            N_sig = len(sig)
            N_bkg = len(bkg)
            N_tar = len(bkg)
            
            # Initialise and fill coordinate arrays
            P_sig = np.zeros((N_sig, len(decorrelation_variables)), dtype=float)
            P_bkg = np.zeros((N_bkg, len(decorrelation_variables)), dtype=float)
            for col, var in enumerate(decorrelation_variables):
                P_sig[:,col] = np.log(sig[var])
                P_bkg[:,col] = np.log(bkg[var])
                pass            
            #P_sig[:,1] = np.log(sig['pt'])
            #P_bkg[:,1] = np.log(bkg['pt'])
            P_tar = np.random.rand(N_tar, len(decorrelation_variables))
            
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
        with Profile():
            log.debug("Performing re-weighting using GBReweighter")
            reweighter_filename = 'trained/reweighter_{}d.pkl.gz'.format(len(decorrelation_variables))
            if not os.path.isfile(reweighter_filename):
                reweighter = GBReweighter(n_estimators=80, max_depth=7)
                reweighter.fit(P_bkg, target=P_tar, original_weight=bkg['weight'])
                log.info("Saving re-weighting object to file '%s'" % reweighter_filename)
                with gzip.open(reweighter_filename, 'wb') as f:
                    pickle.dump(reweighter, f)
                    pass
            else:
                log.info("Loading re-weighting object from file '%s'" % reweighter_filename)
                with gzip.open(reweighter_filename, 'r') as f:
                    reweighter = pickle.load(f)
                    pass
                pass
            pass

        # Re-weight for uniform prior(s)
        with Profile():
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
    with Profile("Plotting: Re-weighting"):
        

        fig, ax = plt.subplots(2, 4, figsize=(12,6))

        w_bkg  = bkg['weight']
        rw_bkg = bkg['reweight']
        w_tar  = np.ones((N_tar,)) * np.sum(bkg['weight']) / float(N_tar)

        for row, var in enumerate(decorrelation_variables):
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
        if len(decorrelation_variables) == 2:
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
        pass


    # Prepare arrays for training
    # --------------------------------------------------------------------------
    with Profile("Data preparation"):

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
        U = W # @TEMP

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

        # Define single data container
        data = dict(X=X, Y=Y, P=P, W=W, U=U, sig=sig, bkg=bkg)
        pass


    # Classifier-only fit
    # --------------------------------------------------------------------------
    # Adapted from: https://github.com/asogaard/AdversarialSubstructure/blob/master/train.py
    # Resources:
    #  [https://github.com/fchollet/keras/issues/7515]
    #  [https://stackoverflow.com/questions/43821786/data-parallelism-in-keras]
    #  [https://stackoverflow.com/a/44771313]
    
    with Profile("Classifier-only fit, cross-validation"):
        # @TODO: - Implement checkpointing

        # Define variables
        basename = 'crossval_classifier'
        
        # Get indices for each fold in stratified k-fold training
        skf = StratifiedKFold(n_splits=args.folds)

        # Importe module creator methods and optimiser options
        from adversarial.models import classifier_model, adversary_model, combined_model, classifier_from_combined

        # Create unique set of random indices to use with stratification
        random_indices = np.arange(num_samples)
        np.random.shuffle(random_indices)

        # Collection of classifiers and their associated training histories
        classifiers = list()
        histories   = list()

        # Train or load classifiers
        if args.train:
            log.info("Training cross-validation classifiers")
            
            # Loop `k` folds
            for fold, (train, validation) in enumerate(skf.split(X,Y)):
                with Profile("Fold {}/{}".format(fold + 1, args.folds)):

                    # StratifiedKFold provides stratification, but since the
                    # input arrays are not randomised, neither will the
                    # folds. Therefore, the fold should be taken with respect to
                    # a set of randomised indices rather than range(N).
                    train      = random_indices[train]
                    validation = random_indices[validation]

                    # Define unique tag and name for current classifier
                    tag  = '{}of{}'.format(fold + 1, args.folds)
                    name = '{}__{}'.format(basename, tag)

                    # Get classifier
                    classifier = classifier_model(num_features, **cfg['classifier']['model'])

                    # Compile model (necessary to save properly)
                    classifier.compile(**cfg['classifier']['model'])
                    
                    # Fit classifier model                    
                    result = train_in_parallel(classifier,
                                               {'input':   X,
                                                'target':  Y,
                                                'weights': W,
                                                'mask':    train},
                                               {'input':   X,
                                                'target':  Y,
                                                'weights': W,
                                                'mask':    validation},
                                               config=cfg['classifier'],
                                               num_devices=args.devices, mode=args.mode, seed=seed)

                    histories.append(result['history'])
                    
                    # Save classifier model and training history to file, both
                    # in unique output directory and in the directory for
                    # pre-trained classifiers
                    for destination in [args.output, 'trained/']:
                        classifier.save        (destination + '{}.h5'        .format(name))
                        classifier.save_weights(destination + '{}_weights.h5'.format(name))
                        with open(destination + 'history__{}.json'.format(name), 'wb') as f:
                            json.dump(result['history'], f)
                            pass
                        pass
                    
                    # Add to list of classifiers
                    classifiers.append(classifier)
                    pass
                pass
        else:
            log.info("Loading cross-validation classifiers from file")
            
            # Load pre-trained classifiers
            classifier_files = sorted(glob.glob('trained/{}__*of{}.h5'.format(basename, args.folds)))
            assert len(classifier_files) == args.folds, "Number of pre-trained classifiers ({}) does not match number of requested folds ({})".format(len(classifier_files), args.folds)
            for classifier_file in classifier_files:
                classifiers.append(load_model(classifier_file))
                pass

            # Load associated training histories
            history_files = sorted(glob.glob('trained/history__{}__*of{}.json'.format(basename, args.folds)))
            assert len(history_files) == args.folds, "Number of training histories for pre-trained classifiers ({}) does not match number of requested folds ({})".format(len(history_files), args.folds)
            for history_file in history_files:
                with open(history_file, 'r') as f:
                    histories.append(json.load(f))
                    pass
                pass

            pass # end: train/load
        pass
        

    # Plotting: Cost log for classifier-only fit
    # --------------------------------------------------------------------------
    # Optimal number of training epochs
    opt_epochs = None
    
    with Profile("Plotting: Cost log, cross-val."):        

        fig, ax = plt.subplots()
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
        
        # @NOTE: Assuming no early stopping
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
        plt.xlabel("Training epochs",    horizontalalignment='right', x=1.0)
        plt.ylabel("Objective function", horizontalalignment='right', y=1.0)
        
        epochs = [0] + list(epochs)
        step = max(int(np.floor(len(epochs) / 10.)), 1)
        
        plt.xticks(filter(lambda x: x % step == 0, epochs))
        plt.legend()
        plt.savefig(args.output + 'costlog_classifier.pdf')
        pass
    

    # Classifier-only fit, full
    # --------------------------------------------------------------------------
    with Profile("Classifier-only fit, full"):

        # Define variables
        name = 'full_classifier'

        if args.train:
            log.info("Training full classifier")
            
            # Get classifier
            classifier = classifier_model(num_features, **cfg['classifier']['model'])

            # Compile model (necessary to save properly)
            classifier.compile(**cfg['classifier']['compile'])
            
            # Overwrite number of training epochs with optimal number found from
            # cross-validation
            cfg['classifier']['fit']['epochs'] = opt_epochs
            
            # Train final classifier
            result = train_in_parallel(classifier,
                                       {'input':   X,
                                        'target':  Y,
                                        'weights': W},
                                       config=cfg['classifier'],
                                       mode=args.mode,
                                       num_devices=args.devices,
                                       seed=seed)

            # Save classifier model and training history to file, both
            # in unique output directory and in the directory for
            # pre-trained classifiers
            for destination in [args.output, 'trained/']:
                classifier.save        (destination + '{}.h5'        .format(name))
                classifier.save_weights(destination + '{}_weights.h5'.format(name))
                with open(destination + 'history__{}.json'.format(name), 'wb') as f:
                    json.dump(result['history'], f)
                    pass
                pass
            
        else:

            log.info("Loading full classifier from file")
            
            # Load pre-trained classifiers
            classifier_file = 'trained/{}.h5'.format(name)
            classifier = load_model(classifier_file)

            # Load associated training histories
            history_file = 'trained/history__{}.json'.format(name)
            with open(history_file, 'r') as f:
                history = json.load(f)
                pass

            pass # end: train/load

        # Save classifier model diagram to file
        plot_model(classifier, to_file=args.output + 'model_classifier.png', show_shapes=True)    

        # ...
        
        # Store classifier output as tagger variables. @NOTE This works only
        # _provided_ the input array X has the same ordering as sig/bkg.
        msk_sig = (Y == 1.)
        sig = append_fields(sig, 'NN', classifier.predict(X[ msk_sig], batch_size=1024).flatten(), dtypes=K.floatx())
        bkg = append_fields(bkg, 'NN', classifier.predict(X[~msk_sig], batch_size=1024).flatten(), dtypes=K.floatx())

        # Update `data` container
        data = dict(X=X, Y=Y, P=P, W=W, U=U, sig=sig, bkg=bkg)        
        pass

    # '''
    # def plot_posterior (adversary, Y, P, U, destination=args.output, name='posterior', title=''):
    # " ""..."" "
    # 
    # # Create figure
    # fig, ax = plt.subplots()
    # 
    # # Variable definitions
    # edges  = np.linspace(-0.2, 1.2, 2 * 70 + 1, endpoint=True)
    # 
    # z_slices  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # #P2_slices = [0, 1]
    # P1        = np.linspace(0, 1, 1000 + 1, endpoint=True)
    # 
    # linestyles =  ['-', '--', '-.', ':']
    # colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
    # 
    # # Plot prior
    # msk = (Y == 0)
    # plt.hist(P[:,0][msk], bins=edges, weights=U[msk], normed=True, color='gray', alpha=0.5, label='Prior')
    # 
    # # Plot adversary posteriors
    # # for i, P2_slice in enumerate(P2_slices):
    # # P2 = np.ones_like(P1) * P2_slice
    # # P_test = np.vstack((P1, P2)).T
    # P_test = P1
    # posteriors = list()
    # for j, z_slice in enumerate(z_slices):
    # z_test = np.ones_like(P1) * z_slice
    # posterior = adversary.predict([z_test, P_test])
    # posteriors.append(posterior)
    # plt.plot(P1, posterior, color=colours[j], label='clf. = %.1f' % z_slices[j])
    # pass
    # 
    # # Decorations
    # plt.xlabel("Normalised jet log(m)", horizontalalignment='right', x=1.0)
    # plt.ylabel("Jets",                  horizontalalignment='right', y=1.0)
    # plt.title("De-correlation p.d.f.'s{}".format((': ' + title) if title else ''), fontweight='medium')
    # plt.ylim(0, 5.)
    # plt.legend()
    # 
    # # Save figure
    # plt.savefig(destination + name + '.pdf')
    # plt.close(fig)
    # return
    # '''

    class LRCallback (Callback):
        def on_epoch_end (self, epoch, logs={}):
            optimizer = self.model.optimizer
            lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
            print('\nLR: {:.6f} / ({:6f}) after {:f} iterations\n'.format(lr, K.eval(optimizer.lr), K.eval(optimizer.iterations)))
            return
        pass


    class PosteriorCallback (Callback):
        def __init__ (self, data, args, adversary):
            self.opts = dict(data=data, args=args, adversary=adversary)
            return
        
        def on_train_begin (self, logs={}):
            plot_posterior(name='posterior_begin', title="Beginning of training", **self.opts)
            return
        
        def on_epoch_end (self, epoch, logs={}):
            plot_posterior(name='posterior_epoch_{:03d}'.format(epoch + 1), title="Epoch {}".format(epoch + 1), **self.opts)
            return
        pass


    class ProfilesCallback (Callback):
        def __init__ (self, data, args, var):
            self.opts = dict(data=data, args=args, var=var)
            return
        
        def on_train_begin (self, logs={}):
            plot_profiles(name='profiles_begin', title="Beginning of training", **self.opts)
            return
        
        def on_epoch_end (self, epoch, logs={}):
            plot_profiles(name='profiles_epoch_{:03d}'.format(epoch + 1), title="Epoch {}".format(epoch + 1), **self.opts)
            return
        pass


    # Combined fit, full (@TODO: Cross-val?)
    # --------------------------------------------------------------------------
    with Profile("Combined fit, full"):
        # @TODO: - Make work
        #        - Train/load
        #        - Add 'combined' category to configs
        #        - Checkpointing
        #        - Add per-epoch callback logging mean(s) and width(s) of GMM
        #        component(s) for fixed percentiles of classifier output.

        # Define variables
        name = 'full_combined'

        # Set up adversary
        adversary = adversary_model(gmm_dimensions=P.shape[1],
                                    **cfg['adversary']['model'])            

        # Save adversarial model diagram
        plot_model(adversary, to_file=args.output + 'model_adversary.png', show_shapes=True)

        # Create callback logging the adversary p.d.f.'s during training
        callback_posterior = PosteriorCallback(data, args, adversary)

        # Create callback logging the adversary p.d.f.'s during training
        callback_profiles  = ProfilesCallback(data, args, classifier)

        # Create callback that tracks the learning rate during training
        callback_lr = LRCallback()

        # Set up combined, adversarial model
        combined = combined_model(classifier,
                                  adversary,
                                  **cfg['combined']['model'])
        
        # Save combiend model diagram
        plot_model(combined, to_file=args.output + 'model_combined.png', show_shapes=True)
            
        if args.train or True: # @TEMP
            log.info("Training full, combined model")

            # Create custom objective function for posterior: - log(p) of the
            # posterior p.d.f.
            def maximise (p_true, p_pred):
                return - K.log(p_pred)

            cfg['combined']['compile']['loss'][1] = maximise

            # Compile model (necessary to save properly)
            combined.compile(**cfg['combined']['compile'])            

            # Train final classifier
            classifier.trainable = True # @TEMP
            result = train_in_parallel(combined,
                                       {'input':   [X, P],
                                        'target':  [Y, np.ones_like(Y)],
                                        'weights': [U, np.multiply(U, 1 - Y)]},
                                       config=cfg['combined'],
                                       mode=args.mode,
                                       num_devices=args.devices,
                                       seed=seed,
                                       callbacks=[callback_posterior, callback_lr]) # @TEMP ..., callback_profiles, ...
            
            # Save combined model and training history to file, both
            # in unique output directory and in the directory for
            # pre-trained classifiers
            history = result['history']
            for destination in [args.output, 'trained/']:
                combined.save        (destination + '{}.h5'        .format(name))
                combined.save_weights(destination + '{}_weights.h5'.format(name))
                with open(destination + 'history__{}.json'.format(name), 'wb') as f:
                    json.dump(history, f)
                    pass
                pass
            
        else:

            log.info("Loading full, combined model from file")

            # Improt GradientReversalLayerto allow reading of model
            from adversarial.layers import GradientReversalLayer, PosteriorLayer

            # Load pre-trained combined
            # combined_file = 'trained/{}.h5'.format(name)
            # combined = load_model(combined_file, custom_objects={
            # 'GradientReversalLayer': GradientReversalLayer,
            # 'PosteriorLayer':        PosteriorLayer
            # })
            combined_weights_file = 'trained/{}_weights.h5'.format(name)
            combined.load_weights(combined_weights_file)

            # @TODO: Instead:
            # - Create classifier and adversary
            # - From the, create combined
            # - Read in _weights_ only
            # - Use classifier and combined with stored weights

            # Extract classifier from loaded combined
            classifier = classifier_from_combined(combined)

            # Load associated training histories
            history_file = 'trained/history__{}.json'.format(name)
            with open(history_file, 'r') as f:
                history = json.load(f)
                pass
            
            pass
            
        # Store classifier output as tagger variables. @NOTE This works only
        # _provided_ the input array X has the same ordering as sig/bkg.
        msk_sig = (Y == 1.)
        sig = append_fields(sig, 'ANN', classifier.predict(X[ msk_sig], batch_size=1024).flatten(), dtypes=K.floatx())
        bkg = append_fields(bkg, 'ANN', classifier.predict(X[~msk_sig], batch_size=1024).flatten(), dtypes=K.floatx())
        
        # Update `data` container
        data = dict(X=X, Y=Y, P=P, W=W, U=U, sig=sig, bkg=bkg)        
        pass

    plot_posterior(data, args, adversary, name='posterior_end', title="End of training")


    # Plotting: Cost log for adversarial fit
    # --------------------------------------------------------------------------
    with Profile("Plotting: Cost log, adversarial, full"):        

        fig, ax = plt.subplots()
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
        epochs = 1 + np.arange(len(history['loss']))
        lambda_reg = cfg['combined']['model']['lambda_reg']
        lr_ratio   = cfg['combined']['model']['lr_ratio']        

        classifier_loss = np.mean([loss for key,loss in history.iteritems() if key.startswith('combined') and int(key.split('_')[-1]) % 2 == 1 ], axis=0)
        adversary_loss  = np.mean([loss for key,loss in history.iteritems() if key.startswith('combined') and int(key.split('_')[-1]) % 2 == 0 ], axis=0) * lambda_reg
        combined_loss   = classifier_loss + adversary_loss
        
        plt.plot(epochs, classifier_loss, color=colours[0],  linewidth=1.4,  label='Classifier')
        plt.plot(epochs, adversary_loss,  color=colours[1],  linewidth=1.4,  label=r'Adversary ($\lambda$ = {})'.format(lambda_reg))
        plt.plot(epochs, combined_loss,   color=colours[-1], linestyle='--', label='Combined')

        plt.title('Adversarial training', fontweight='medium')
        plt.xlabel("Training epochs", horizontalalignment='right', x=1.0)
        plt.ylabel("Objective function",   horizontalalignment='right', y=1.0)
        #ax.set_yscale('log')
        
        epochs = [0] + list(epochs)
        step = max(int(np.floor(len(epochs) / 10.)), 1)
        
        plt.xticks(filter(lambda x: x % step == 0, epochs))
        plt.legend()
        plt.savefig(args.output + 'costlog_combined.pdf')
        pass

    
    # Plotting: Distributions/ROC
    # --------------------------------------------------------------------------
    with Profile("Plotting: Distributions/ROC"):

        # Tagger variables
        variables = ['tau21', 'D2', 'NN', 'ANN']

        # Plotted 1D tagger variable distributions
        fig, ax = plt.subplots(1, len(variables), figsize=(len(variables) * 4, 4))

        for ivar, var in enumerate(variables):

            # Get axis limits
            if var == 'D2':
                edges = np.linspace(0, 5, 50 + 1, endpoint=True)
            else:
                edges = np.linspace(0, 1, 50 + 1, endpoint=True)
                pass

            # Get value- and weight arrays
            v_sig = np.array(sig[var])
            v_bkg = np.array(bkg[var])

            w_sig = np.array(sig['weight'])
            w_bkg = np.array(bkg['weight'])

            # Mask out NaN's
            msk = ~np.isnan(v_sig)
            v_sig = v_sig[msk]
            w_sig = w_sig[msk]
            
            msk = ~np.isnan(v_bkg)
            v_bkg = v_bkg[msk]
            w_bkg = w_bkg[msk]

            # Plot distributions
            ax[ivar].hist(v_bkg, bins=edges, weights=w_bkg, alpha=0.5, normed=True, label='Background')
            ax[ivar].hist(v_sig, bins=edges, weights=w_sig, alpha=0.5, normed=True, label='Signal')

            ax[ivar].set_xlabel("Jet {}".format(var),
                                horizontalalignment='right', x=1.0)

            ax[ivar].set_ylabel("Jets / {:.3f}".format(np.diff(edges)[0]),
                                horizontalalignment='right', y=1.0)
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


    # Plotting: Profiles
    # --------------------------------------------------------------------------
    with Profile("Plotting: Profiles"):
        '''
        # Plotting variables
        edges = np.linspace(0, 300, 60 + 1, endpoint=True)
        bins  = edges[:-1] + 0.5 * np.diff(edges)
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
        
        step = 10.
        percentiles = np.linspace(step, 100 - step, int(100 / step) - 1, endpoint=True)
        '''
        # Loop tagger variables
        for var in ['tau21', 'tau21DDT', 'D2', 'NN', 'ANN']:

            plot_profiles(data, args, var)
            
            '''
            profiles = [[] for _ in percentiles]

            fig, ax = plt.subplots()

            # Loop edges
            for (mass_down, mass_up) in zip(edges[:-1], edges[1:]):
                
                # Get array of `var` within the current jet-mass band
                msk = (bkg['m'] >= mass_down) & (bkg['m'] < mass_up)
                array = bkg[var][msk]
                
                # Loop percentiles
                for idx, perc in enumerate(percentiles):
                    if len(array) == 0:
                        profiles[idx].append(np.nan)
                    else:
                        profiles[idx].append(np.percentile(array, perc))
                        pass
                    pass # end: loop percentiles

                pass # end: loop edges

            # Plot profile
            for profile, perc in zip(profiles, percentiles):
                plt.plot(bins, profile, color=colours[0], linewidth=2 if perc == 50 else 1, label='Median' if perc == 50 else None)
                pass

            # Add text
            mid = len(percentiles) // 2

            arr_profiles = np.array(profiles).flatten()
            arr_profiles = arr_profiles[~np.isnan(arr_profiles)]
            diff = np.max(arr_profiles) - np.min(arr_profiles)

            opts = dict(horizontalalignment='center', verticalalignment='bottom', fontsize='x-small')
            text_string = r"$\varepsilon_{bkg.}$ = %d%%"

            plt.text(edges[-1], profiles[-1] [-1] + 0.02 * diff, text_string % percentiles[-1],  **opts) # 90%

            opts = dict(horizontalalignment='left', verticalalignment='center', fontsize='x-small')
            text_string = "%d%%"

            plt.text(edges[-1], profiles[mid][-1], text_string % percentiles[mid], **opts) # 50%
            plt.text(edges[-1], profiles[0]  [-1], text_string % percentiles[0],   **opts) # 10%
            
            plt.xlabel("Jet mass [GeV]",  horizontalalignment='right', x=1.0)
            plt.ylabel("{}".format(var),  horizontalalignment='right', y=1.0)
            plt.title('Percentile profiles for {}'.format(var), fontweight='medium')
            plt.legend()
            plt.savefig(args.output + 'tagger_profile_{}.pdf'.format(var))
            '''
            pass # end: loop variables

        pass


    # Plotting: Posterior p.d.f.'s
    # --------------------------------------------------------------------------
    """
    with Profile("Plotting: Posterior p.d.f.'s"):

        masses = np.linspace(-0.2, 1.2, 35 + 1, endpoint=True)
        pts    = np.linspace(-0.2, 1.2, 35 + 1, endpoint=True)
        params = np.vstack([M.flatten() for M in np.meshgrid(masses, pts)]).T

        fig, ax = plt.subplots()
        step = 0.2
        for z in linspace(step, 1 - step, int(1 / float(step)) - 1, endpoint=True):
            adverary.predict(np.ones((params.shape[0],)) * z, params)
            pass

        plt.xlabel("Log-scaled and normalised jet mass", horizontalalignment='right', x=1.0)
        plt.ylabel("Probability density",                horizontalalignment='right', y=1.0)
        plt.title("Adversary posterior p.d.f.'s",        fontweight='medium')
        plt.legend()
        plt.savefig(args.output + 'adverary_posterior.pdf')

        pass
        """
        
    # ...

    return 0


# Main function call
if __name__ == '__main__':
    print ""
    main()
    pass
