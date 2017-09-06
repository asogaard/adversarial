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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

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
parser.add_argument('--threads',      dest='threads', action='store', type=int,
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

        # Initialise Keras backend
        initialise_backend(args)

        import keras
        import keras.backend as K
        from keras.models import Model
        from keras.callbacks import Callback
        from keras.utils.vis_utils import plot_model

        if args.tensorflow:
            import tensorflow as tf
            pass

        # ========== BEGIN: initialise
        """
        # Specify Keras backend and import module
        os.environ['KERAS_BACKEND'] = "tensorflow" if args.tensorflow else "theano"

        print_memory()

        # Configure backends
        if args.tensorflow:

            # Set print level to avoid unecessary warnings, e.g.
            #  $ The TensorFlow library wasn't compiled to use <SSE4.1, ...>
            #  $ instructions, but these are available on your machine and could
            #  $ speed up CPU computations.
            #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

            # Switch: CPU/GPU 
            if args.gpu:

                # Set this environment variable to "0,1,...", to make Tensorflow
                # use the first N available GPUs
                # @NOTE: No discernible improvement with multiple GPUs... Maybe 
                # under more demanding conditions? Or maybe we have to manually
                #  use the different devices through Tensorflow scoping.
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str,range(args.threads)))
                #if 'CUDA_VISIBLE_DEVICES' in  os.environ:
                #    del os.environ['CUDA_VISIBLE_DEVICES']
                #    pass
            else:
                # Setting this enviorment variable to "" makes all GPUs 
                # invisible to tensorflow, thus forcing it to run on CPU (on as 
                # many cores as possible)
                os.environ['CUDA_VISIBLE_DEVICES'] = ""
                pass

            # Reload the tensorflow module here to make sure only the correct
            # GPU devices are set up
            import tensorflow as tf

            # Get list of all available GPUs
            from tensorflow.python.client import device_lib
            local_device_protos = device_lib.list_local_devices()
            gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
            
            # Cap the thread count if too few GPUs are available
            if len(gpus) < args.threads:
                log.warning("Requested more threads than there are available GPUs ({} vs. {}). Restricting.".format(args.threads, len(gpus)))
                args.threads = len(gpus)
                pass
                
            # @TODO: Some smart selection of GPUs to used based on actual
            # utilisation?

            # Manually configure Tensorflow session
            #log.info("STDERR 2: " + "=" * 80)
            config = tf.ConfigProto(intra_op_parallelism_threads=1, 
                                    inter_op_parallelism_threads=1,
                                    allow_soft_placement=True, 
                                    device_count = {args.mode.upper(): args.threads},
                                    #log_device_placement=True,
                                    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1,
                                                              #allow_growth=True
                                                          ),
                                )
            session = tf.Session(config=config)

        else:

            if args.gpu:
                if args.threads > 1:
                    log.warning("Currently it is not possible to specify more than one GPU thread for Theano backend.")
                    pass
            else:
                # Set number of OpenMP threads to use; even if 1, set to force use
                # of OpenMP which doesn't happen otherwise, for some reason. Gives 
                # speed-up of factor of ca. 6-7. (60 sec./epoch -> 9 sec./epoch)
                os.environ['OMP_NUM_THREADS'] = str(args.threads)
                pass

            # Switch: CPU/GPU
            cuda_version = '8.0.61'
            dnn_flags = [
                'dnn.enabled=True',
                'dnn.include_path=/exports/applications/apps/SL7/cuda/{}/include/'.format(cuda_version),
                'dnn.library_path=/exports/applications/apps/SL7/cuda/{}/lib64/'.format(cuda_version)
                ]
            os.environ["THEANO_FLAGS"] = "device={},floatX=float32,openmp=True{}".format('cuda' if args.gpu else 'cpu', ','.join([''] + dnn_flags) if args.gpu else '')

            pass
        
        # Import Keras
        #log.info("STDERR 3: " + "=" * 80)
        print_memory()
        import keras
        import keras.backend as K
        from keras.callbacks import Callback
        from keras.utils.vis_utils import plot_model
        from keras.models import Model # For merged, stratified k-fold training
        K.set_floatx('float32')
        if args.tensorflow:
            K.set_session(session)
            pass

        #log.info("STDERR 4: " + "=" * 80)
        print_memory()

        # Check backend
        assert K.backend() == os.environ['KERAS_BACKEND'], \
            "Wrong Keras backend was loaded (%s vs. %s)." % (K.backend(), os.environ['KERAS_BACKEND'])
        """
        # ========== END: initialise
        
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
            pass
        
        # Get data- and info arrays
        datatreename = 'BoostedJet+ISRgamma/Nominal/EventSelection/Pass/NumLargeRadiusJets/Postcut'
        infotreename = 'BoostedJet+ISRgamma/Nominal/outputTree'
        prefix = 'Jet_'
        # @TODO: Is it easier/better to drop the prefix, list the names explicitly, and then rename manually afterwards? Or should the 'loadData' method support a 'rename' argument, using regex, with support for multiple such operations? That last one is probably the way to go, actually... In that case, I should probably also allow for regex support for branches? Like branches=['Jet_.*', 'leading_Photons_.*'], rename=[('Jet_', ''),]

        #branches = ['m', 'pt', 'C2', 'D2', 'ECF1', 'ECF2', 'ECF3', 'Split12', 'Split23', 'Split34', 'eta', 'leading_Photons_E', 'leading_Photons_eta', 'leading_Photons_phi', 'leading_Photons_pt', 'nTracks', 'phi', 'rho', 'tau21']

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

    print_memory()

    # Re-weighting to flatness (1D)
    # --------------------------------------------------------------------------
    with Profiler("Re-weighting (1D)"):
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
            if False:
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
            bkg = append_fields(bkg, 'reweight', new_weights, dtypes=float)

            # Appending similary ("dummy") 'reweight' field to signal sample, for consistency
            sig = append_fields(sig, 'reweight', sig['weight'], dtypes=float)
            pass

        pass

    print_memory()

    # Plotting: Re-weighting
    # --------------------------------------------------------------------------
    with Profiler("Plotting: Re-weighting"):
        
        # Plotting 1D priors for log(m) and log(pt)
        fig, ax = plt.subplots(2, 4, figsize=(12,6))

        w_bkg  = bkg['weight']
        rw_bkg = bkg['reweight']
        w_tar  = np.ones((N_tar,)) * np.sum(bkg['weight']) / float(N_tar)

        for row, var in enumerate(['m', 'pt']):
            edges = np.linspace(0, np.max(bkg[var]), 60 + 1, endpoint=True)
            nbins  = len(edges) - 1

            v_bkg  = bkg[var]     # Original    mass/pt values for the background
            rv_bkg = P_bkg[:,row] # Transformed mass/pt values for the background
            rv_tar = P_tar[:,row] # Transformed mass/pt values for the targer

            ax[row,0].hist(v_bkg,  bins=edges, weights=w_bkg,  alpha=0.5, label='Original')
            ax[row,1].hist(v_bkg,  bins=edges, weights=rw_bkg, alpha=0.5, label='Original')
            ax[row,2].hist(rv_bkg, bins=nbins, weights=rw_bkg, alpha=0.5, label='Original')
            ax[row,2].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')
            ax[row,3].hist(rv_bkg, bins=nbins, weights=rw_bkg, alpha=0.5, label='Original')
            ax[row,3].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')

            for col in range(4):
                if col < 3:
                    ax[row,col].set_yscale('log')
                    ax[row,col].set_ylim(1E+01, 1E+06)
                    pass
                ax[row,col].set_xlabel("Jet %s%s" % (var, " (transformed)" if col > 1 else ''))
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

    print "=" * 80
    print_memory()

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
        Y = np.hstack((np.ones(N_sig, dtype=int), np.zeros(N_bkg, dtype=int)))
        
        # Input(s)
        X_sig = np.vstack(tuple(sig[var] for var in names)).T
        X_bkg = np.vstack(tuple(bkg[var] for var in names)).T
        
        # De-correlation variable(s): These have already been manually define above
        #P_sig = np.vstack(tuple(sig[var] for var in decorrelation_vars)).T
        #P_bkg = np.vstack(tuple(bkg[var] for var in decorrelation_vars)).T
        
        # Data pre-processing
        # @NOTE: If we're using batch normalisation after the input layer, this 
        # scaling should be done automatically. But no harm in doing it either way? 
        # Depends on whether we _want_ batch normalisation. That should probably be
        # left for the hyperparameter optimisation to decide. In that case, we 
        # should probably perform the manual scaling either way.
        substructure_scaler = StandardScaler().fit(X_bkg)
        X_sig = substructure_scaler.transform(X_sig)
        X_bkg = substructure_scaler.transform(X_bkg)
        
        # This is already done by hand above
        #decorrelation_scaler = StandardScaler().fit(P_bkg)
        #P_sig = decorrelation_scaler.transform(P_sig)
        #P_bkg = decorrelation_scaler.transform(P_bkg)
        
        # Concatenate signal and background samples
        X = np.vstack((X_sig, X_bkg)).astype(K.floatx())
        P = np.vstack((P_sig, P_bkg)).astype(K.floatx())
        pass

    print_memory()

    # Classifier-only fit
    # --------------------------------------------------------------------------
    # Adapted from: https://github.com/asogaard/AdversarialSubstructure/blob/master/train.py
    with Profiler("Classifier-only fit"):
        # @TODO: - Add 'train' flag
        #        - Implement checkpointing
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
        # Adapted from [https://github.com/fchollet/keras/issues/1711#issuecomment-185801662]
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True)
        histories = list()

        # Get minimal number of samples for each fold, to ensure proper
        # batching. This will on average lead to a loss of around `args.folds /
        # 2` samples per classifier, which is something we can live with.
        train_samples = min(len(train) for train, _ in skf.split(X,Y))
        test_samples  = min(len(test)  for _, test  in skf.split(X,Y))

        # Importe module creator methods and optimiser options
        from adversarial.models import opts, classifier_model, adversarial_model

        # Create a dummy device/context manager, for Theano
        class DummyDevice (object):
            def __init__ (self, *args, **kwargs): return
            def __enter__(self, *args, **kwargs): return
            def __exit__ (self, *args, **kwargs): return
            pass

        # Crate 'folds' classifiers to train separately. If using Tensorflow, 
        # create the classifier on each of the 'threads' available GPUs.
        # ... @TODO
        classifiers = list()
        train_inputs, train_targets, train_weights = list(), list(), list()
        test_inputs,  test_targets,  test_weights  = list(), list(), list()
        #train_generator, test_generator = list(), list()
        for fold, (train, test) in enumerate(skf.split(X,Y)):
            
            # Choose device on which the model should live
            #device = tf.device('/{mode}:{idx}'.format(mode='gpu' if args.gpu
            #else 'cpu', idx=fold % args.threads)) if args.tensorflow else
            #DummyDevice ...

            device_name = '/{mode}:{idx}'.format(mode=args.mode, idx=fold%args.threads)
            device      = tf.device if args.tensorflow else DummyDevice

            with device(device_name):
                log.info("Creating classifier {} on device '{}'".format(fold + 1, device_name))
            
                # Create new classifier model instance
                classifier = classifier_model(X.shape[1],
                                              default=cfg['classifier']['default'], 
                                              architecture=cfg['classifier']['architecture'],
                                              name='classifier_{}'.format(fold))
            
                # Explicitly move data to the target GPU
                ###train_inputs .append(tf.constant(X[train[:train_samples],:]))
                ###train_targets.append(tf.constant(Y[train[:train_samples]]))
                ###train_weights.append(tf.constant(W[train[:train_samples]]))
                ###
                ###test_inputs  .append(tf.constant(X[test [:test_samples],:]))
                ###test_targets .append(tf.constant(Y[test [:test_samples]]))
                ###test_weights .append(tf.constant(W[test [:test_samples]]))
                
                ###train_generator.append((tf.constant(              X[train[:train_samples],:]),
                ###tf.constant(np.atleast_2d(Y[train[:train_samples]]).T),
                ###tf.constant(              W[train[:train_samples]])))
                ###
                ###test_inputs  .append((tf.constant(              X[test [:test_samples],:]),
                ###tf.constant(np.atleast_2d(Y[test [:test_samples]]).T),
                ###tf.constant(              W[test [:test_samples]])))
                
                #pass

                # Compile with optimiser configuration
                classifier.compile(**opts['classifier'])        

                # Save to list
                classifiers.append(classifier)
                pass


            # Save classifier model diagram to file
            if fold == 0:
                plot_model(classifier, to_file=args.output + 'classifier.png', show_shapes=True)
                pass                
            pass

        print_memory()
        # Perform stratified k-fold training
        with Profiler("Classifier-only training"):
            '''
            # Create merged model
            merged = Model(inputs =[clf.input  for clf in classifiers], 
                           outputs=[clf.output for clf in classifiers], 
                           name='StratifiedTraining')
            # Compile
            merged.compile(**opts['classifier'])        
            
            # Save
            plot_model(merged, to_file=args.output + 'merged.png', show_shapes=True)
            
            # Concatenate inputs, targets, and weights, and ensure that they all
            # have the same number of samples (to allow form proper
            # batching). 
            #train_samples, test_samples = np.min([(len(train), len(test)) for train, test in skf.split(X,Y)], axis=0)

            #train_inputs = list()
            #train_targets = list()
            #train_weights = list()
            
            #test_inputs = list()
            #test_targets = list()
            #test_weights = list()
            
            train_inputs  = [X[train[:train_samples],:] for train,_ in skf.split(X,Y)]
            train_targets = [Y[train[:train_samples]]   for train,_ in skf.split(X,Y)]
            train_weights = [W[train[:train_samples]]   for train,_ in skf.split(X,Y)]
            
            test_inputs   = [X[test[:test_samples],:]   for _,test  in skf.split(X,Y)]
            test_targets  = [Y[test[:test_samples]]     for _,test  in skf.split(X,Y)]
            test_weights  = [W[test[:test_samples]]     for _,test  in skf.split(X,Y)]
            
            # Ensure same number of samples
            ###train_samples = np.min(M.shape[0] for M in train_targets)
            ###test_samples  = np.min(M.shape[0] for M in test_targets)
            ###
            ###train_inputs  = [M[:train_samples,:] for M in train_inputs]
            ###train_targets = [M[:train_samples]   for M in train_targets]
            ###train_weights = [M[:train_samples]   for M in train_weights]
            ###
            ###test_inputs   = [M[:test_samples,:]  for M in test_inputs]
            ###test_targets  = [M[:test_samples]    for M in test_targets]
            ###test_weights  = [M[:test_samples]    for M in test_weights]
            
            # Fit
            hist = merged.fit(train_inputs, train_targets, sample_weight=train_weights, validation_data=(test_inputs, test_targets, test_weights), **cfg['classifier']['fit'])
            #hist = merged.fit_generator(train_generator, 1, validation_data=test_generator)#, **cfg['classifier']['fit'])
            '''
            # ----- @TODO -----
            #'''
            for fold, (train, test) in enumerate(skf.split(X,Y)):                

                # Fit classifier model
                with Profiler("Fold %d/%d" % (fold + 1, args.folds)):
                    print "- " * 40
                    print_memory()
                    hist = classifiers[fold].fit(X[train,:], Y[train], sample_weight=W[train], validation_data=(X[test,:], Y[test], W[test]), **cfg['classifier']['fit'])
                    histories.append(hist.history)
                    pass
                pass        
            #'''

            pass

        print histories

        return

        with Profiler():
            # Log history (?)
            fig, ax = plt.subplots()
            plt.plot(range(len(history.losses['loss'])), history.losses['loss'])
            plt.savefig(args.output + 'costlog.pdf')
            
            # Save classifier model to file
            classifier.save('trained/classifier.h5')
            pass
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
        """ Update the learning rate of the two optimisers. """
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
