#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing uBoost classifier for de-correlated jet tagging."""

# Basic import(s)
import gzip
import pickle

# Parallelisation import(s)
from joblib import Parallel, delayed

# Scientific import(s)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, roc_curve

# Project import(s)
from adversarial.utils import wpercentile, parse_args, initialise, load_data, mkdir
from adversarial.profile import profile, Profile


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, features, _ = load_data(args.input + 'data.h5', sample=0.01)  # @TEMP

    # Define classifier configuration(s)
    pattern = 'uboost_ur_{:4.2f}_te_92_rel21_fixed_def_cfg_1000boost'
    urs = [0.0, 0.01, 0.1, 0.3]
    classifiers = [('AdaBoost' if ur == 0 else 'uBoost (#alpha={:4.2f})'.format(ur), pattern.format(ur).replace('.', 'p')) for ur in urs]

    # Test classifiers in parallel
    njobs = min(7, len(classifiers))
    with Profile("Run tests in parallel"):
        Parallel(n_jobs=njobs)(delayed(test)(data, title, name) for title, name in classifiers)
        pass

    return 0


def test (data, title, name):
    """
    Common method to perform tests on named uBoost/Adaboost classifier.
    """

    with Profile("Testing {}".format(title)):

        # Loading classifier
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Loading {} classifier".format(title)):
            with gzip.open('models/uboost/{}.pkl.gz'.format(name), 'r') as f:
                clf = pickle.load(f)
                pass
            pass

        # Adding classifier variable
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Adding classifier variable"):
            data[name] = pd.Series(clf.predict_proba(data)[:,1], index=data.index)
            x = data[name]        .values
            w = data['weight_adv'].values
            pass


        # Plotting learning process
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Plotting learning process"):

            print("name:",name)

            # Get training/test split masks
            msk_train = data['train'] == 1
            msk_test  = data['train'] == 0

            # Get predictions
            pred_train = clf.predict_proba(data[msk_train])[:,1].flatten()
            pred_test  = clf.predict_proba(data[msk_test]) [:,1].flatten()

            # Get staged predicting, i.e. for each successive estimator
            staged_pred_train = list(clf.staged_predict_proba(data[msk_train]))
            staged_pred_test  = list(clf.staged_predict_proba(data[msk_test]))

            y_train = data.loc[msk_train, 'signal']    .values.flatten()
            y_test  = data.loc[msk_test,  'signal']    .values.flatten()
            w_train = data.loc[msk_train, 'weight_adv'].values.flatten()
            w_test  = data.loc[msk_test,  'weight_adv'].values.flatten()

            ll_train,  ll_test  = list(), list()
            auc_train, auc_test = list(), list()

            for idx, (pred_train, pred_test) in enumerate(zip(staged_pred_train, staged_pred_test)):
                p_train = pred_train[:,1]
                p_test  = pred_test [:,1]

                ll_train.append( log_loss(y_train, p_train, sample_weight=w_train) )
                ll_test .append( log_loss(y_test,  p_test,  sample_weight=w_test) )

                auc_train.append( roc_auc_score(y_train, p_train, sample_weight=w_train) )
                auc_test .append( roc_auc_score(y_test,  p_test,  sample_weight=w_test) )
                pass

            x = np.arange(len(ll_train))

            # Log-loss for each boosting iteration
            fig, ax = plt.subplots()
            plt.plot(x, ll_train, label='Train')
            plt.plot(x, ll_test,  label='Test')
            plt.legend()
            plt.xlabel("Boosting step / no. estimator")
            plt.ylabel("Log-loss")

            # Save
            mkdir('figures/')
            plt.savefig('figures/temp_logloss_{}.pdf'.format(name))

            # ROC AUC for each boosting iteration
            fig, ax = plt.subplots()
            plt.plot(x, auc_train, label='Train')
            plt.plot(x, auc_test,  label='Test')
            plt.legend()
            plt.xlabel("Boosting step / no. estimator")
            plt.ylabel("ROC AUC")

            # Save
            mkdir('figures/')
            plt.savefig('figures/temp_auc_{}.pdf'.format(name))
            pass

        '''

        # Plotting ROC curves
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Plotting ROCs"):

            fpr_test,  tpr_test,  thresholds_test  = roc_curve(y_test,  p_test,  sample_weight=w_test)
            fpr_train, tpr_train, thresholds_train = roc_curve(y_train, p_train, sample_weight=w_train)

            # TPR vs FPR
            fig, ax = plt.subplots()
            plt.plot(fpr_train, tpr_train, label='Train')
            plt.plot(fpr_test, tpr_test,   label='Test')
            plt.legend()
            plt.xlabel("Fake positive rate")
            plt.ylabel("True positive rate")

            # Save
            mkdir('figures/')
            plt.savefig('figures/temp_roc_{}.pdf'.format(name))
            pass


        # Plotting distributions
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Plotting distributions"):
            bins = np.linspace(-0.5, 1.5, 100 + 1, endpoint=True)
            fig, ax = plt.subplots()
            plt.hist(data.loc[data['signal'] == 1, name], bins=bins, weights=data.loc[data['signal'] == 1, 'weight_test'], alpha=0.5, label='Signal')
            plt.hist(data.loc[data['signal'] == 0, name], bins=bins, weights=data.loc[data['signal'] == 0, 'weight_test'], alpha=0.5, label='Background')
            plt.legend()
            plt.xlabel("uBoost classifier variable")
            plt.ylabel("Events")

            # Save
            mkdir('figures/')
            plt.savefig('figures/temp_dist_{}.pdf'.format(name))
            pass


        # Plotting efficiencies
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Plotting efficiencies"):

            #bins = np.linspace(40, 300, 2 * 26 + 1, endpoint=True)
            bins = np.linspace(50, 300, 1 * 26 + 1, endpoint=True)
            fig, ax = plt.subplots()

            # Loop training/test dataset
            for is_train in [0, 1]:

                # Select appropriate subset of data
                msk_bkg  = data['signal'] == 0
                msk_bkg &= data['train'] == is_train  # @TEMP

                # Loop all target inclusive efficiencies
                effs = [8, 10, 20, 50, 80]
                for icut, eff in enumerate(effs):

                    # Cut mask
                    cut = wpercentile(np.array(data.loc[msk_bkg, name]).flatten(), 100 - eff, weights=np.array(data.loc[msk_bkg, 'weight_test']).flatten())
                    msk_cut = data[name] > cut

                    # Compute numerator/denominator histograms
                    num, _   = np.histogram(data.loc[msk_bkg & msk_cut, 'm'], bins=bins, weights=data.loc[msk_bkg & msk_cut, 'weight_test'])
                    denom, _ = np.histogram(data.loc[msk_bkg,           'm'], bins=bins, weights=data.loc[msk_bkg,           'weight_test'])

                    # Cast
                    num   = num  .astype(np.float)
                    denom = denom.astype(np.float)

		    #print("mass: ",data.loc[msk_bkg,'m'].head(5))
		    #print("weight: ",data.loc[msk_bkg,'weight_test'].head(5))
		    #print("histo: ",denom)

                    # Plot efficiency profile
                    plt.plot(bins[:-1] + 0.5 * np.diff(bins), num/denom, color='b' if is_train else 'r', alpha=(100 - eff) / 100., label='{} > {:.2f} (bkg. eff. = {:.0f}%)'.format("uB", cut, eff) if is_train else None)   # ...format(title, cut, eff) ...
                    pass
                pass
            # Decorations
            plt.legend()
            plt.xlabel("Large-radius jet mass [GeV]")
            plt.ylabel("Background efficiency")

            # Save
            mkdir('figures/')
            plt.savefig('figures/temp_eff_{}.pdf'.format(name))
            pass
        #'''
        pass
    return

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
