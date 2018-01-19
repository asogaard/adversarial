#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing uBoost classifier for de-correlated jet tagging."""

# Basic import(s)
import pickle

# Scientific import(s)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score

# Project import(s)
from adversarial.utils import wpercentile
from adversarial.new_utils import parse_args, initialise, load_data
from adversarial.profile import profile, Profile


# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data(args.input + 'data.h5')

    # Subset @TEMP
    data = data.head(1000000)


    # Loading uBoost classifier
    # --------------------------------------------------------------------------
    with Profile("Loading uBoost classifier"):
        with open('models/uboost/uboost.pkl', 'r') as f:
            clf = pickle.load(f)
            pass
        pass


    # Adding classifier variable
    # --------------------------------------------------------------------------
    with Profile("Adding classifier variable"):
        data['uB'] = pd.Series(clf.predict_proba(data)[:,1], index=data.index)
        x = data['uB']    .as_matrix().flatten()
        w = data['weight'].as_matrix().flatten()
        N = float(x.size)
        m = np.sum(x * w) / np.sum(w)
        s = np.sqrt(np.sum(w * np.square(x - m)) / ((N - 1) * np.sum(w) / N))
        data['uB'] -= m
        data['uB'] /= s * 3.
        data['uB'] += 0.5
        pass


    # Plotting learning process
    # --------------------------------------------------------------------------
    with Profile("Plotting learning process"):

        # Get predictions
        pred_test  = clf.predict_proba(data[data['train'] == 0]) [:,1]
        pred_train = clf.predict_proba(data[data['train'] == 1])[:,1]

        # Get staged predicting, i.e. for each successive estimator
        staged_pred_train = list(clf.staged_predict_proba(data[data['train'] == 1]))
        staged_pred_test  = list(clf.staged_predict_proba(data[data['train'] == 0]))

        y_train = data.loc[data['train'] == 1, 'signal'].as_matrix().flatten()
        y_test  = data.loc[data['train'] == 0, 'signal'].as_matrix().flatten()

        ll_train, ll_test = list(), list()
        auc_train, auc_test = list(), list()
        for idx, (pred_train, pred_test) in enumerate(zip(staged_pred_train, staged_pred_test)):
            p_train = pred_train[:,1]
            p_test  = pred_test [:,1]

            ll_train.append( log_loss(y_train, p_train) )
            ll_test .append( log_loss(y_test,  p_test) )

            auc_train.append( roc_auc_score(y_train, p_train) )
            auc_test .append( roc_auc_score(y_test,  p_test) )
            pass

        x = np.arange(len(ll_train))

        # Log-loss for each boosting iteration
        fig, ax = plt.subplots()
        plt.plot(x, ll_train, label='Train')
        plt.plot(x, ll_test,  label='Test')
        plt.legend()
        plt.xlabel("Boosting step / no. estimator")
        plt.ylabel("Log-loss")
        plt.savefig('temp_logloss.pdf')

        # ROC AUC for each boosting iteration
        fig, ax = plt.subplots()
        plt.plot(x, auc_train, label='Train')
        plt.plot(x, auc_test,  label='Test')
        plt.legend()
        plt.xlabel("Boosting step / no. estimator")
        plt.ylabel("ROC AUC")
        plt.savefig('temp_auc.pdf')
        pass


    # Plotting distributions
    # --------------------------------------------------------------------------
    with Profile("Plotting distributions"):
        bins = np.linspace(0, 1, 100 + 1, endpoint=True)
        fig, ax = plt.subplots()
        plt.hist(data.loc[data['signal'] == 1, 'uB'], bins=bins, weights=data.loc[data['signal'] == 1, 'weight'], alpha=0.5, label='Signal')
        plt.hist(data.loc[data['signal'] == 0, 'uB'], bins=bins, weights=data.loc[data['signal'] == 0, 'weight'], alpha=0.5, label='Background')
        plt.legend()
        plt.xlabel("uBoost classifier variable")
        plt.ylabel("Events")
        plt.savefig('temp_dist.pdf')
        pass


    # Plotting efficiencies
    # --------------------------------------------------------------------------
    with Profile("Plotting efficiencies"):

        bins = np.linspace(40, 300, 2 * 26 + 1, endpoint=True)
        fig, ax = plt.subplots()
        msk_bkg = data['signal'] == 0
        effs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        cuts = [wpercentile(np.array(data.loc[msk_bkg, 'uB']).flatten(), eff, weights=np.array(data.loc[msk_bkg, 'weight']).flatten()) for eff in effs]
        for cut, eff in zip(cuts,effs):
            msk_cut = data['uB'] > cut
            num, _   = np.histogram(data.loc[msk_bkg & msk_cut, 'm'], bins=bins, weights=data.loc[msk_bkg & msk_cut, 'weight'])
            denom, _ = np.histogram(data.loc[msk_bkg,           'm'], bins=bins, weights=data.loc[msk_bkg,           'weight'])
            plt.plot(bins[:-1] + 0.5 * np.diff(bins), num/denom, label='uB < {:.2f} (bkg. eff. = {:.0f}%)'.format(cut, eff))
            pass
        plt.legend()
        plt.xlabel("Large-radius jet mass [GeV]")
        plt.ylabel("Background efficiency")
        plt.savefig('temp_eff.pdf')

        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
