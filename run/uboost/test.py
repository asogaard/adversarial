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

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, features, _ = load_data(args.input + 'data.h5', sample=0.01)  # @TEMP

    # Define classifier configuration(s)
    pattern = 'uboost_ur_{:4.2f}_te_92_rel21_fixed'
    urs = sorted([0.0, 0.01, 0.1, 0.3])
    classifiers = [('AdaBoost' if ur == 0 else 'uBoost (#alpha={:4.2f})'.format(ur), pattern.format(ur).replace('.', 'p')) for ur in urs]

    # Compute classifiers variables in parallel
    njobs = min(7, len(classifiers))
    with Profile("Run tests in parallel"):
        ret = Parallel(n_jobs=njobs)(delayed(compute)(data, name) for _, name in classifiers)
        pass

    # Add classifier variables to data
    for name, staged_series in ret:
        for stage, series in enumerate(staged_series):
            data['{:s}__{:d}'.format(name, stage)] = series
            pass
        pass

    # Plot learning curves
    plot(data, urs, classifiers)

    return 0


def compute (data, name):
    """
    Common method to compute uBoost/Adaboost classifier variables.
    """

    # Load classifier
    with gzip.open('models/uboost/{}.pkl.gz'.format(name), 'r') as f:
        clf = pickle.load(f)
        pass

    # Add classifier variable
    return (name, [pd.Series(series[:,1], index=data.index) for series in clf.staged_predict_proba(data)])


def plot (data, urs, classifiers):
    """
    Common method to perform tests on named uBoost/Adaboost classifier.
    """

    # Plotting learning process
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    with Profile("Plotting learning process"):

        for alpha, (title, name) in zip(urs, classifiers):
            if title is 'AdaBoost': continue
            print "===", name, title

            # Get training/test split masks
            msk_train = data['train'] == 1
            msk_test  = data['train'] == 0

            # Get target and weight arrays
            y_train = data.loc[msk_train, 'signal']    .values.flatten()
            y_test  = data.loc[msk_test,  'signal']    .values.flatten()
            w_train = data.loc[msk_train, 'weight_adv'].values.flatten()
            w_test  = data.loc[msk_test,  'weight_adv'].values.flatten()

            # Compute log-loss for each epoch
            ll_ab_train, ll_ab_test = list(), list()
            ll_ub_train, ll_ub_test = list(), list()

            nb_epochs = len(filter(lambda col: col.startswith(name), data.columns))
            x = np.arange(nb_epochs)

            for epoch in range(nb_epochs):

                # -- Get column names for current epoch
                col_ab = '{:s}__{:d}'.format(classifiers[0][1], epoch)  # Assuming `AdaBoost` is first classifier
                col_ub = '{:s}__{:d}'.format(name, epoch)

                # -- Get classifier variables for current epoch
                p_ab_train = data.loc[msk_train, col_ab]
                p_ab_test  = data.loc[msk_test,  col_ab]
                p_ub_train = data.loc[msk_train, col_ub]
                p_ub_test  = data.loc[msk_test,  col_ub]

                # -- Compute log-loss for current epoch
                ll_ab_train.append( log_loss(y_train, p_ab_train, sample_weight=w_train) )
                ll_ab_test .append( log_loss(y_test,  p_ab_test,  sample_weight=w_test) )
                ll_ub_train.append( log_loss(y_train, p_ub_train, sample_weight=w_train) )
                ll_ub_test .append( log_loss(y_test,  p_ub_test,  sample_weight=w_test) )
                pass

            # Plot log-loss curves
            c = rp.canvas(batch=True)

            # -- Common plotting options
            opts = dict(linewidth=2, legend_option='L')
            c.graph(ll_ab_train, bins=x, linecolor=rp.colours[5], linestyle=1, option='AL', label='AdaBoost', **opts)
            c.graph(ll_ab_test,  bins=x, linecolor=rp.colours[5], linestyle=2, option='L',                    **opts)
            c.graph(ll_ub_train, bins=x, linecolor=rp.colours[1], linestyle=1, option='L',  label='uBoost',   **opts)
            c.graph(ll_ub_test,  bins=x, linecolor=rp.colours[1], linestyle=2, option='L',                    **opts)

            # -- Decorations
            c.pad()._yaxis().SetNdivisions(505)
            c.xlabel("Training epoch")
            c.ylabel("BDT classifier loss")
            c.xlim(0, len(x))
            c.ylim(0.3, 1.4)
            c.legend(width=0.28)
            c.legend(header='Dataset:',
                     categories=[('Training', {'linestyle': 1}),
                                 ('Testing',  {'linestyle': 2})],
                     width=0.28, ymax=0.69)

            for leg in c.pad()._legends:
                leg.SetFillStyle(0)
                pass

            c.text(["#sqrt{s} = 13 TeV",
                    "#it{W} jet tagging",
                    "Uniforming rate #alpha = {:3.1f}".format(alpha)],
                   qualifier="Simulation Internal")

            # -- Save
            c.save('figures/loss_uboost__alpha{:4.2f}'.format(alpha).replace('.', 'p') + '.pdf')

            pass
        pass

    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
