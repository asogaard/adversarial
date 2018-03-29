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

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data(args.input + 'data.h5')
    #data = data.sample(frac=0.1, random_state=32)   # @TEMP


    # Test classifiers in parallel
    # --------------------------------------------------------------------------
    classifiers = [
        #('Adaboost', 'adaboost'),
        ('uBoost (ur= 0.0, te=0.92)', 'uboost_ur_0p0_te_92'),
        ('uBoost (ur= 0.1, te=0.92)', 'uboost_ur_0p1_te_92'),
        ('uBoost (ur= 0.3, te=0.92)', 'uboost_ur_0p3_te_92'),
        ('uBoost (ur= 1.0, te=0.92)', 'uboost_ur_1p0_te_92'),
        ('uBoost (ur= 3.0, te=0.92)', 'uboost_ur_3p0_te_92'),
        ('uBoost (ur= 5.0, te=0.92)', 'uboost_ur_5p0_te_92'),
        ('uBoost (ur=10.0, te=0.92)', 'uboost_ur_10p0_te_92'),
    ]

    njobs = min(10, len(classifiers))

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


        # Feature importances
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #### with Profile("Feature importances"):
        ####     # @NOTE: From scikit-learn (http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_)
        ####     #          "The importance of a feature is computed as the
        ####     #           (normalized) total reduction of the criterion
        ####     #           brought by that feature. It is also known as the
        ####     #           Gini importance."
        ####     for feature, importance in zip(features, clf.feature_importances_):
        ####         print "{:12s}: {:.4f}".format(feature, importance)
        ####         pass
        ####     pass


        # Adding classifier variable
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Adding classifier variable"):
            data[name] = pd.Series(clf.predict_proba(data)[:,1], index=data.index)
            x = data[name]    .as_matrix().flatten()
            w = data['weight'].as_matrix().flatten()

            # Rescale to have weighted mean and RMS of 1/2 and 1/3, resp.
            N = float(x.size)
            m = np.sum(x * w) / np.sum(w)
            s = np.sqrt(np.sum(w * np.square(x - m)) / ((N - 1) * np.sum(w) / N))
            data[name] -= m
            data[name] /= s * 3.
            data[name] += 0.5
            pass


        # Plotting learning process
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Plotting learning process"):

            # Get predictions
            pred_train = clf.predict_proba(data[data['train'] == 1])[:,1]
            pred_test  = clf.predict_proba(data[data['train'] == 0])[:,1]

            # Get staged predicting, i.e. for each successive estimator
            idx_train = np.where(data['train'] == 1)[0]
            idx_test  = np.where(data['train'] == 0)[0]

            idx_train = np.random.choice(idx_train, int(0.05 * idx_train.size), replace=False)
            idx_test  = np.random.choice(idx_test , int(0.05 * idx_test .size), replace=False)

            staged_pred_train = list(clf.staged_predict_proba(data.loc[idx_train]))
            staged_pred_test  = list(clf.staged_predict_proba(data.loc[idx_test]))

            y_train = data.loc[idx_train, 'signal'].values
            y_test  = data.loc[idx_test,  'signal'].values
            w_train = data.loc[idx_train, 'weight'].values
            w_test  = data.loc[idx_test,  'weight'].values

            #staged_pred_train = list(clf.staged_predict_proba(data[data['train'] == 1]))
            #staged_pred_test  = list(clf.staged_predict_proba(data[data['train'] == 0]))

            #y_train = data.loc[data['train'] == 1, 'signal'].values
            #y_test  = data.loc[data['train'] == 0, 'signal'].values
            #w_train = data.loc[data['train'] == 1, 'weight'].values
            #w_test  = data.loc[data['train'] == 0, 'weight'].values

            ll_train, ll_test = list(), list()
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



        # Plotting ROC curves
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Plotting ROCs"):
            # Get predictions
            pred_test  = clf.predict_proba(data[data['train'] == 0])[:,1]
            pred_train = clf.predict_proba(data[data['train'] == 1])[:,1]

            fpr_test, tpr_test, thresholds_test = roc_curve(y_test,  p_test,  sample_weight=w_test)
            fpr_train, tpr_train, thresholds_train  = roc_curve(y_train,  p_train,  sample_weight=w_train)

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
            plt.hist(data.loc[data['signal'] == 1, name], bins=bins, weights=data.loc[data['signal'] == 1, 'weight'], alpha=0.5, label='Signal')
            plt.hist(data.loc[data['signal'] == 0, name], bins=bins, weights=data.loc[data['signal'] == 0, 'weight'], alpha=0.5, label='Background')
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
            bins = np.linspace(40, 300, 1 * 26 + 1, endpoint=True)
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
                    cut = wpercentile(np.array(data.loc[msk_bkg, name]).flatten(), 100 - eff, weights=np.array(data.loc[msk_bkg, 'weight']).flatten())
                    msk_cut = data[name] > cut

                    # Compute numerator/denominator histograms
                    num, _   = np.histogram(data.loc[msk_bkg & msk_cut, 'm'], bins=bins, weights=data.loc[msk_bkg & msk_cut, 'weight'])
                    denom, _ = np.histogram(data.loc[msk_bkg,           'm'], bins=bins, weights=data.loc[msk_bkg,           'weight'])

                    # Cast
                    num   = num  .astype(np.float)
                    denom = denom.astype(np.float)

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
        pass
    return

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
