#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training uBoost classifier for de-correlated jet tagging."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import numpy as np
from hep_ml.uboost import uBoostBDT, uBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Project import(s)
from adversarial.utils import apply_patch
from adversarial.new_utils import parse_args, initialise, load_data, mkdir
from adversarial.profile import profile, Profile

# Global variable(s)
SEED=21


# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data(args.input + 'data.h5')
    data = data.sample(frac=0.001, random_state=32)  # @TEMP
    data = data[data['train'] == 1]


    # Config, to be relegated to configuration file
    cfg = {
        'DecisionTreeClassifier': {
            'criterion': 'gini',
            'max_depth': 10,             # Optimise
            'min_samples_split': 2,      # Optimise (?)
            'min_samples_leaf': 1        # Optimise (?)
        },

        'uBoost': {                      # @NOTE: or uBoostClassifier?
            'n_estimators': 500,         # Optimise
            'n_neighbors': 50,           # Optimise

            'target_efficiency': 0.80,   # @NOTE: Make ~50% sig. eff.
            #'efficiency_steps': 3,      # For uBoostClassifier only

            'smoothing': 0.0,            # Optimise (?)
            'uniforming_rate': 1.,       # Parametrisation of decorrelation
            'learning_rate': .2,         # Optimise (?)
        }
    }

    # Common options, which shouldn't be put in config file
    opts = {
        'DecisionTreeClassifier': {
            'random_state': SEED,        # For reproducibility
        },
        'uBoost': {
            'uniform_label': 0,          # Want flat _background_ efficiency
            'uniform_features': ['m'],
            'train_features': features,
            'random_state': SEED,        # For reproducibility
            #'n_threads': 16,            # For uBoostClassifier only
        }
    }
    opts = apply_patch(opts, cfg)

    # Arrays
    X = data
    w = np.array(data['weight']).flatten()
    y = np.array(data['signal']).flatten()


    # Fit uBoost classifier
    # --------------------------------------------------------------------------
    with Profile("Fitting uBoost classifier"):

        # @NOTE: There might be an issue with the sample weights, because the
        #        local efficiencies computed using kNN does not seem to take the
        #        sample weights into account.
        #
        #        See:
        #          https://github.com/arogozhnikov/hep_ml/blob/master/hep_ml/uboost.py#L247-L248
        #        and
        #          https://github.com/arogozhnikov/hep_ml/blob/master/hep_ml/metrics_utils.py#L159-L176
        #        with `divided_weights` not set.
        #
        #        `sample_weight` seem to be use only as a starting point for the
        #        boosted, and so not used for the efficiency calculation.
        #
        #        If this is indeed the case, it would be possible to simply
        #        sample MC events by their weight, and use `sample_weight = 1`
        #        for all samples passed to uBoost.
        #
        # @NOTE: I have gotten less sure of the above, so probably no panic.

        # Create base classifier
        base_tree = DecisionTreeClassifier(**opts['DecisionTreeClassifier'])

        # Create uBoost classifier
        uboost = uBoostBDT(base_estimator=base_tree,
                           **opts['uBoost'])

        # Fit uBoost classifier
        uboost.fit(X, y, sample_weight=w)
        pass


    # Fit Adaboost classifier
    # --------------------------------------------------------------------------
    with Profile("Fitting Adaboost classifier"):

        # Create base classifier
        base_tree = DecisionTreeClassifier(**opts['DecisionTreeClassifier'])

        # Create Adaboost classifier
        opts['uBoost']['uniforming_rate'] = 0.  # Disable uniformity boosting
        adaboost = uBoostBDT(base_estimator=base_tree,
                             **opts['uBoost'])

        # Fit Adaboost classifier
        adaboost.fit(X, y, sample_weight=w)
        pass


    # Saving classifiers
    # --------------------------------------------------------------------------
    with Profile("Saving classifiers"):

        # Ensure model directory exists
        mkdir('models/uboost/')

        # Save uBoost classifier
        with gzip.open('models/uboost/uboost_{:d}.pkl.gz'.format(int(opts['uBoost']['target_efficiency'] * 100)), 'w') as f:
            pickle.dump(uboost, f)
            pass

        # Save Adaboost classifier
        with gzip.open('models/uboost/adaboost.pkl.gz', 'w') as f:
            pickle.dump(adaboost, f)
            pass
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
