#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training uBoost classifier for de-correlated jet tagging."""

# Basic import(s)
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
    data = data[data['train'] == 1]

    # Subset
    #data = data.head(1000000)  # @TEMP

    # Config, to be relegated to configuration file
    cfg = {
        'DecisionTreeClassifier': {
            'max_depth': 4,
            'min_samples_split': 2,
            'min_samples_leaf': 1
        },

        'uBoost': {                      # @NOTE: or uBoostClassifier?
            'n_estimators': 100,
            'n_neighbors': 50,

            'target_efficiency': 0.80,
            #'efficiency_steps': 3,      # For uBoostClassifier only

            'smoothing': 0.0,
            'uniforming_rate': 1.,
            'learning_rate': 1.,
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
        with open('models/uboost/uboost_{:d}.pkl'.format(int(opts['uBoost']['target_efficiency'] * 100)), 'w') as f:
            pickle.dump(uboost, f)
            pass

        # Save Adaboost classifier
        with open('models/uboost/adaboost.pkl', 'w') as f:
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
