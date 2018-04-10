#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training uBoost classifier for de-correlated jet tagging."""

# Basic import(s)
import gzip
import pickle

# Parallelisation import(s)
from joblib import Parallel, delayed

# Scientific import(s)
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from hep_ml.uboost import uBoostBDT, uBoostClassifier

# Project import(s)
from adversarial.utils import apply_patch, parse_args, initialise, load_data, mkdir
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
    #data = data.sample(frac=0.1, random_state=32)  # @TEMP
    data = data[data['train'] == 1]

    # Reduce size of data
    drop_features = [feat for feat in list(data) if feat not in features + ['m', 'signal', 'weight_train']]
    data.drop(drop_features, axis=1)


    # Config, to be relegated to configuration file
    cfg = {
        'DecisionTreeClassifier': {
            'criterion': 'gini',
            'max_depth': 5,             # Optimise
            'min_samples_split': 2,     # Optimise (?)
            'min_samples_leaf': 1       # Optimise (?)
        },

        'uBoost': {                      # @NOTE: or uBoostClassifier?
            'n_estimators': 500,          # Optimise
            'n_neighbors': 50,           # Optimise

            'target_efficiency': 0.92,   # @NOTE: Make ~50% sig. eff.

            'smoothing': 0.0,            # Optimise (?)
            'uniforming_rate': 1.0,      # Parametrisation of decorrelation
            'learning_rate': 1.0,        # Optimise (?)  # Default: 1.
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
        }
    }
    opts = apply_patch(opts, cfg)

    # Arrays
    X = data
    w = np.array(data['weight_train']).flatten()
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

        def train_uBoost (X, y, w, opts, uniforming_rate):
            """
            ...
            """

            # Create base classifier
            base_tree = DecisionTreeClassifier(**opts['DecisionTreeClassifier'])

            # Update training configuration
            these_opts = dict(**opts['uBoost'])
            these_opts['uniforming_rate'] = uniforming_rate

            # Create uBoost classifier
            uboost = uBoostBDT(base_estimator=base_tree, **these_opts)

            # Fit uBoost classifier
            uboost.fit(X, y, sample_weight=w)

            return uboost

        uniforming_rates = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        n_jobs = min(2, len(uniforming_rates))  # ...(10, ...

        jobs = [delayed(train_uBoost, check_pickle=False)(X, y, w, opts, uniforming_rate) for uniforming_rate in uniforming_rates]

        result = Parallel(n_jobs=n_jobs, backend="threading")(jobs)
        pass


    # Saving classifiers
    # --------------------------------------------------------------------------
    for uboost, uniforming_rate in zip(result, uniforming_rates):
        with Profile("Saving classifiers"):

            # Ensure model directory exists
            mkdir('models/uboost/')

            suffix_ur = "ur_{:s}".format(("%.1f" % uniforming_rate).replace('.', 'p'))
            suffix_te = "te_{:d}".format(int(opts['uBoost']['target_efficiency'] * 100))

            # Save uBoost classifier
            with gzip.open('models/uboost/uboost_{}_{}.pkl.gz'.format(suffix_ur, suffix_te), 'w') as f:
                pickle.dump(uboost, f)
                pass
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
