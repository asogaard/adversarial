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

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
            'learning_rate': .3,         # Optimise (?)
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

    #AdaBoost optimization parameters
    #learning_rate = [0.001,0.01,0.1,0.2,0.3]
    #n_estimators = [200, 300, 400, 500, 600]

    #max_depth = [1, 2, 3, 5, 7, 10, 20, 50, 100]           #range used in W/Top tagger analysis
    #min_samples_split = ?
    #min_samples_leaf = [0.5, 1.0, 2.5, 5.0, 10.0, 20.0]    #range used in W/Top tagger analysis

    #param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    #param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)

    # Fit uBoost classifier
    # --------------------------------------------------------------------------
#    with Profile("Fitting uBoost classifier"):
#
#        # @NOTE: There might be an issue with the sample weights, because the
#        #        local efficiencies computed using kNN does not seem to take the
#        #        sample weights into account.
#        #
#        #        See:
#        #          https://github.com/arogozhnikov/hep_ml/blob/master/hep_ml/uboost.py#L247-L248
#        #        and
#        #          https://github.com/arogozhnikov/hep_ml/blob/master/hep_ml/metrics_utils.py#L159-L176
#        #        with `divided_weights` not set.
#        #
#        #        `sample_weight` seem to be use only as a starting point for the
#        #        boosted, and so not used for the efficiency calculation.
#        #
#        #        If this is indeed the case, it would be possible to simply
#        #        sample MC events by their weight, and use `sample_weight = 1`
#        #        for all samples passed to uBoost.
#        #
#        # @NOTE: I have gotten less sure of the above, so probably no panic.
#
#        # Create base classifier
#        base_tree = DecisionTreeClassifier(**opts['DecisionTreeClassifier'])
#
#        # Create uBoost classifier
#        uboost = uBoostBDT(base_estimator=base_tree,
#                           **opts['uBoost'])
#
#        # Fit uBoost classifier
#        uboost.fit(X, y, sample_weight=w)
#        pass


    # Fit Adaboost classifier
    # --------------------------------------------------------------------------
    with Profile("Optimizing Adaboost classifier"):

        # Create base classifier
        base_tree = DecisionTreeClassifier(**opts['DecisionTreeClassifier'])

        # Create Adaboost classifier
        opts['uBoost']['uniforming_rate'] = 0.  # Disable uniformity boosting
        adaboost = uBoostBDT(base_estimator=base_tree,
                             **opts['uBoost'])

	estimators = [('ada', adaboost)]
	pipe = Pipeline(estimators)
	#param_grid = dict(ada__base_estimator__max_depth = [1, 2, 3, 5, 7, 10, 20, 50, 100], ada__base_estimator__min_samples_leaf = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2], ada__learning_rate = [0.01,0.1,0.2,0.3,0.5], ada__n_estimators = [200, 300, 400, 500, 600])
	param_grid = dict(ada__base_estimator__max_depth = [20], ada__base_estimator__min_samples_leaf = [0.005], ada__learning_rate = [0.2], ada__n_estimators = [500])

        #clf = GridSearchCV(adaboost, param_grid,scoring='roc_auc',cv=2,verbose=1, refit=True, n_jobs=4)
        clf = GridSearchCV(pipe, param_grid,scoring='roc_auc',cv=2,verbose=1, refit=True, n_jobs=10)
        #clf.fit(X, y,sample_weight=w)
        clf.fit(X, y,**{'ada__sample_weight': w})

        print("Best parameters set found on development set: ")
        print(clf.best_params_)
	means = clf.cv_results_['mean_test_score']
	stds = clf.cv_results_['std_test_score']
	params = clf.cv_results_['params']

	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))

        pass


    # Saving classifiers
    # --------------------------------------------------------------------------
    with Profile("Saving classifiers"):

        # Ensure model directory exists
        mkdir('models/uboost/')

#        # Save uBoost classifier
#        with gzip.open('models/uboost/uboost_{:d}.pkl.gz'.format(int(opts['uBoost']['target_efficiency'] * 100)), 'w') as f:
#            pickle.dump(uboost, f)
#            pass

        # Save Adaboost classifier
        with gzip.open('models/uboost/adaboost.pkl.gz', 'w') as f:
            #pickle.dump(clf.best_estimator_, f)
            pickle.dump(clf.best_estimator_.named_steps['ada'], f)
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
