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
    data, features, _ = load_data(args.input + 'data_1M_10M.h5')
    #data = data.sample(frac=0.5, random_state=32)  # @TEMP
    data = data[data['train'] == 1]

    # Reduce size of data
    drop_features = [feat for feat in list(data) if feat not in features + ['m', 'signal', 'weight_adv']]
    data.drop(drop_features, axis=1)


    cfg['uBoost']['train_features'] = features
    cfg['uBoost']['random_state'] = SEED
    cfg['DecisionTreeClassifier']['random_state'] = SEED


    # Arrays
    X = data

    #print(X.head())

    w = np.array(data['weight_adv']).flatten()
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

        def train_uBoost (X, y, w, cfg, uniforming_rate):
            """
            ...
            """

            # Create base classifier
            base_tree = DecisionTreeClassifier(**cfg['DecisionTreeClassifier'])

            # Update training configuration
            these_cfg = dict(**cfg['uBoost'])
            these_cfg['uniforming_rate'] = uniforming_rate

            # Create uBoost classifier
            uboost = uBoostBDT(base_estimator=base_tree, **these_cfg)

            # Fit uBoost classifier
            uboost.fit(X, y, sample_weight=w)

            return uboost

        #uniforming_rates = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
        uniforming_rates = [0.0, 0.01, 0.1, 0.3, 0.5, 1.0]
        #uniforming_rates = [0.5, 1.0]
        n_jobs = min(7, len(uniforming_rates))  # ...(10, ...

        jobs = [delayed(train_uBoost, check_pickle=False)(X, y, w, cfg, uniforming_rate) for uniforming_rate in uniforming_rates]

        result = Parallel(n_jobs=n_jobs, backend="threading")(jobs)
        pass


    # Saving classifiers
    # --------------------------------------------------------------------------
    for uboost, uniforming_rate in zip(result, uniforming_rates):
        with Profile("Saving classifiers"):

            # Ensure model directory exists
            mkdir('models/uboost/')

            suffix_ur = "ur_{:s}".format(("%.2f" % uniforming_rate).replace('.', 'p'))
            suffix_te = "te_{:d}".format(int(cfg['uBoost']['target_efficiency'] * 100))

            # Save uBoost classifier
            with gzip.open('models/uboost/uboost_{}_{}_rel21_fixed_def_cfg_1000boost.pkl.gz'.format(suffix_ur, suffix_te), 'w') as f:
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
