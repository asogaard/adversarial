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

from sklearn.metrics import log_loss

from sklearn.model_selection import check_cv

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
    data, features, _ = load_data(args.input + 'data_1M.h5')
    data = data.sample(frac=0.001, random_state=32)  # @TEMP
    data = data[data['train'] == 1]

    # Reduce size of data
    drop_features = [feat for feat in list(data) if feat not in features + ['m', 'signal', 'weight_train']]
    data.drop(drop_features, axis=1)


    cfg['uBoost']['train_features'] = features
    cfg['uBoost']['random_state'] = SEED
    cfg['DecisionTreeClassifier']['random_state'] = SEED


   # # Arrays
    X = data

    w = np.array(data['weight_train']).flatten()
    y = np.array(data['signal']).flatten()

    Losses = []    

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

        def train_uBoost (Data, train, test, cfg, uniforming_rate):
            """
            ...
            """
             
            X_train = Data.loc[Data.index.get_values()[train]]
            X_val = Data.loc[Data.index.get_values()[test]]
	    
	        w_train = np.array(X_train['weight_train']).flatten()
	        w_val = np.array(X_val['weight_train']).flatten()
    	    
	        y_train = np.array(X_train['signal']).flatten()
    	    y_val = np.array(X_val['signal']).flatten()
	     
            # Create base classifier
            base_tree = DecisionTreeClassifier(**cfg['DecisionTreeClassifier'])

            # Update training configuration
            these_cfg = dict(**cfg['uBoost'])
            these_cfg['uniforming_rate'] = uniforming_rate

            # Create uBoost classifier
            uboost = uBoostBDT(base_estimator=base_tree, **these_cfg)

            # Fit uBoost classifier
            uboost.fit(X_train, y_train, sample_weight=w_train)

            #pred_train = uboost.predict_proba(X_train)[:,1]
            #train_loss = log_loss(y_train, pred_train, sample_weight=w_train)
	    
	        pred_test = uboost.predict_proba(X_val)[:,1]
	        test_loss = log_loss(y_val, pred_test, sample_weight=w_val)

	        Losses.append(test_loss)
		
            return 0
	
	    n_splits = 3 # (default)

	    # declare cv object
	    cv = check_cv(cv=n_splits, y=y, classifier=True)

        uniforming_rate = cfg['uBoost']['uniforming_rate']
        
	    n_jobs = min(7, n_splits)  # ...(10, ...


        jobs = [delayed(train_uBoost, check_pickle=False)(X, train, test, cfg, uniforming_rate) for (train, test) in cv.split(X, y)]

        result = Parallel(n_jobs=n_jobs, backend="threading")(jobs)

	    # compute mean over k-fold losses
	    val_avg = np.mean(Losses)	

	    #print("Losses: ",Losses)
	    print("Avg_loss: {:.4f}".format(val_avg))

	    pass


    return val_avg


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
