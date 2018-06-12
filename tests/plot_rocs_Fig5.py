#!/usr/bin/env python

import numpy as np
rng = np.random.RandomState(21)  # For reproducibility
import pandas as pd

import matplotlib.pyplot as plt

import gzip
import pickle

from sklearn.metrics import log_loss, roc_auc_score, roc_curve


def load_data (path, name='dataset', train=None, test=None, signal=None, background=None, sample=None, seed=21):
    """
    General script to load data, common to all run scripts.

    Arguments:
        path: The path to the HDF5 file, from which data should be loaded.
        name: Name of the dataset, as stored in the HDF5 file.
        ...

    Returns:
        Tuple of pandas.DataFrame containing the loaded; list of loaded features
        to be used for training; and list of features to be used for mass-
        decorrelation.

    Raises:
        IOError: If no HDF5 file exists at the specified `path`.
        KeyError: If the HDF5 does not contained a dataset named `name`.
        KeyError: If any of the necessary features are not present in the loaded
            dataset.
    """

    # Check(s)
    assert False not in [train, test, signal, background]
    if sample:                           assert 0 < sample and sample < 1.
    if None not in [train, test]:        assert train != test
    if None not in [signal, background]: assert signal != background

    # Read data from HDF5 file
    data = pd.read_hdf(path, name)

    # Define feature collections to use
    features_input         = ['Tau21', 'C2', 'D2', 'Angularity', 'Aplanarity', 'FoxWolfram20', 'KtDR', 'PlanarFlow', 'Split12', 'ZCut12']
    features_decorrelation = ['m']

    # Split data
    if train:      data = data[data['train']  == 1]
    if test:       data = data[data['train']  == 0]
    if signal:     data = data[data['signal'] == 1]
    if background: data = data[data['signal'] == 0]
    if sample:     data = data.sample(frac=sample, random_state=seed)

    # Return
    return data, features_input, features_decorrelation


def OpenFile (fi,opt):

    with gzip.open(fi, opt) as f:
        clf = pickle.load(f)
        pass

    return clf

# Main function definition
def main ():

    # Initialising
    # --------------------------------------------------------------------------
    #args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data('../ANN_rel21/adversarial/inputs/data_1M.h5') #hardcoded path to input file
    #data = data.sample(frac=0.5, random_state=32)   # @TEMP

    clf = OpenFile('../ANN_rel21/adversarial/models/uboost/uboost_ur_1p00_te_92.pkl.gz', 'r')
    clf_ada = OpenFile('../ANN_rel21/adversarial/models/uboost/uboost_ur_0p00_te_92.pkl.gz', 'r')



    idx_train = np.where(data['train'] == 1)[0]
    idx_test  = np.where(data['train'] == 0)[0]

    idx_train = np.random.choice(idx_train, int(0.05 * idx_train.size), replace=False)
    idx_test  = np.random.choice(idx_test , int(0.05 * idx_test .size), replace=False)



    # Get predictions
   
    staged_pred_train = list(clf.staged_predict_proba(data.loc[data.index.get_values()[idx_train]]))
    staged_pred_test  = list(clf.staged_predict_proba(data.loc[data.index.get_values()[idx_test]]))
    staged_pred_train_ada = list(clf_ada.staged_predict_proba(data.loc[data.index.get_values()[idx_train]]))
    staged_pred_test_ada  = list(clf_ada.staged_predict_proba(data.loc[data.index.get_values()[idx_test]]))


    y_train = data.loc[data.index.get_values()[idx_train],'signal']
    y_test  = data.loc[data.index.get_values()[idx_test],'signal']
    w_train = data.loc[data.index.get_values()[idx_train],'weight_train']
    w_test = data.loc[data.index.get_values()[idx_test],'weight_test']

    for idx, (pred_train, pred_test, pred_train_ada, pred_test_ada) in enumerate(zip(staged_pred_train, staged_pred_test, staged_pred_train_ada, staged_pred_test_ada)):
	    p_train = pred_train[:,1]
	    p_test = pred_test[:,1]
	    p_train_ada = pred_train_ada[:,1]
	    p_test_ada = pred_test_ada[:,1]
	    pass

    fpr_test, tpr_test, thresholds_test = roc_curve(y_test,  p_test,  sample_weight=w_test)
    fpr_train, tpr_train, thresholds_train  = roc_curve(y_train,  p_train,  sample_weight=w_train)
    
    fpr_test_ada, tpr_test_ada, thresholds_test_ada = roc_curve(y_test,  p_test_ada,  sample_weight=w_test)
    fpr_train_ada, tpr_train_ada, thresholds_train_ada  = roc_curve(y_train,  p_train_ada,  sample_weight=w_train)



    # TPR vs FPR
    fig, ax = plt.subplots()
    plt.plot(fpr_train, tpr_train, label='Train uBoost, ur=1')
    plt.plot(fpr_test, tpr_test,  label='Test uBoost, ur=1')
    plt.plot(fpr_train_ada, tpr_train_ada, linestyle='--',label='Train adaBoost')
    plt.plot(fpr_test_ada, tpr_test_ada, linestyle='--', label='Test adaBoost')
    plt.legend()
    plt.xlabel("Fake positive rate")
    plt.ylabel("True positive rate")

    # Save
    plt.savefig('figures/temp_roc.pdf')

    plt.close()


    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    #args = parse_args()

    # Call main function
    main()
    pass

