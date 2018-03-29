#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training fixed-efficiency kNN regressor."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import roc_curve

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Local import(s)
from .common import *


# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    data, _, _ = load_data(args.input + 'data.h5')
    data = data[(data['train'] == 1)]


    # Adding variable(s)
    # --------------------------------------------------------------------------
    add_variables(data)


    # Compute background efficiency at sig. eff. = 50%
    # --------------------------------------------------------------------------
    eff_sig = 0.5
    fpr, tpr, thresholds = roc_curve(data['signal'], data[VAR], sample_weight=data['weight'])
    idx = np.argmin(np.abs(tpr - eff_sig))
    print "Background acceptance @ {:.2f}% sig. eff.: {:.2f}% ({} < {:.2f})".format(eff_sig * 100., (1 - fpr[idx]) * 100., VAR, thresholds[idx])
    print "Chosen target efficiency: {:.2f}%".format(EFF)


    # Filling profile
    # --------------------------------------------------------------------------
    data = data[data['signal'] == 0]
    profile_meas, (x,y,z) = fill_profile(data)


    # Fit KNN classifier
    # --------------------------------------------------------------------------
    with Profile("Fitting KNN classifier"):
        # Format arrays
        X = np.vstack((x.flatten(), y.flatten())).T
        Y = z.flatten()

        # Remove NaN's
        msk_nan = np.isnan(Y)

        # Fit KNN regressor
        knn = KNeighborsRegressor(weights='distance')
        knn.fit(X[~msk_nan,:], Y[~msk_nan])
        pass


    # Saving KNN classifier
    # --------------------------------------------------------------------------
    with Profile("Saving KNN classifier"):

        # Ensure model directory exists
        mkdir('models/knn/')

        # Save classifier
        with gzip.open('models/knn/knn_{:s}_{:.0f}.pkl.gz'.format(VAR, EFF), 'w') as f:
            pickle.dump(knn, f)
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
