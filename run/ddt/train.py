#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training DDT transform."""

# Basic import(s)
import gzip
import pickle
from array import array

# Scientific import(s)
import ROOT
import numpy as np
from sklearn.linear_model import LinearRegression

# Project import(s)
from adversarial.profile import profile, Profile
from adversarial.new_utils import parse_args, initialise, load_data, mkdir
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
    data, features, _ = load_data(args.input + 'data.h5')
    data = data[(data['train'] == 1) & (data['signal'] == 0)]


    # Adding variable(s)
    # --------------------------------------------------------------------------
    add_variables(data)


    # Filling Tau21 profile
    # --------------------------------------------------------------------------
    profile = fill_profile(data, 'Tau21')


    # Fitting profile
    # --------------------------------------------------------------------------
    with Profile("Fitting profile"):
        fit = ROOT.TF1('fit', 'pol1', *FIT_RANGE)
        profile.Fit('fit', 'RQ0')
        intercept, coef = fit.GetParameter(0), fit.GetParameter(1)

        # Create scikit-learn transform
        ddt = LinearRegression()
        ddt.coef_      = np.array([ coef])
        ddt.intercept_ = np.array([-coef * FIT_RANGE[0]])
        ddt.offset_    = np.array([ coef * FIT_RANGE[0] + intercept])

        print "Fitted function:"
        print "  intercept: {:f}".format(intercept)
        print "  coef:      {:f}".format(coef)
        pass


    # Saving DDT transform
    # --------------------------------------------------------------------------
    with Profile("Saving DDT transform"):

        # Ensure model directory exists
        mkdir('models/ddt/')

        # Save classifier
        with gzip.open('models/ddt/ddt.pkl.gz', 'w') as f:
            pickle.dump(ddt, f)
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
