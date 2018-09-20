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
from adversarial.utils import parse_args, initialise, load_data, mkdir, saveclf
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Local import(s)
from .common import *


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, features, _ = load_data(args.input + 'data.h5', train=True, background=True)

    # Fill Tau21 profile
    profile = fill_profile(data, VAR_TAU21)

    # Fit profile
    fit = ROOT.TF1('fit', 'pol1', *FIT_RANGE)
    profile.Fit('fit', 'RQ0')
    intercept_val, coef_val = fit.GetParameter(0), fit.GetParameter(1)
    intercept_err, coef_err = fit.GetParError(0),  fit.GetParError(1)

    # Create scikit-learn transform
    ddt = LinearRegression()
    ddt.coef_      = np.array([ coef_val])
    ddt.intercept_ = np.array([-coef_val * FIT_RANGE[0]])
    ddt.offset_    = np.array([ coef_val * FIT_RANGE[0] + intercept_val])

    print "Fitted function:"
    print "  intercept: {:7.4f} ± {:7.4f}".format(intercept_val, intercept_err)
    print "  coef:      {:7.4f} ± {:7.4f}".format(coef_val, coef_err)

    # Save DDT transform
    saveclf(ddt, 'models/ddt/ddt.pkl.gz')

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
