#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training reweighting regressor."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import numpy as np
import pandas as pd
from hep_ml.reweight import BinsReweighter, GBReweighter

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


    # Get input array and -weights
    # --------------------------------------------------------------------------
    original, original_weight = get_input(data)

    # Targets
    num_targets = data.shape[0]  # Number of targets to  use in reweighting (arbitrary)
    target = np.random.rand(num_targets, len(DECORRELATION_VARIABLES))


    # Fitting reweighter
    # --------------------------------------------------------------------------
    with Profile("Fitting reweighter"):
        reweighter = GBReweighter()
        reweighter.fit(original, target=target, original_weight=original_weight)
        pass


    # Saving reweighter to file
    # --------------------------------------------------------------------------
    with Profile("Saving reweighter to file"):
        mkdir('models/reweight/')
        with gzip.open('models/reweight/gbreweighter.pkl.gz', 'w') as f:
            pickle.dump(reweighter, f)
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
