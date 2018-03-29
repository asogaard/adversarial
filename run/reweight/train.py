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
    data, features, _ = load_data(args.input + 'data.h5')
    data       = data[data['train']  == 1]
    signal     = data[data['signal'] == 1]
    background = data[data['signal'] == 0]


    # Flatness reweighting
    # --------------------------------------------------------------------------
    with Profile("flatness"):

        # Get input array and -weights
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        original, original_weight = get_input(background, Scenario.FLATNESS)

        # Targets
        target = np.random.rand(*original.shape)


        # Fitting reweighter
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Fitting reweighter"):
            #reweighter = GBReweighter()
            reweighter = BinsReweighter(n_bins=200)
            reweighter.fit(original, target=target, original_weight=original_weight)
            pass


        # Saving reweighter to file
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Saving reweighter to file"):
            mkdir('models/reweight/')
            with gzip.open('models/reweight/reweighter_flatness.pkl.gz', 'w') as f:
                pickle.dump(reweighter, f)
                pass
            pass
        pass


    # Jet mass reweighting
    # --------------------------------------------------------------------------
    with Profile("Mass"):

        # Get input array and -weights
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        sig, bkg, sig_weight, bkg_weight = get_input(data, Scenario.MASS)


        # Fitting reweighter
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Fitting reweighter"):
            #reweighter = GBReweighter(n_estimators=80, max_depth=4)
            reweighter = BinsReweighter(n_bins=200)
            reweighter.fit(sig, target=bkg, original_weight=sig_weight, target_weight=bkg_weight)
            pass


        # Saving reweighter to file
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Saving reweighter to file"):
            mkdir('models/reweight/')
            with gzip.open('models/reweight/reweighter_mass.pkl.gz', 'w') as f:
                pickle.dump(reweighter, f)
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
