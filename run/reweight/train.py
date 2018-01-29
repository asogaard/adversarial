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
    #data = data.head(100000)  # @TEMP


    # Common definition(s)
    # --------------------------------------------------------------------------
    # @NOTE: This is the crucial point: If the target is flat in (m,pt) the
    # re-weighted background _won't_ be flat in (log m, log pt), and vice
    # versa. It should go without saying, but draw target samples from a
    # uniform prior on the coordinates which are used for the decorrelation.
    decorrelation_variables = ['logm']


    # Adding log(m) variable
    # --------------------------------------------------------------------------
    with Profile("Adding log(m) variable"):
        data['logm'] = pd.Series(np.log(data['m']), index=data.index)
        pass


    # Performing pre-processing of de-correlation coordinates
    # --------------------------------------------------------------------------
    with Profile("Performing pre-processing"):

        # Get number of background events and number of target events (arb.)
        num_targets = data.shape[0]

        # Initialise and fill coordinate original arrays
        original = data[decorrelation_variables].as_matrix()
        original_weight = data['weight'].as_matrix().flatten()

        # Scale coordinates to range [0,1]
        original -= np.min(original, axis=0)
        original /= np.max(original, axis=0)

        # Targets
        target = np.random.rand(num_targets, len(decorrelation_variables))
        pass


    # Fitting reweighter
    # --------------------------------------------------------------------------
    with Profile("Fitting reweighter"):
        reweighter = GBReweighter()  # n_estimators=80, max_depth=7)
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
