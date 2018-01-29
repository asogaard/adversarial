#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing reweighting regressor."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    data = data[(data['train'] == 0) & (data['signal'] == 0)]


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
        N = data.shape[0]

        # Initialise and fill coordinate original arrays
        original = data[decorrelation_variables].as_matrix()
        original_weight = data['weight'].as_matrix().flatten()

        # Scale coordinates to range [0,1]
        original -= np.min(original, axis=0)
        original /= np.max(original, axis=0)
        pass


    # Loading reweighter from file
    # --------------------------------------------------------------------------
    with Profile("Loading reweighter from file"):
        with gzip.open('models/reweight/gbreweighter.pkl.gz', 'r') as f:
            reweighter = pickle.load(f)
            pass
        pass


    # Re-weighting for uniform prior
    # --------------------------------------------------------------------------
    with Profile("Re-weighting for uniform prior"):
        uniform  = reweighter.predict_weights(original, original_weight=original_weight)
        uniform *= np.sum(data['weight']) / np.sum(uniform)
        pass


    # Adding uniform-weight variable
    # --------------------------------------------------------------------------
    with Profile("Adding uniform-weight variable"):
        data['uniform'] = pd.Series(uniform, index=data.index)
        pass


    # Making test plot(s)
    # --------------------------------------------------------------------------
    with Profile("Making test plot(s)"):

        # Canvas
        fig, ax = plt.subplots()

        # Plot
        bins = np.linspace(np.log(40), np.log(300), 60 + 1, endpoint=True)
        plt.hist(data['logm'], bins=bins, weights=data['weight'],  alpha=0.3, label='Background, original')
        plt.hist(data['logm'], bins=bins, weights=data['uniform'], alpha=0.3, label='Background, reweighted')

        # Decoration(s)
        plt.legend()
        plt.xlabel('log(m)')
        plt.ylabel('Jets')

        # Save
        mkdir('figures/')
        plt.savefig('figures/temp_reweight.pdf')
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
