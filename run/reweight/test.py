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
    data = data[(data['train'] == 0) & (data['signal'] == 0)]


    # Get input array and -weights
    # --------------------------------------------------------------------------
    original, original_weight = get_input(data)


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
