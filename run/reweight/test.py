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
    data       = data[data['train']  == 0]
    signal     = data[data['signal'] == 1]
    background = data[data['signal'] == 0]


    # Flatness reweighting
    # --------------------------------------------------------------------------
    with Profile("flatness"):

        # Get input array and -weights
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        original, original_weight = get_input(background, Scenario.FLATNESS)


        # Loading reweighter from file
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Loading reweighter from file"):
            with gzip.open('models/reweight/reweighter_flatness.pkl.gz', 'r') as f:
                reweighter = pickle.load(f)
                pass
            pass


        # Re-weighting for flat prior
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Re-weighting for flat prior"):
            uniform  = reweighter.predict_weights(original, original_weight=original_weight)
            uniform *= np.sum(background['weight']) / np.sum(uniform)
            pass


        # Adding uniform-weight variable
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Adding uniform-weight variable"):
            uniform_ = data['weight'].as_matrix()
            uniform_[data['signal'] == 0] = uniform
            background['uniform'] = pd.Series(uniform_, index=data.index)
            pass


        # Making test plot(s)
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Making test plot(s)"):

            # Canvas
            fig, ax = plt.subplots()

            # Plot
            bins = np.linspace(np.log(40), np.log(300), 60 + 1, endpoint=True)
            plt.hist(background['logm'], bins=bins, weights=background['weight'],  alpha=0.3, label='Background, original')
            plt.hist(background['logm'], bins=bins, weights=background['uniform'], alpha=0.3, label='Background, reweighted')

            # Decoration(s)
            plt.legend()
            plt.xlabel('log(m)')
            plt.ylabel('Jets')

            # Save
            if args.save:
                mkdir('figures/')
                plt.savefig('figures/temp_reweight_flatness.pdf')
                pass
            pass
        pass


        # Jet mass reweighting
        # --------------------------------------------------------------------------
        with Profile("Mass"):

            # Get input array and -weights
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            sig, _, sig_weight, _ = get_input(data, Scenario.MASS)


            # Loading reweighter from file
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            with Profile("Loading reweighter from file"):
                with gzip.open('models/reweight/reweighter_mass.pkl.gz', 'r') as f:
                    reweighter = pickle.load(f)
                    pass
                pass


            # Re-weighting jet mass spectrum
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            with Profile("Re-weighting for flat prior"):
                jetmass  = reweighter.predict_weights(sig, original_weight=sig_weight)
                jetmass *= np.sum(signal['weight']) / np.sum(jetmass)
                pass


            # Adding jet-mass-weight variable
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            with Profile("Adding jet-mass-weight variable"):
                jetmass_ = data['weight'].as_matrix()
                jetmass_[data['signal'] == 1] = jetmass
                signal['jetmass'] = pd.Series(jetmass_, index=data.index)
                pass


            # Making test plot(s)
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            with Profile("Making test plot(s)"):

                # Jet mass distribution
                # ---------------------
                # Canvas
                fig, ax = plt.subplots()

                # Plot
                bins = np.linspace(40, 300, 60 + 1, endpoint=True)
                plt.hist(background['m'], bins=bins, weights=background['weight'], alpha=0.3, label='Background')
                plt.hist(signal['m'],     bins=bins, weights=signal['weight'],     alpha=0.3, label='Signal, original')
                plt.hist(signal['m'],     bins=bins, weights=signal['jetmass'],    alpha=0.3, label='Signal, reweighted')

                # Decoration(s)
                plt.legend()
                plt.xlabel('Jet mass [GeV]')
                plt.ylabel('Jets')
                plt.yscale('log', nonposy='clip')

                # Save
                if args.save:
                    mkdir('figures/')
                    plt.savefig('figures/temp_reweight_mass_m.pdf')
                    pass


                # Jet pT distribution
                # -------------------
                # Canvas
                fig, ax = plt.subplots()

                # Plot
                bins = np.linspace(200, 2000, 60 + 1, endpoint=True)
                plt.hist(background['pt'], bins=bins, weights=background['weight'], alpha=0.3, label='Background')
                plt.hist(signal['pt'],     bins=bins, weights=signal['weight'],     alpha=0.3, label='Signal, original')
                plt.hist(signal['pt'],     bins=bins, weights=signal['jetmass'],    alpha=0.3, label='Signal, reweighted')

                # Decoration(s)
                plt.legend()
                plt.xlabel('Jet pt [GeV]')
                plt.ylabel('Jets')
                plt.yscale('log', nonposy='clip')

                # Save
                if args.save:
                    mkdir('figures/')
                    plt.savefig('figures/temp_reweight_mass_pt.pdf')
                    pass
                pass

            pass


    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(plots=True)

    # Call main function
    main(args)
    pass
