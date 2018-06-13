#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing CSS transform."""

# Basic import(s)
import gzip
import pickle
from array import array

# Scientific import(s)
import numpy as np
import pandas as pd

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir, latex
from adversarial.profile import profile, Profile
from adversarial.constants import *
from tests.studies.common import TEXT

# Local import(s)
from .common import *
from .train import fit

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, features, _ = load_data(args.input + 'data.h5', train=True, background=True)

    # Add CSS variable
    var = "D2"
    add_css(var, data)

    # Plot D2(CSS) distributions for each mass bin
    plot_distributions(data, var)

    return 0


def plot_distributions (data, var):
    """
    Method for delegating plotting
    """

    h_D2lowmass = None
    bins = D2BINS
    for mass, (mass_down, mass_up) in enumerate(zip(MASS_BINS[:-1], MASS_BINS[1:])):

        # Canvas
        c = rp.canvas(batch=True)

        # Fill histograms
        msk = (data['m'] >= mass_down) & (data['m'] < mass_up)
        h_D2    = c.hist(data.loc[msk, var].values,         bins=bins, weights=data.loc[msk, 'weight_test'].values, display=False)
        h_D2CSS = c.hist(data.loc[msk, var + "CSS"].values, bins=bins, weights=data.loc[msk, 'weight_test'].values, display=False)

        if h_D2lowmass is not None:
            sumChi2, bestOmega, profile_css, profile0rebin = fit(h_D2, 1.0, h_D2lowmass, "%.2f"%mass)
            normalise(profile_css, density=True)
        else:
            profile_css = None
            pass

        h_D2    = kde(h_D2)
        h_D2CSS = kde(h_D2CSS)

        normalise(h_D2,    density=True)
        normalise(h_D2CSS, density=True)

        if h_D2lowmass is None:
            h_D2lowmass = h_D2.Clone('h_lowmass')
            pass

        # Draw histograms
        lowmassbin = "#it{{m}} #in  [{:.1f}, {:.1f}] GeV".format(MASS_BINS[0],    MASS_BINS[1])     .replace('.0', '')
        massbin    = "#it{{m}} #in  [{:.1f}, {:.1f}] GeV".format(MASS_BINS[mass], MASS_BINS[mass+1]).replace('.0', '')
        c.hist(h_D2lowmass, label=latex(var, ROOT=True) + ",    {}".format(lowmassbin),
               linecolor=rp.colours[1], fillcolor=rp.colours[1], alpha=0.5, option='HISTL', legend_option='FL')
        c.hist(h_D2,        label=latex(var, ROOT=True) + ",    {}".format(massbin),
               linecolor=rp.colours[4], linestyle=2, option='HISTL')
        c.hist(h_D2CSS,     label=latex(var + 'CSS', ROOT=True) + ", {}".format(massbin),
               linecolor=rp.colours[3], option='HISTL')

        ''' # Draw reference histogram from fit.
        if profile_css is not None:
            c.hist(profile_css, linecolor=ROOT.kBlack, linestyle=2, label='Transformed hist (CSS)')
            pass
        #'''
            
        # Decorations
        c.xlabel(latex(var, ROOT=True) + ", " + latex(var + 'CSS', ROOT=True))
        c.ylabel("Number of jets p.d.f.")
        c.legend(xmin=0.45, ymax=0.76, width=0.25)
        c.text(["#sqrt{s} = 13 TeV,  Multijets",
                "KDE smoothed"], qualifier=QUALIFIER)
        c.pad()._xaxis().SetTitleOffset(1.3)
        c.pad()._yaxis().SetNdivisions(105)
        c.pad()._primitives[-1].Draw('SAME AXIS')

        # Save
        c.save('figures/css/cssProfile_{}_{}.pdf'.format(var, mass))
        pass

    return

# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
