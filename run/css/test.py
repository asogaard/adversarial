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
    bins = D2BINS#[::5]
    for mass, (mass_down, mass_up) in enumerate(zip(MASS_BINS[:-1], MASS_BINS[1:])):

        # Canvas
        c = rp.canvas(batch=True)

        # Fill histograms
        msk = (data['m'] >= mass_down) & (data['m'] < mass_up)
        h_D2    = c.hist(data.loc[msk, var].values,         bins=bins, weights=data.loc[msk, 'weight_test'].values, display=False, normalise=True)
        h_D2CSS = c.hist(data.loc[msk, var + "CSS"].values, bins=bins, weights=data.loc[msk, 'weight_test'].values, display=False, normalise=True)

        h_D2    = kde(h_D2)
        h_D2CSS = kde(h_D2CSS)

        h_D2   .Scale(1./h_D2   .GetBinWidth(1))
        h_D2CSS.Scale(1./h_D2CSS.GetBinWidth(1))

        if h_D2lowmass is None:
            h_D2lowmass = h_D2.Clone('h_lowmass')
            pass

        # Draw histograms
        c.hist(h_D2lowmass, label=latex(var, ROOT=True) + ", low-mass bin",
               linecolor=rp.colours[1], fillcolor=rp.colours[1], alpha=0.5, option='HISTL')
        c.hist(h_D2,        label=latex(var, ROOT=True),
               linecolor=rp.colours[4], linestyle=2, option='HISTL')
        c.hist(h_D2CSS,     label=latex(var + 'CSS', ROOT=True),
               linecolor=rp.colours[3], option='HISTL')

        # Decorations
        c.xlabel(latex(var, ROOT=True) + ", " + latex(var + 'CSS', ROOT=True))
        c.ylabel("Number of jets p.d.f.")
        c.legend()
        c.text(["#sqrt{s} = 13 TeV,  QCD jets", "#it{{m}} #in  [{:.1f}, {:.1f}] GeV".format(MASS_BINS[mass], MASS_BINS[mass+1]).replace('.0', ''), "KDE smoothed"], qualifier=QUALIFIER)
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
