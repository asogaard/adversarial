#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
# ...

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy

# Scientific import(s)
import math
import numpy as np
from numpy.lib.recfunctions import append_fields
import pandas as pd

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, latex, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *
from .studies.common import *

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, features, _ = load_data(args.input + 'data.h5')

    # Perform selection  @NOTE: For Rel. 20.7 only
    #data = data[(data['m']  >  50) & (data['m']  <  300)]
    #data = data[(data['pt'] > 200) & (data['pt'] < 2000)]

    # Add variables  @NOTE: For Rel. 20.7 only
    #data['rho']    = pd.Series(np.log(np.square(data['m']) / np.square(data['pt'])), index=data.index)
    #data['rhoDDT'] = pd.Series(np.log(np.square(data['m']) / data['pt'] / 1.), index=data.index)

    data['logm'] = pd.Series(np.log(data['m']), index=data.index)

    # Check variable distributions
    axes = {
        'pt':   (45, 200, 2000),
        'm':    (50,  50,  300),
        'rho':  (50,  -8,    0),
        'logm': (50,  np.log(50),  np.log(300)),
    }
    weight = 'weight_train'  # 'weight_test' / 'weight'

    for var in axes:

        # Canvas
        c = rp.canvas(batch=True)

        # Plot
        bins = np.linspace(axes[var][1], axes[var][2], axes[var][0] + 1, endpoint=True)
        for signal in [0,1]:
            msk = data['signal'] == signal
            opts = dict(normalise=True, **HISTSTYLE[signal])
            c.hist(data.loc[msk, var].values, bins=bins, weights=data.loc[msk, weight].values, **opts)
            pass

        # Decorations
        c.legend()
        c.xlabel(latex(var, ROOT=True))
        c.ylabel("Fraction of jets")
        #c.logy()
        c.text(TEXT, qualifier=QUALIFIER)

        # Save
        mkdir('figures/distributions')
        c.save('figures/distributions/incl_{}.pdf'.format(var))
        pass


    # 2D histograms
    msk = data['signal'] == 0
    axisvars = sorted(list(axes))
    for i,varx in enumerate(axisvars):
        for vary in axisvars[i+1:]:
            # Canvas
            c = ROOT.TCanvas()
            c.SetRightMargin(0.20)

            # Create, fill histogram
            h2 = ROOT.TH2F('{}_{}'.format(varx, vary), "", *(axes[varx] + axes[vary]))
            root_numpy.fill_hist(h2, data.loc[msk, [varx, vary]].values, 100. * data.loc[msk, weight].values)

            # Draw
            h2.Draw("COLZ")

            # Decorations
            h2.GetXaxis().SetTitle(latex(varx, ROOT=True))
            h2.GetYaxis().SetTitle(latex(vary, ROOT=True))
            c.SetLogz()

            # Save
            c.SaveAs('figures/distributions/2d_{}_{}.pdf'.format(varx, vary))
            pass
        pass

    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass
