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

    # Definitions
    histstyle = dict(**HISTSTYLE)

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, features, _ = load_data(args.input + 'data.h5', background=True, train=True)

    pt_bins = np.linspace(200, 2000, 18 + 1, endpoint=True)
    pt_bins = zip(pt_bins[:-1], pt_bins[1:])
    bins = np.linspace(50, 300, (300 - 50) // 10 + 1, endpoint=True)

    for pt_bin in pt_bins:

        histstyle[True] ['label'] = 'Inclusive'
        histstyle[False]['label'] = 'p_{{T}} #in  [{:.0f}, {:.0f}] GeV'.format(*pt_bin)

        # Canvas
        c = rp.canvas(batch=True)

        # Plots
        msk = (data['pt'] > pt_bin[0]) & (data['pt'] < pt_bin[1])
        c.hist(data['m'].values,      bins=bins, weight=data['weight_adv'] .values,      normalise=True, **histstyle[True])
        c.hist(data['m'].values[msk], bins=bins, weight=data['weight_adv'] .values[msk], normalise=True, **histstyle[False])
        c.hist(data['m'].values[msk], bins=bins, weight=data['weight_test'].values[msk], normalise=True, label="Testing weight", linewidth=2, linecolor=ROOT.kGreen)

        # Decorations
        c.legend()
        c.xlabel("Large-#it{R} jet mass [GeV]")
        c.ylabel("Fraction of jets")

        # Save
        c.save('figures/temp_mass_pT{:.0f}_{:.0f}.pdf'.format(*pt_bin))
        pass

    return


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
    weight = 'weight_adv'  # 'weight_test' / 'weight'
    pt_range = (200., 2000.)
    msk_pt = (data['pt'] > pt_range[0]) & (data['pt'] < pt_range[1])
    for var in axes:

        # Canvas
        c = rp.canvas(num_pads=2, batch=True)

        # Plot
        bins = np.linspace(axes[var][1], axes[var][2], axes[var][0] + 1, endpoint=True)
        for adv in [0,1]:
            msk  = data['signal'] == 0   # @TEMP signal
            msk &= msk_pt
            opts = dict(normalise=True, **HISTSTYLE[adv])  # @TEMP signal
            opts['label'] = 'adv' if adv else 'test'
            if adv:
                h1 = c.hist(data.loc[msk, var].values, bins=bins, weights=data.loc[msk, weight].values, **opts)
            else:
                h2 = c.hist(data.loc[msk, var].values, bins=bins, weights=data.loc[msk, 'weight_test'].values, **opts)
                pass
            pass

        # Ratio
        c.pads()[1].ylim(0,2)
        c.ratio_plot((h1,h2), oob=True)

        # Decorations
        c.legend()
        c.xlabel(latex(var, ROOT=True))
        c.ylabel("Fraction of jets")
        c.pads()[1].ylabel("adv/test")
        #c.logy()
        c.text(TEXT + ['p_{{T}} #in  [{:.0f}, {:.0f}] GeV'.format(pt_range[0], pt_range[1])], qualifier=QUALIFIER)

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
