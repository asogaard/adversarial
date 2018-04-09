#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing DDT transform."""

# Basic import(s)
import gzip
import pickle
from array import array

# Scientific import(s)
import numpy as np
import pandas as pd

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Local import(s)
from .common import *

# Custom import(s)
import rootplotting as rp


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
    profiles, graphs = dict(), dict()


    # Adding variable(s)
    # --------------------------------------------------------------------------
    add_variables(data)


    # Loading DDT transform
    # --------------------------------------------------------------------------
    with Profile("Loading DDT transform"):
        with gzip.open('models/ddt/ddt.pkl.gz', 'r') as f:
            ddt = pickle.load(f)
            pass
        pass


    # Adding Tau21DDT variable
    # --------------------------------------------------------------------------
    with Profile("Adding Tau21DDT variable"):
        data['Tau21DDT'] = pd.Series(data['Tau21'] - ddt.predict(data['rhoDDT'].as_matrix().reshape((-1,1))), index=data.index)
        pass


    # Filling profiles
    # --------------------------------------------------------------------------
    for var in ['Tau21', 'Tau21DDT']:
        profiles[var] = fill_profile(data, var)
        pass


    # Convert to graphs
    # --------------------------------------------------------------------------
    with Profile("Convert to graphs"):

        # Loop profiles
        for key, profile in profiles.iteritems():

            # Create arrays from profile
            arr_x, arr_y, arr_ex, arr_ey = array('d'), array('d'), array('d'), array('d')
            for ibin in range(1, profile.GetXaxis().GetNbins() + 1):
                if profile.GetBinContent(ibin) != 0. or profile.GetBinError(ibin) != 0.:
                    arr_x .append(profile.GetBinCenter (ibin))
                    arr_y .append(profile.GetBinContent(ibin))
                    arr_ex.append(profile.GetBinWidth  (ibin) / 2.)
                    arr_ey.append(profile.GetBinError  (ibin))
                    pass
                pass

            # Create graph
            graphs[key] = ROOT.TGraphErrors(len(arr_x), arr_x, arr_y, arr_ex, arr_ey)
            pass
        pass


    # Creating figure
    # --------------------------------------------------------------------------
    with Profile("Creating figure"):

        # Style
        ROOT.gStyle.SetTitleOffset(1.4, 'x')

        # Canvas
        c = rp.canvas(batch=True)

        # Setup
        pad = c.pads()[0]._bare()
        pad.cd()
        pad.SetTopMargin(0.10)
        pad.SetTopMargin(0.10)

        # Profiles
        c.graph(graphs['Tau21'],    label="Original, #tau_{21}",          linecolor=rp.colours[5], markercolor=rp.colours[5], markerstyle=24, legend_option='PE')
        c.graph(graphs['Tau21DDT'], label="Transformed, #tau_{21}^{DDT}", linecolor=rp.colours[1], markercolor=rp.colours[1], markerstyle=20, legend_option='PE')

        # Fit
        x1, x2 = min(arr_x), max(arr_x)
        intercept, coef = ddt.intercept_ + ddt.offset_, ddt.coef_
        y1 = intercept + x1 * coef
        y2 = intercept + x2 * coef
        c.plot([y1,y2], bins=[x1,x2], color=rp.colours[-1], label='Linear fit', linewidth=1, linestyle=1, option='L')

        # Decorations
        c.xlabel("Large-#it{R} jet #rho^{DDT} = log(m^{2}/ p_{T} / 1 GeV)")
        c.ylabel("#LT#tau_{21}#GT, #LT#tau_{21}^{DDT}#GT")

        lines = ["#sqrt{s} = 13 TeV,  QCD jets"]
        c.text(["#sqrt{s} = 13 TeV,  QCD jets"], qualifier=QUALIFIER)
        c.legend(width=0.25, xmin=0.57, ymax=None if "Internal" in QUALIFIER else 0.85)
        
        c.ylim(0, 1.4)
        c.latex("Fit range", sum(FIT_RANGE) / 2., 0.08, textsize=13, textcolor=ROOT.kGray + 2)
        c.xline(FIT_RANGE[0], ymax=0.82, text_align='BR', linecolor=ROOT.kGray + 2)
        c.xline(FIT_RANGE[1], ymax=0.82, text_align='BL', linecolor=ROOT.kGray + 2)

        # Save
        mkdir('figures/')
        c.save('figures/ddt.pdf')
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
