#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing DDT transform."""

# Basic import(s)
from array import array

# Scientific import(s)
import numpy as np

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir, saveclf
from adversarial.profile import profile, Profile
from adversarial.constants import *

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
    data, features, _ = load_data(args.input + 'data.h5')
    data = data[(data['train'] == 0) & (data['signal'] == 0)]

    # Load transform
    ddt = loadclf('models/ddt/ddt.pkl.gz')

    # Add Tau21DDT variable
    add_ddt(data, 'Tau21')

    # Fill profiles
    profiles = dict()
    for var in ['Tau21', 'Tau21DDT']:
        profiles[var] = fill_profile(data, var)
        pass

    # Convert to graphs
    graphs = dict()
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

    # Plot
    plot(graphs, ddt, arr_x)

    return


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    graphs, ddt, arr_x = argv

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
    mkdir('figures/ddt/')
    c.save('figures/ddt/ddt.pdf')
    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
