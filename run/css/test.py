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
from adversarial.utils import parse_args, initialise, load_data, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Local import(s)
from .common import *

# Custom import(s)
import rootplotting as rp


# Main function definition


def fill_css (data, jssVar, mass, doApply):
    profile = ROOT.TH1F('profile_{}_{}_{}'.format(jssVar,mass,doApply), "", len(BINS) - 1, BINS)
    jssData = data[jssVar].as_matrix().flatten()
    massData = data['m'].as_matrix().flatten()
    weightData = data['weight'].as_matrix().flatten()
    if doApply:
      AddCSS(jssVar, data)
      jssData = data["%sCSS"%jssVar]

    for cmass,ctau,cweight in zip(massData, tau21Data, weightData):
      if cmass > MASS_BINS[mass] and cmass < MASS_BINS[mass+1]:
        profile.Fill(ctau,cweight)

    return profile

def fill_mass_profile (data, var, doApply):
    """Fill ROOT.TProfile with the average `var` as a function of rhoCSS."""
    profile = ROOT.TH2F('profile_{}'.format(var), "", len(BINS) - 1, BINS, len(MASS_BINS)-1, MASS_BINS)

    c = rp.canvas(batch=True)

    for mass in range(len(MASS_BINS)-1):
      print "Starting mass bin ", mass
      massProf = fill_css(data, var, mass, doApply)
      if massProf.Integral() > 0:
          massProf.Scale(1. / massProf.Integral())
      massProf.Rebin(20)
      c.hist(massProf, label="D_2", linecolor=rp.colours[mass%5], markercolor=rp.colours[mass%5])

      for i in range(massProf.GetNbinsX()):
        profile.SetBinContent(i, mass, massProf.GetBinContent(i))

    mkdir('figures/')
    c.save('figures/cssProfile_{}.pdf'.format(doApply))
    return profile


@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    #data, features, _ = load_data(args.input + 'data.h5')
    data, features, _ = load_data("/afs/cern.ch/work/j/jroloff/adversarial/data.h5")
    data = data[(data['train'] == 0) & (data['signal'] == 0)]


    # Common definition(s)
    # --------------------------------------------------------------------------
    profiles, graphs = dict(), dict()

    # Filling profiles
    # --------------------------------------------------------------------------
    myvar = 'Tau21'
    profiles['{}CSS'.format(myvar)] = fill_mass_profile(data, myvar, True)
    profiles[myvar] = fill_mass_profile(data, myvar, False)

    # Convert to graphs
    # --------------------------------------------------------------------------
    with Profile("Convert to graphs"):

        # Loop profiles
        for key, profile in profiles.iteritems():
            # Create arrays from profile
            arr_x, arr_y, arr_ex, arr_ey = array('d'), array('d'), array('d'), array('d')

            for ibin in range(1, profile.GetYaxis().GetNbins() + 1):
                projection = profile.ProjectionX("%s_py"%profile.GetName(),ibin, ibin+1)
                arr_x .append(profile.GetXaxis().GetBinCenter(ibin))
                arr_y .append(projection.GetMean())
                arr_ex.append(projection.GetBinWidth(ibin) / 2.)
                arr_ey.append(projection.GetBinError  (ibin))

            # Create graph
            graphs[key] = ROOT.TGraphErrors(len(arr_x), arr_x, arr_y, arr_ex, arr_ey)
            pass
        pass


    # Creating figure
    # --------------------------------------------------------------------------
    with Profile("Creating figure"):

        # Canvas
        c = rp.canvas(batch=True)

        # Profiles
        c.graph(graphs[myvar],    label="Original, #tau_{21}",          linecolor=rp.colours[5], markercolor=rp.colours[5])
        c.graph(graphs['{}CSS'.format(myvar)], label="Transformed, #tau_{21}^{CSS}", linecolor=rp.colours[1], markercolor=rp.colours[1], markerstyle=21)

        # Decorations
        c.xlabel("Large-#it{R} jet #rho^{CSS} = log(m^{2}/ p_{T} / 1 GeV)")
        c.ylabel("#LT#tau_{21}#GT, #LT#tau_{21}^{CSS}#GT")
        c.text(["#sqrt{s} = 13 TeV,  QCD jets",
                "Training dataset",
                "Baseline selection",
                ],
               qualifier=QUALIFIER)

        # Save
        mkdir('figures/')
        c.save('figures/css.pdf')
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
