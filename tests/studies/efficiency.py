#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import latex, wpercentile
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


@showsave
def efficiency (data, args, feat):
    """
    Perform study of background efficiency vs. mass for different inclusive
    efficiency cuts

    Saves plot `figures/efficiency_[feat].pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        feat: Feature for which to study efficiencies
    """

    # Define common variables
    msk  = data['signal'] == 0
    effs = np.linspace(0, 100, 10, endpoint=False)[1:].astype(int)

    # Define cuts
    cuts = list()
    for eff in effs:
        cut = wpercentile(data.loc[msk, feat].values, eff, weights=data.loc[msk, 'weight'].values)
        cuts.append(cut)
        pass

    # Ensure correct direction of cut
    if not signal_low(feat):
        cuts = list(reversed(cuts))
        pass

    # Compute cut efficiency vs. mass
    profiles = list()
    for cut, eff in zip(cuts, effs):

        # Get correct pass-cut mask
        msk_pass = data[feat] > cut
        if signal_low(feat):
            msk_pass = ~msk_pass
            pass

        # Fill efficiency profile
        profile = ROOT.TProfile('profile_{}_{}'.format(feat, cut), "",
                                len(MASSBINS) - 1, MASSBINS)

        M = np.vstack((data.loc[msk, 'm'].values, msk_pass[msk])).T
        weights = data.loc[msk, 'weight'].values

        root_numpy.fill_profile(profile, M, weights=weights)

        # Add to list
        profiles.append(profile)
        pass

    # Force style
    ref_titleoffsety = ROOT.gStyle.GetTitleOffset('y')
    ROOT.gStyle.SetTitleOffset(1.6, 'y')

    # Perform plotting
    c = plot(args, data, feat, profiles, cuts, effs)

    # Output
    path = 'figures/efficiency_{}.pdf'.format(standardise(feat))

    # Reset style
    ROOT.gStyle.SetTitleOffset(ref_titleoffsety, 'y')

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, feat, profiles, cuts, effs = argv

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Plots
    for idx, (profile, cut, eff) in enumerate(zip(profiles, cuts, effs)):
        colour = rp.colours[1]
        linestyle = 1
        c.hist(profile, linecolor=colour, linestyle=linestyle, option='HIST L')
        c.hist(profile, fillcolor=colour, alpha=0.3, option='E3')
        pass

    # Text
    for idx, (profile, cut, eff) in enumerate(zip(profiles, cuts, effs)):
        if int(eff) in [10, 50, 90]:
            c.latex('#bar{#varepsilon}_{bkg.} = %d%%' % eff,
                    260., profile.GetBinContent(np.argmin(np.abs(MASSBINS - 270.)) + 1) + 0.025,
                    textsize=13,
                    textcolor=ROOT.kGray + 2, align=11)
            pass
        pass

    # Decorations
    c.xlabel("Large-#it{R} jet mass [GeV]")
    c.ylabel("Background efficiency, #varepsilon_{bkg.}")
    c.text(["#sqrt{s} = 13 TeV,  QCD jets",
            "Testing dataset",
            "Baseline selection",
            "Sequential cuts on {}".format(latex(feat, ROOT=True)),
            ],
           qualifier=QUALIFIER)
    c.ylim(0, 1.9)

    return c
