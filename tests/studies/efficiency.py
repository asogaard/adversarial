#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import latex, wpercentile, signal_low, MASSBINS
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
    effs = [5, 10, 20, 40, 80]

    # Define cuts
    cuts = list()
    for eff in effs:
        cut = wpercentile(data.loc[msk, feat].values, eff if signal_low(feat) else 100 - eff, weights=data.loc[msk, 'weight_test'].values)
        cuts.append(cut)
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
        weights = data.loc[msk, 'weight_test'].values

        root_numpy.fill_profile(profile, M, weights=weights)

        # Add to list
        profiles.append(profile)
        pass

    # Perform plotting
    c = plot(args, data, feat, profiles, cuts, effs)

    # Output
    path = 'figures/efficiency_{}.pdf'.format(standardise(feat))

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, feat, profiles, cuts, effs = argv

    with TemporaryStyle() as style:

        # Style
        style.SetTitleOffset(1.6, 'y')

        # Canvas
        c = rp.canvas(batch=not args.show)

        # Plots
        for idx, (profile, cut, eff) in enumerate(zip(profiles, cuts, effs)):
            colour = rp.colours[idx + 0]
            linestyle = 1
            c.hist(profile, linecolor=colour, linestyle=linestyle, option='HIST L')
            c.hist(profile, linecolor=colour, fillcolor=colour, alpha=0.3, option='E3', label=(" " if eff < 10 else "") + "{:d}%".format(eff))
            pass

        # Decorations
        c.xlabel("Large-#it{R} jet mass [GeV]")
        c.ylabel("Background efficiency, #varepsilon_{bkg}^{rel}")
        c.text(["#sqrt{s} = 13 TeV,  Multijets",
                #"#it{W} jet tagging",
                "Cuts on {}".format(latex(feat, ROOT=True)),
                ],
               qualifier=QUALIFIER)
        c.ylim(0, 2.0)
        c.legend(reverse=True, width=0.25, ymax=0.87, header="Incl. #bar{#varepsilon}_{bkg}^{rel}:")
        pass

    return c
