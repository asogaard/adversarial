#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import numpy as np

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


# Global variable definition(s)
ROOT.gStyle.SetTitleOffset(2.0, 'y')


@showsave
def jsd (data, args, features):
    """
    Perform study of ...

    Saves plot `figures/jsd.pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for ...
    """

    # Define common variables
    msk  = data['signal'] == 0
    effs = np.linspace(0, 100, 10 * 2, endpoint=False)[1:].astype(int)

    # Loop tagger features
    jsd = {feat: [] for feat in features}
    c = rp.canvas(batch=not args.show)
    for feat in features:

        # Define cuts
        cuts = list()
        for eff in effs:
            cut = wpercentile(data.loc[msk, feat].values, eff, weights=data.loc[msk, 'weight_test'].values)
            cuts.append(cut)
            pass

        # Ensure correct direction of cut
        if not signal_low(feat):
            cuts = list(reversed(cuts))
            pass

        # Compute KL divergence for successive cuts
        for cut, eff in zip(cuts, effs):

            # Create ROOT histograms
            msk_pass = data[feat] > cut
            h_pass = c.hist(data.loc[ msk_pass & msk, 'm'].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk, 'weight_test'].values, normalise=True, display=False)
            h_fail = c.hist(data.loc[~msk_pass & msk, 'm'].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk, 'weight_test'].values, normalise=True, display=False)

            # Convert to numpy arrays
            p = root_numpy.hist2array(h_pass)
            f = root_numpy.hist2array(h_fail)

            # Compute Jensen-Shannon divergence
            jsd[feat].append(JSD(p, f, base=2))
            pass
        pass

    # Perform plotting
    c = plot(args, data, effs, jsd, features)

    # Output
    path = 'figures/jsd.pdf'

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, effs, jsd, features = argv

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Plots
    ref = ROOT.TH1F('ref', "", 10, 0., 1.)
    for i in range(ref.GetXaxis().GetNbins()):
        ref.SetBinContent(i + 1, 1)
        pass
    c.hist(ref, linecolor=ROOT.kGray + 2, linewidth=1)

    width = 0.18
    for is_simple in [True, False]:
        for ifeat, feat in enumerate(features):
            if is_simple != signal_low(feat): continue
            ifeat += 3 if ifeat > 3 else 0  # @TEMP
            colour = rp.colours[(ifeat // 2) % len(rp.colours)]
            linestyle   =  1 + (ifeat % 2)
            markerstyle = 20 + (ifeat % 2) * 4
            c.plot(jsd[feat], bins=np.array(effs) / 100., linecolor=colour, markercolor=colour, linestyle=linestyle, markerstyle=markerstyle, label=latex(feat, ROOT=True), option='PL')
            pass

        c.legend(header=("Simple:" if is_simple else "MVA:"),
                 width=width, xmin=0.56 + (width + 0.02) * (is_simple), ymax=0.888)  # ymax=0.782)
        pass

    # Redraw axes
    c.pads()[0]._primitives[0].Draw('AXIS SAME')

    # Decorations
    c.xlabel("Background efficiency #varepsilon_{bkg.}")
    c.ylabel("Mass correlation, JSD")
    c.text([], xmin=0.15, ymax = 0.96, qualifier=QUALIFIER)
    c.text(["#sqrt{s} = 13 TeV,  QCD jets"],
           ymax=0.85, ATLAS=None)

    c.latex("Maximal sculpting", 0.065, 1.2, align=11, textsize=11, textcolor=ROOT.kGray + 2)
    c.xlim(0, 1)
    c.ymin(5E-05)
    c.padding(0.45)
    c.logy()

    return c
