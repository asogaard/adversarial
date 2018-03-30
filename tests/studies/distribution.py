#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


@showsave
def distribution (data, args, feat):
    """
    Perform study of substructure variable distributions.

    Saves plot `figures/distribution_[feat].pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        feat: Feature for which to plot signal- and background distributions.
    """

    # Define bins
    xmin = wpercentile (data[feat].values,  1, weights=data['weight'].values)
    xmax = wpercentile (data[feat].values, 99, weights=data['weight'].values)

    snap = 0.5  # Snap to nearest multiple in appropriate direction
    xmin = np.floor(xmin / snap) * snap
    xmax = np.ceil (xmax / snap) * snap

    bins = np.linspace(xmin, xmax, 50 + 1, endpoint=True)

    # Perform plotting
    c = plot(args, data, feat, bins)

    # Output
    path = 'figures/distribution_{}.pdf'.format(standardise(feat))

    return c, args, path


def plot (args, data, feat, bins):
    """
    Method for delegating plotting.
    """

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Plots
    ROOT.gStyle.SetHatchesLineWidth(3)
    base = dict(bins=bins, alpha=0.5, normalise=True, linewidth=3)
    for signal in [0, 1]:
        msk    = (data['signal'] == signal)
        colour = rp.colours[5 if signal else 1]
        opts = dict(fillstyle=3454 if signal else 3445,
                    label="#it{W} jets" if signal else "QCD jets",
                    fillcolor=colour, linecolor=colour)
        opts.update(base)
        c.hist(data.loc[msk, feat].values, weights=data.loc[msk, 'weight'].values, **opts)
        pass

    # Decorations
    ROOT.gStyle.SetTitleOffset(1.6, 'y')
    c.xlabel("Large-#it{R} jet " + latex(feat, ROOT=True))
    c.ylabel("Fraction of jets")
    c.text(["#sqrt{s} = 13 TeV",
            "Testing dataset",
            "Baseline selection",
            ],
        qualifier=QUALIFIER)
    c.ylim(2E-03, 2E+00)
    c.logy()
    c.legend()
    return c
