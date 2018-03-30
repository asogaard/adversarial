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

    # Style
    ROOT.gStyle.SetHatchesLineWidth(3)
    ROOT.gStyle.SetTitleOffset(1.6, 'y')
    base = dict(bins=bins, alpha=0.5, normalise=True, linewidth=3)
    style = {  # key = signal
        True: {
            'fillcolor': rp.colours[5],
            'linecolor': rp.colours[5],
            'fillstyle': 3454,
            'label': "#it{W} jets",
            },
        False: {
            'fillcolor': rp.colours[1],
            'linecolor': rp.colours[1],
            'fillstyle': 3445,
            'label': "QCD jets",
            }
    }

    # Plots
    for signal in [0, 1]:
        msk = (data['signal'] == signal)
        style[signal].update(base)
        c.hist(data.loc[msk, feat].values, weights=data.loc[msk, 'weight'].values, **style[signal])
        pass

    # Decorations
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
