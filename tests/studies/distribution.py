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


# Global variable definition(s)
HISTSTYLE[True] ['label'] = "#it{W} jets"
HISTSTYLE[False]['label'] = "QCD jets"


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
    xmin = wpercentile (data[feat].values,  1, weights=data['weight_test'].values)
    xmax = wpercentile (data[feat].values, 99, weights=data['weight_test'].values)

    snap = 0.5  # Snap to nearest multiple in appropriate direction
    xmin = np.floor(xmin / snap) * snap
    xmax = np.ceil (xmax / snap) * snap

    bins = np.linspace(xmin, xmax, 50 + 1, endpoint=True)

    # Perform plotting
    c = plot(args, data, feat, bins)

    # Output
    path = 'figures/distribution_{}.pdf'.format(standardise(feat))

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, feat, bins = argv

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Style
    base = dict(bins=bins, alpha=0.5, normalise=True, linewidth=3)

    # Plots
    for signal in [0, 1]:
        msk = (data['signal'] == signal)
        HISTSTYLE[signal].update(base)
        c.hist(data.loc[msk, feat].values, weights=data.loc[msk, 'weight_test'].values, **HISTSTYLE[signal])
        pass

    # Decorations
    c.xlabel("Large-#it{R} jet " + latex(feat, ROOT=True))
    c.ylabel("Fraction of jets")
    c.text(TEXT + ["#it{W} jet tagging"], qualifier=QUALIFIER)
    c.ylim(4E-03, 4E-01)
    c.logy()
    c.legend()
    return c
