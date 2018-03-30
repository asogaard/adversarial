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
    if 'knn' in feat.lower():
        xmin, xmax = -1, 2
    elif 'NN' in feat or 'tau21' in feat.lower() or 'boost' in feat.lower():
        xmin, xmax = 0., 1.
    elif feat == 'D2':
        xmin, xmax = 0, 3.5
    else:
        xmin = wpercentile (data[feat].values,  1, weights=data['weight'].values)
        xmax = wpercentile (data[feat].values, 99, weights=data['weight'].values)
        pass

    bins = np.linspace(xmin, xmax, 50 + 1, endpoint=True)

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Plots
    ROOT.gStyle.SetHatchesLineWidth(3)
    base =  dict(bins=bins, alpha=0.5, normalise=True, linewidth=3)
    c.hist(data.loc[(data['signal'] == 0), feat].values, weights=data.loc[(data['signal'] == 0), 'weight'].values, fillstyle=3445, fillcolor=rp.colours[1], linecolor=rp.colours[1], label="QCD jets",    **base)
    c.hist(data.loc[(data['signal'] == 1), feat].values, weights=data.loc[(data['signal'] == 1), 'weight'].values, fillstyle=3454, fillcolor=rp.colours[5], linecolor=rp.colours[5], label="#it{W} jets", **base)

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

    # Output
    path = 'figures/distribution_{}.pdf'.format(standardise(feat))

    return c, args, path
