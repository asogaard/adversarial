#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

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
def roc (data, args, features, masscut=False):
    """
    Perform study of ...

    Saves plot `figures/roc.pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for ...
        masscut: ...
    """

    # Computing ROC curves
    ROCs = dict()
    for feat in features:

        sign = -1. if signal_low(feat) else 1.

        eff_bkg, eff_sig, thresholds = roc_curve(data['signal'].values,
                                                 data[feat]    .values * sign,
                                                 sample_weight=data['weight_test'].values)

        # Filter, to advoid background rejection blowing up
        indices = np.where((eff_bkg > 0) & (eff_sig > 0))
        eff_sig = eff_sig[indices]
        eff_bkg = eff_bkg[indices]

        # Subsample to 1% steps
        targets = np.linspace(0, 1, 100 + 1, endpoint=True)
        indices = np.array([np.argmin(np.abs(eff_sig - t)) for t in targets])
        eff_sig = eff_sig[indices]
        eff_bkg = eff_bkg[indices]

        # Store
        ROCs[feat] = (eff_sig, eff_bkg)
        pass

    # Computing ROC AUCs
    AUCs = dict()
    for feat in features:
        sign = -1. if signal_low(feat) else 1.
        AUCs[feat] = roc_auc_score(data['signal'].values,
                                   data[feat]    .values * sign,
                                   sample_weight=data['weight_test'].values)
        pass

    # Perform plotting
    c = plot(args, data, features, ROCs, AUCs, masscut)

    # Output
    path = 'figures/roc.pdf'

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, features, ROCs, AUCs, masscut = argv

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Plots
    # -- Random guessing
    bins = np.linspace(0, 1., 100 + 1, endpoint=True)
    bins = bins[1:-1]
    c.graph(np.power(bins, -1.), bins=bins, linecolor=ROOT.kGray + 2, linewidth=1, option='AL')

    # -- AUCs
    categories = list()
    for feat in features:
         line = "#scale[0.6]{#color[13]{AUC: %.3f}}" % AUCs[feat]
         categories += [(line, {'linestyle': 0, 'fillstyle': 0, 'markerstyle': 0, 'option': ''})]
         pass
    c.legend(categories=categories, xmin=0.80, width=0.04)

    # -- ROCs
    for ifeat, feat in enumerate(features):
        eff_sig, eff_bkg = ROCs[feat]
        c.graph(np.power(eff_bkg, -1.), bins=eff_sig, linestyle=1 + (ifeat % 2), linecolor=rp.colours[(ifeat // 2) % len(rp.colours)], linewidth=2, label=latex(feat, ROOT=True), option='L')
        pass
    c.legend(xmin=0.58, width=0.22)

    # Decorations
    c.xlabel("Signal efficiency #varepsilon_{sig}")
    c.ylabel("Background rejection 1/#varepsilon_{bkg}")
    c.text(["#sqrt{s} = 13 TeV",
            "#it{W} jet tagging"] + \
            (["m #in  [60, 100] GeV"] if masscut else []),
           qualifier=QUALIFIER)

    c.latex("Random guessing", 0.3, 1./0.3 * 0.9, align=23, angle=-12, textsize=13, textcolor=ROOT.kGray + 2)
    c.xlim(0.2, 1.)
    c.ylim(1E+00, 1E+03)
    c.logy()
    c.legend()

    return c
