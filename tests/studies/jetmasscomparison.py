#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, signal_low, MASSBINS
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


@showsave
def jetmasscomparison (data, args, features, simple_features, eff_sig=50):
    """
    Perform study of jet mass distributions before and after subtructure cut for
    different substructure taggers.

    Saves plot `figures/jetmasscomparison__eff_sig_[eff_sig].pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for which to plot signal- and background distributions.
        simple_features: Whether to plot only simple features. Alternative is
            MVA features.
        eff_sig: Signal efficiency at which to impose cut.
    """

    # Define masks and direction-dependent cut value
    msk_sig = data['signal'] == 1
    cuts, msks_pass = dict(), dict()
    for feat in features:
        eff_cut = eff_sig if signal_low(feat) else 100 - eff_sig
        cut = wpercentile(data.loc[msk_sig, feat].values, eff_cut, weights=data.loc[msk_sig, 'weight_test'].values)
        msks_pass[feat] = data[feat] > cut

        # Ensure correct cut direction
        if signal_low(feat):
            msks_pass[feat] = ~msks_pass[feat]
            pass
        pass

    # Perform plotting
    c = plot(data, args, features, simple_features, msks_pass, eff_sig)

    # Output
    path = 'figures/jetmasscomparison__eff_sig_{:d}__{}.pdf'.format(int(eff_sig), 'simple' if simple_features else 'mva')

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, simple_features, msks_pass, eff_sig = argv

    # Global variable override(s)
    HISTSTYLE[True] ['label'] = " #it{W} jets"
    HISTSTYLE[False]['label'] = " QCD jets"

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Plots
    hist = dict()
    opts_legend = dict(width=0.25, xmax=0.87)

    # -- Inclusive
    base = dict(bins=MASSBINS, alpha=0.3, normalise=True, linewidth=3)
    for signal, name in zip([False, True], ['bkg', 'sig']):
        msk = data['signal'] == signal
        HISTSTYLE[signal].update(base)
        hist[name] = c.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight_test'].values, **HISTSTYLE[signal])
        pass
    c.legend(header=" Baseline selection:", ymax=0.8875, **opts_legend)

    # -- Tagged
    base['linewidth'] = 2
    base.pop('alpha')
    for ifeat, feat in filter(lambda t: simple_features == signal_low(t[1]), enumerate(features)):
        opts = dict(
            linecolor   = rp.colours[(ifeat // 2)],
            linestyle   = 1 + (ifeat % 2),
            )
        cfg = dict(**base)
        cfg.update(opts)
        msk = (data['signal'] == 0) & msks_pass[feat]
        hist[feat] = c.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight_test'].values, label=" " + latex(feat, ROOT=True), **cfg)
        pass

    c.legend(header=" Tagged QCD jets:", ymax=0.70, **opts_legend)

    # Re-draw axes
    c.pads()[0]._primitives[0].Draw('AXIS SAME')

    # Decorations
    c.xlabel("Large-#it{R} jet mass [GeV]")
    c.ylabel("Fraction of jets")

    c.text([], qualifier=QUALIFIER, ymax=0.96, xmin=0.15)
    c.text(["#sqrt{s} = 13 TeV",
            "#it{W} jet tagging",
            "{} taggers".format("Simple" if simple_features else "MVA"),
            "Tagging at #varepsilon_{sig} = %.0f%%" % eff_sig,
            ], ATLAS=False)

    c.ylim(5E-05, 5E+02)
    c.logy()

    return c
