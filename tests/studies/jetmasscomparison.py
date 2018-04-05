#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
global HISTSTYLE
HISTSTYLE[True] ['label'] = "#it{W} jets"
HISTSTYLE[False]['label'] = "QCD jets"


@showsave
def jetmasscomparison (data, args, features, eff_sig=50):
    """
    Perform study of jet mass distributions before and after subtructure cut for
    different substructure taggers.

    Saves plot `figures/jetmasscomparison__eff_sig_[eff_sig].pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for which to plot signal- and background distributions.
        eff_sig: Signal efficiency at which to impose cut.
    """

    # Define masks and direction-dependent cut value
    msk_sig = data['signal'] == 1
    msk_bkg = ~msk_sig
    cuts, msks_pass = dict(), dict()
    for feat in features:
        eff_cut = eff_sig if signal_low(feat) else 100 - eff_sig
        cut = wpercentile(data.loc[msk_sig, feat].values, eff_cut, weights=data.loc[msk_sig, 'weight'].values)
        msks_pass[feat] = data[feat] > cut

        # Ensure correct cut direction
        if signal_low(feat):
            msks_pass[feat] = ~msks_pass[feat]
            pass
        pass

    # Perform plotting
    c = plot(data, args, features, msks_pass, msk_bkg, eff_sig)

    # Output
    path = 'figures/jetmasscomparison__eff_sig_{:d}.pdf'.format(int(eff_sig))

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, msks_pass, msk_bkg, eff_sig = argv

    # Canvas
    c = rp.canvas(batch=not args.show)

    # Plots
    base = dict(bins=MASSBINS, alpha=0.3, normalise=True, linewidth=3)
    hist = dict()
    for signal, name in zip([False, True], ['bkg', 'sig']):
        msk = msk_bkg if signal else ~msk_bkg
        HISTSTYLE[signal].update(base)
        hist[name] = c.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight'].values, **HISTSTYLE[signal])
        pass

    for ifeat, feat in enumerate(features):
        print "",
        pass

    # Decorations
    c.xlabel("Large-#it{R} jet mass [GeV]")
    c.ylabel("Fraction of jets")
    c.text(["#sqrt{s} = 13 TeV,  QCD jets",
            "Testing dataset",
            "Baseline selection",
            "Fixed #varepsilon_{sig.} = %d%% cut on %s" % (eff_sig, latex(feat, ROOT=True)),
            ], qualifier=QUALIFIER)

    c.ylim(2E-04, 2E+02)
    c.logy()
    c.legend()

    return c
