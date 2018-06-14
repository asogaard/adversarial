#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, MASSBINS, signal_low
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


@showsave
def jetmass (data, args, feat, eff_sig=50):
    """
    Perform study of jet mass distributions before and after subtructure cut.

    Saves plot `figures/jetmass_[feat]__eff_sig_[eff_sig].pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        feat: Feature for which to plot signal- and background distributions.
        eff_sig: Signal efficiency at which to impose cut
    """

    # Define masks and direction-dependent cut value
    msk_sig = data['signal'] == 1
    msk_bkg = ~msk_sig
    eff_cut = eff_sig if signal_low(feat) else 100 - eff_sig
    cut = wpercentile(data.loc[msk_sig, feat].values, eff_cut, weights=data.loc[msk_sig, 'weight_test'].values)
    msk_pass = data[feat] > cut

    # Ensure correct cut direction
    if signal_low(feat):
        msk_pass = ~msk_pass
        pass

    # Perform plotting
    c = plot(data, args, feat, msk_pass, msk_bkg, eff_sig)

    # Output
    path = 'figures/jetmass_{}__eff_sig_{:d}.pdf'.format(standardise(feat), int(eff_sig))

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, feat, msk_pass, msk_bkg, eff_sig = argv

    # Global variable override(s)
    HISTSTYLE[True] ['label'] = "Passing cut"
    HISTSTYLE[False]['label'] = "Failing cut"

    # Canvas
    c = rp.canvas(num_pads=2, size=(int(800 * 600 / 857.), 600), batch=not args.show)

    # Plots
    base = dict(bins=MASSBINS, alpha=0.3, normalise=True, linewidth=3)
    hist = dict()
    for passing, name in zip([False, True], ['fail', 'pass']):
        msk = msk_bkg & (msk_pass if passing else ~msk_pass)
        HISTSTYLE[passing].update(base)
        hist[name] = c.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight_test'].values, **HISTSTYLE[passing])
        pass

    # Ratio plots
    c.ratio_plot((hist['pass'], hist['pass']), option='HIST', fillstyle=0, linecolor=ROOT.kGray + 1, linewidth=1, linestyle=1)
    c.ratio_plot((hist['pass'], hist['fail']), option='E2', fillstyle=1001, fillcolor=rp.colours[0], linecolor=rp.colours[0], alpha=0.3)

    # -- Set this before drawing OOB markers
    c.pads()[1].logy()
    c.pads()[1].ylim(1E-01, 1E+01)

    h_ratio = c.ratio_plot((hist['pass'], hist['fail']), option='HIST', fillstyle=0, linewidth=3, linecolor=rp.colours[0])  # oob=True, oob_color=rp.colours[0])

    # Decorations
    c.xlabel("Large-#it{R} jet mass [GeV]")
    c.ylabel("Fraction of jets")
    c.text(["#sqrt{s} = 13 TeV,  Multijets",
            "#varepsilon_{sig} = %d%% cut on %s" % (eff_sig, latex(feat, ROOT=True)),
            ], qualifier=QUALIFIER)

    c.ylim(2E-04, 2E+02)
    c.logy()
    c.legend()

    c.pads()[1].ylabel("Passing / failing")

    return c
