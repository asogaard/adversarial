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
STYLE = {  # key = passing
    True: {
        'fillcolor': rp.colours[5],
        'linecolor': rp.colours[5],
        'fillstyle': 3454,
        'label': "Passing cut",
        },
    False: {
        'fillcolor': rp.colours[1],
        'linecolor': rp.colours[1],
        'fillstyle': 3445,
        'label': "Failing jets",
        }
}


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
    eff_cut = eff_sig if signal_high(feat) else 100 - eff_sig
    cut = wpercentile(data.loc[msk_sig, feat].values, eff_cut, weights=data.loc[msk_sig, 'weight'].values)
    msk_pass = data[feat] > cut

    # Ensure correct cut direction
    if signal_high(feat):
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

    # Canvas
    c = rp.canvas(num_pads=2, size=(int(800 * 600 / 857.), 600), batch=not args.show)

    # Plots
    ROOT.gStyle.SetHatchesLineWidth(3)
    base = dict(bins=MASSBINS, alpha=0.3, normalise=True, linewidth=3)
    hist = dict()
    for passing, name in zip([False, True], ['fail', 'pass']):
        msk = msk_bkg & (msk_pass if passing else ~msk_pass)
        STYLE[passing].update(base)
        hist[name] = c.hist(data.loc[msk, 'm'].values, weights=data.loc[msk, 'weight'].values, **STYLE[passing])
        pass

    # Ratio plots
    c.pads()[1].hist([1], bins=[MASSBINS[0], MASSBINS[-1]], linecolor=ROOT.kGray + 1, linewidth=1, linestyle=1)
    h_ratio = c.ratio_plot((hist['pass'], hist['fail']), option='E2',   fillstyle=1001, fillcolor=rp.colours[0], linecolor=rp.colours[0], alpha=0.3)
    c.ratio_plot((hist['pass'], hist['fail']), option='HIST', fillstyle=0, linewidth=3, linecolor=rp.colours[0])

    # Out-of-bounds indicators
    ymin, ymax = 1E-01, 1E+01
    ratio = root_numpy.hist2array(h_ratio)
    centres = MASSBINS[:-1] + 0.5 * np.diff(MASSBINS)
    offset = 0.05  # Relative offset from top- and bottom of ratio pad

    lymin, lymax = map(np.log10, (ymin, ymax))
    ldiff = lymax - lymin

    oobx = map(lambda t: t[0], filter(lambda t: t[1] > ymax, zip(centres,ratio)))
    ooby = np.ones_like(oobx) * np.power(10, lymax - offset * ldiff)
    if len(oobx) > 0:
        c.pads()[1].graph(ooby, bins=oobx, markercolor=rp.colours[0], markerstyle=22, option='P')
        pass

    oobx = map(lambda t: t[0], filter(lambda t: t[1] < ymin, zip(centres,ratio)))
    ooby = np.ones_like(oobx) * np.power(10, lymin + offset * ldiff)
    if len(oobx) > 0:
        c.pads()[1].graph(ooby, bins=oobx, markercolor=rp.colours[0], markerstyle=23, option='P')
        pass

    # Decorations
    ROOT.gStyle.SetTitleOffset(1.6, 'y')
    c.xlabel("Large-#it{R} jet mass [GeV]")
    c.ylabel("Fraction of jets")
    c.text(["#sqrt{s} = 13 TeV,  QCD jets",
            "Testing dataset",
            "Baseline selection",
            "Fixed #varepsilon_{sig.} = %d%% cut on %s" % (eff_sig, latex(feat, ROOT=True)),
            ],
        qualifier=QUALIFIER)
    c.ylim(2E-04, 2E+02)

    c.pads()[1].ylabel("Passing / failing")
    c.pads()[1].logy()
    c.pads()[1].ylim(ymin, ymax)

    c.logy()
    c.legend()

    return c
