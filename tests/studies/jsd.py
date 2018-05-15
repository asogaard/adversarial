#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import numpy as np

# ROOT import(s)
import ROOT
import root_numpy

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, wpercentile, signal_low, JSD, MASSBINS
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


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
    for ifeat, feat in enumerate(features):

        # Define cuts
        cuts = list()
        for eff in effs:
            cut = wpercentile(data.loc[msk, feat].values, eff if signal_low(feat) else 100 - eff, weights=data.loc[msk, 'weight_test'].values)
            cuts.append(cut)
            pass

        # Compute KL divergence for successive cuts
        for cut, eff in zip(cuts, effs):

            # Create ROOT histograms
            msk_pass = data[feat] > cut
            if signal_low(feat):
                msk_pass = ~msk_pass
                pass

            h_pass = c.hist(data.loc[ msk_pass & msk, 'm'].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk, 'weight_test'].values, normalise=True, display=False)
            h_fail = c.hist(data.loc[~msk_pass & msk, 'm'].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk, 'weight_test'].values, normalise=True, display=False)

            # Convert to numpy arrays
            p = root_numpy.hist2array(h_pass)
            f = root_numpy.hist2array(h_fail)

            # Compute Jensen-Shannon divergence
            jsd[feat].append(JSD(p, f, base=2))

            pass
        pass

    # Compute meaningful limit on JSD
    jsd_limits = list()
    sigmoid = lambda x: 1. / (1. + np.exp(-x))
    for eff in sigmoid(np.linspace(-5, 5, 20 + 1, endpoint=True)):
        limits = jsd_limit(data[msk], eff, num_bootstrap=5)
        jsd_limits.append((eff, np.mean(limits), np.std(limits)))
        pass

    # Perform plotting
    c = plot(args, data, effs, jsd, jsd_limits, features)

    # Output
    path = 'figures/jsd.pdf'

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    args, data, effs, jsd, jsd_limits, features = argv

    with TemporaryStyle() as style:

        # Style
        style.SetTitleOffset(1.5, 'x')
        style.SetTitleOffset(2.0, 'y')

        # Canvas
        c = rp.canvas(batch=not args.show)

        # Plots
        ref = ROOT.TH1F('ref', "", 10, 0., 1.)
        for i in range(ref.GetXaxis().GetNbins()):
            ref.SetBinContent(i + 1, 1)
            pass
        c.hist(ref, linecolor=ROOT.kGray + 2, linewidth=1)

        width = 0.15
        for is_simple in [True, False]:
            for ifeat, feat in enumerate(features):
                if is_simple != signal_low(feat): continue
                colour = rp.colours[(ifeat // 2) % len(rp.colours)]
                linestyle   =  1 + (ifeat % 2)
                markerstyle = 20 + (ifeat % 2) * 4
                c.plot(jsd[feat], bins=np.array(effs) / 100., linecolor=colour, markercolor=colour, linestyle=linestyle, markerstyle=markerstyle, label=latex(feat, ROOT=True), option='PL')
                pass

            c.legend(header=("Simple:" if is_simple else "MVA:"),
                     width=width * (1 + 0.8 * int(is_simple)), xmin=0.42 + (width + 0.05) * (is_simple), ymax=0.888,
                     columns=2 if is_simple else 1,
                     margin=0.35)
            pass

        # Meaningful limits on JSD
        x,y,ey = map(np.array, zip(*jsd_limits))
        ex = np.zeros_like(ey)
        gr = ROOT.TGraphErrors(len(x), x, y, ex, ey)
        smooth_tgrapherrors(gr, ntimes=2)
        c.graph(gr, linestyle=2, linecolor=ROOT.kGray + 1, fillcolor=ROOT.kBlack, alpha=0.03, option='L3')

        # Redraw axes
        c.pads()[0]._primitives[0].Draw('AXIS SAME')

        # Decorations
        c.xlabel("Background efficiency #varepsilon_{bkg}")
        c.ylabel("Mass correlation, JSD")
        c.text([], xmin=0.15, ymax = 0.96, qualifier=QUALIFIER)
        c.text(["#sqrt{s} = 13 TeV",  "QCD jets"],
               ymax=0.85, ATLAS=None)

        c.latex("Maximal sculpting", 0.065, 1.2, align=11, textsize=11, textcolor=ROOT.kGray + 2)
        c.xlim(0, 1)
        c.ymin(5E-05)
        c.padding(0.45)
        c.logy()

        for leg in c.pad()._legends:
            leg.SetMargin(0.5)
            pass

        x_, y_, ex_, ey_ = ROOT.Double(0), ROOT.Double(0), ROOT.Double(0), ROOT.Double(0)
        idx = gr.GetN() - 7
        gr.GetPoint(idx, x_,  y_)
        ey_ = gr.GetErrorY(idx)
        x_, y_ = map(float, (x_, y_))
        c.latex("Statistical limit", x_, y_ - ey_ / 2., align=23, textsize=11, angle=16, textcolor=ROOT.kGray + 2)
        pass

    return c
