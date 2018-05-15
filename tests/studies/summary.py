#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
from sklearn.metrics import roc_curve

# ROOT import(s)
import ROOT

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex, metrics, signal_low, JSD, MASSBINS
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


@showsave
def summary (data, args, features, scan_features, target_tpr=0.5, num_bootstrap=5, masscut=False):
    """
    Perform study of combined classification- and decorrelation performance.

    Saves plot `figures/summary.pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Python list of named features in `data` to study.
        scan_features: Python dict of parameter scan-features. The dict key is
            the base features to which the parameter scan belongs. The dict
            values are a list of tuples of `(name, label)`, where `name` is the
            name of the appropriate feature in `data` and `label` is drawn on
            the plot next to each scan point. For instance:
                scan_features = {'NN': [('ANN(#lambda=3)', '#lambda=3'),
                                        ('ANN(#lambda=10)', '#lambda=10'),
                                        ('ANN(#lambda=30)', '#lambda=30')],
                                 'Adaboost': ...,
                                 }
        target_tpr: ...
        num_bootstrap: ...
    """

    # Check(s)
    assert isinstance(features, list)
    assert isinstance(scan_features, dict)

    # For reproducibility of bootstrap sampling
    np.random.seed(7)

    # Compute metrics for all features
    points = list()
    for feat in features + map(lambda t: t[0], [it for gr in scan_features.itervalues() for it in gr]):
        print  "-- {}".format(feat)

        # Check for duplicates
        if feat in map(lambda t: t[2], points):
            print "    Skipping (already encounted)"
            continue

        # Compute metrics
        _, rej, jsd = metrics(data, feat, masscut=masscut)

        # Add point to array
        points.append((rej, jsd, feat))
        pass

    # Compute meaningful limit for 1/JSD based on bootstrapping
    num_bootstrap = 10
    jsd_limits = list()
    for bkg_rej in np.logspace(np.log10(2.), np.log10(100), 2 * 10 + 1, endpoint=True):
        frac = 1. / float(bkg_rej)

        limits = 1./np.array(jsd_limit(data, frac, num_bootstrap=5))
        jsd_limits.append((bkg_rej, np.mean(limits), np.std(limits)))
        pass

    # Perform plotting
    c = plot(data, args, features, scan_features, points, jsd_limits, masscut)

    # Output
    path = 'figures/summary{}.pdf'.format('_masscut' if masscut else '')

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, scan_features, points, jsd_limits, masscut = argv

    with TemporaryStyle() as style:

        # Define variable(s)
        axisrangex = (1.4,    100.)
        axisrangey = (0.3, 500000.)
        aminx, amaxx = axisrangex
        aminy, amaxy = axisrangey

        # Styling
        style.SetTitleOffset(1.8, 'x')
        style.SetTitleOffset(1.6, 'y')

        # Canvas
        c = rp.canvas(batch=not args.show, size=(600,600))

        # Reference lines
        nullopts = dict(linecolor=0, linewidth=0, linestyle=0, markerstyle=0, markersize=0, fillstyle=0)
        lineopts = dict(linecolor=ROOT.kGray + 2, linewidth=1, option='L')
        boxopts  = dict(fillcolor=ROOT.kBlack, alpha=0.05, linewidth=0, option='HIST')
        c.hist([aminy], bins=list(axisrangex), **nullopts)
        c.plot([1, amaxy], bins=[2, 2],     **lineopts)
        c.plot([1, 1],     bins=[2, amaxx], **lineopts)
        c.hist([amaxy],    bins=[aminx, 2], **boxopts)
        c.hist([1],        bins=[2, amaxx], **boxopts)

        # Markers
        for is_simple in [True, False]:

            # Split the legend into simple- and MVA taggers
            for ifeat, feat in filter(lambda t: is_simple == signal_low(t[1]), enumerate(features)):

                # Coordinates, label
                idx = map(lambda t: t[2], points).index(feat)
                x, y, label = points[idx]

                # Overwrite default name of parameter-scan classifier
                label = 'ANN'    if label.startswith('ANN') else label
                label = 'uBoost' if label.startswith('uBoost') else label

                # Style
                colour      = rp.colours[(ifeat // 2) % len(rp.colours)]
                markerstyle = 20 + (ifeat % 2) * 4

                # Draw
                c.graph([y], bins=[x], markercolor=colour, markerstyle=markerstyle, label=latex(label, ROOT=True), option='P')
                pass

            # Draw class-specific legend
            width = 0.18
            c.legend(header=("Simple:" if is_simple else "MVA:"),
                     width=width, xmin=0.54 + (width + 0.02) * (is_simple), ymax=0.827)
            pass

        # Markers, parametrised decorrelation
        for base_feat, group in scan_features.iteritems():

            # Get index in list of features
            ifeat = features.index(base_feat)

            # Style
            colour      = rp.colours[(ifeat // 2) % len(rp.colours)]
            markerstyle = 24

            for feat, label in group:
                idx = map(lambda t: t[2], points).index(feat)
                x, y, _ = points[idx]

                # Draw
                c.graph([y], bins=[x], markercolor=colour, markerstyle=markerstyle, option='P')
                if base_feat == 'NN':
                    c.latex("   " + label, x, y, textsize=11, align=12, textcolor=ROOT.kGray + 2)
                else:
                    c.latex(label + "   ", x, y, textsize=11, align=32, textcolor=ROOT.kGray + 2)
                    pass
                pass

            # Connecting lines (scan)
            feats = [base_feat] + map(lambda t: t[0], group)
            for feat1, feat2 in zip(feats[:-1], feats[1:]):
                idx1 = map(lambda t: t[2], points).index(feat1)
                idx2 = map(lambda t: t[2], points).index(feat2)

                x1, y1, _ = points[idx1]
                x2, y2, _ = points[idx2]

                c.graph([y1, y2], bins=[x1, x2], linecolor=colour, linestyle=2, option='L')
                pass
            pass

        # Connecting lines (simple)
        for i in range(3):
            x1, y1, _ = points[2 * i + 0]
            x2, y2, _ = points[2 * i + 1]
            colour = rp.colours[i]
            c.graph([y1, y2], bins=[x1, x2], linecolor=colour, linestyle=2, option='L')
            pass

        # Meaningful limits on 1/JSD
        x,y,ey = map(np.array, zip(*jsd_limits))
        ex = np.zeros_like(ey)
        gr = ROOT.TGraphErrors(len(x), x, y, ex, ey)
        smooth_tgrapherrors(gr, ntimes=3)
        c.graph(gr, linestyle=2, linecolor=ROOT.kGray + 1, fillcolor=ROOT.kBlack, alpha=0.03, option='L3')

        x_, y_, ex_, ey_ = ROOT.Double(0), ROOT.Double(0), ROOT.Double(0), ROOT.Double(0)
        idx = 3
        gr.GetPoint(idx, x_,  y_)
        ey_ = gr.GetErrorY(idx)
        x_, y_ = map(float, (x_, y_))
        c.latex("Statistical limit", x_, y_ + ey_, align=21, textsize=11, angle=-5, textcolor=ROOT.kGray + 2)

        # Decorations
        c.xlabel("Background rejection, 1 / #varepsilon_{bkg} @ #varepsilon_{sig} = 50%")
        c.ylabel("Mass decorrelation, 1 / JSD @ #varepsilon_{sig} = 50%")
        c.xlim(*axisrangex)
        c.ylim(*axisrangey)
        c.logx()
        c.logy()

        opts_text = dict(textsize=11, textcolor=ROOT.kGray + 2)
        midpointx = np.power(10, 0.5 * np.log10(amaxx))
        midpointy = np.power(10, 0.5 * np.log10(amaxy))
        c.latex("No separation",                     1.91, midpointy, angle=90, align=21, **opts_text)
        c.latex("Maximal sculpting",                 midpointx, 0.89, angle= 0, align=23, **opts_text)
        c.latex("    Less sculpting #rightarrow",    2.1, midpointy,  angle=90, align=23, **opts_text)
        c.latex("    Greater separtion #rightarrow", midpointx, 1.1,  angle= 0, align=21, **opts_text)

        #c.text(TEXT + ["#it{W} jet tagging"], xmin=0.24, qualifier=QUALIFIER)
        c.text(TEXT + \
               ["#it{W} jet tagging"] + \
               (['m #in  [60, 100] GeV'] if masscut else []),
               xmin=0.26, qualifier=QUALIFIER)
        pass

    return c
