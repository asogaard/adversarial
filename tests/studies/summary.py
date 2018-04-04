#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
from sklearn.metrics import roc_curve

# ROOT import(s)
import ROOT

# Project import(s)
from .common import *
from adversarial.utils import mkdir, latex
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


@showsave
def summary (data, args, tagger_features, scan_features):
    """
    Perform study of combined classification- and decorrelation performance.

    Saves plot `figures/summary.pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        tagger_features: Python list of named features in `data` to study.
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
    """

    # Check(s)
    assert isinstance(tagger_features, list)
    assert isinstance(scan_features, dict)

    # Define variable(s)
    target_tpr = 0.5
    axisrangex = (0.6,   500.)
    axisrangey = (0.4, 50000.)
    aminx, amaxx = axisrangex
    aminy, amaxy = axisrangey

    # Compute metrics for all features
    points = list()
    for feat in tagger_features + map(lambda t: t[0], [it for gr in scan_features.itervalues() for it in gr]):
        print  "-- {}".format(feat)

        # Check for duplicates
        if feat in map(lambda t: t[2], points):
            print "    Skipping (already encounted)"
            continue

        """
        # scikit-learn assumes signal towards 1, background towards 0
        pred = data[feat].values.copy()
        if signal_high(feat):
            print "   Reversing cut direction for {}".format(feat)
            pred *= -1.
            pass

        # Compute ROC curve efficiencies
        fpr, tpr, thresholds = roc_curve(data['signal'], pred, sample_weight=data['weight'])

        # Get background rejection factor @ eff_sig. = 50%
        idx = np.argmin(np.abs(tpr - target_tpr))
        rej = 1. / fpr[idx]
        cut = thresholds[idx]

        # Get JSD(pass || fail) @ eff_sig. = 50%
        msk_bkg  = data['signal'] == 0
        msk_pass = pred > cut

        p, _ = np.histogram(data.loc[ msk_pass & msk_bkg, 'm'].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk_bkg, 'weight'].values, density=True)
        f, _ = np.histogram(data.loc[~msk_pass & msk_bkg, 'm'].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk_bkg, 'weight'].values, density=True)

        jsd = JSD(p, f)
        """
        rej, jsd = metrics(data, feat)

        # Add point to array
        points.append((rej, 1. / jsd, feat))
        pass


    # Pre-styling
    ref_title_offset_x = ROOT.gStyle.GetTitleOffset('x')
    ret_title_offset_y = ROOT.gStyle.GetTitleOffset('y')

    ROOT.gStyle.SetTitleOffset(1.8, 'x')
    ROOT.gStyle.SetTitleOffset(1.6, 'y')

    # Canvas
    c = rp.canvas(batch=not args.show, size=(600,600))

    # Reference lines
    nullopts = dict(linecolor=0, linewidth=0, linestyle=0, markerstyle=0, markersize=0, fillstyle=0)
    c.hist([aminy], bins=list(axisrangex), **nullopts)
    c.plot([1, amaxy], bins=[1, 1],    linecolor=ROOT.kGray + 2, linewidth=1, option='L')
    c.plot([1, 1],     bins=[1, amaxx], linecolor=ROOT.kGray + 2, linewidth=1, option='L')

    # Markers
    for is_simple in [True, False]:
        # Split the legend into simple- and MVA taggers
        for ipoint, feat in enumerate(tagger_features):

            # Select only appropriate taggers
            if is_simple != signal_low(feat): continue

            # Coordinates, label
            idx = map(lambda t: t[2], points).index(feat)
            x, y, label = points[idx]

            # Overwrite default name of parameter-scan classifier
            label = 'ANN' if label.startswith('ANN') else label
            # @TODO: uBoost

            # Style
            ipoint += 3 if ipoint > 3 else 0  # @TEMP

            colour      = rp.colours[(ipoint // 2) % len(rp.colours)]
            markerstyle = 20 + (ipoint % 2) * 4

            # Draw
            c.graph([y], bins=[x], markercolor=colour, markerstyle=markerstyle, label=latex(label, ROOT=True), option='P')
            pass

        # Draw class-specific legend
        width = 0.18
        c.legend(header=("Simple:" if is_simple else "MVA:"),
                 width=width, xmin=0.54 + (width + 0.02) * (is_simple), ymax=0.827)  # ymax=0.782)
        pass

    # Markers, parametrised decorrelation
    for base_feat, group in scan_features.iteritems():
        # Get index in list of features
        ipoint = tagger_features.index(base_feat)
        ipoint += 3 if ipoint > 3 else 0  # @TEMP

        # Style
        colour      = rp.colours[(ipoint // 2) % len(rp.colours)]
        markerstyle = 20 + (ipoint % 2) * 4

        for feat, label in group:
            idx = map(lambda t: t[2], points).index(feat)
            x, y, _ = points[idx]

            # Draw
            c.graph([y], bins=[x], markercolor=colour, markerstyle=markerstyle, option='P')
            c.latex("   " + label, x, y, textsize=11, align=12, textcolor=ROOT.kGray + 2)
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
    for i in [0,1]:
        x1, y1, _ = points[2 * i + 0]
        x2, y2, _ = points[2 * i + 1]
        colour = rp.colours[i]
        c.graph([y1, y2], bins=[x1, x2], linecolor=colour, linestyle=2, option='L')
        pass

    # Boxes
    box1 = ROOT.TBox(aminx, aminy, 1, amaxy)
    box1.SetFillColorAlpha(ROOT.kBlack, 0.05)
    box1.Draw("SAME")

    box2 = ROOT.TBox(1, aminy, amaxx, 1)
    box2.SetFillColorAlpha(ROOT.kBlack, 0.05)
    box2.Draw("SAME")
    c.pads()[0]._primitives[0].Draw('AXIS SAME')

    # Decorations
    c.xlabel("Background rejection, 1 / #varepsilon_{bkg.} @ #varepsilon_{sig.} = 50%")
    c.ylabel("Mass decorrelation, 1 / JSD @ #varepsilon_{sig.} = 50%")
    c.xlim(*axisrangex)
    c.ylim(*axisrangey)
    c.logx()
    c.logy()

    opts_text = dict(textsize=11, textcolor=ROOT.kGray + 2)
    midpointx = np.power(10, 0.5 * np.log10(amaxx))
    midpointy = np.power(10, 0.5 * np.log10(amaxy))
    c.latex("No separation",                     0.90, midpointy, angle=90, align=21, **opts_text)
    c.latex("Maximal sculpting",                 midpointx, 0.90, angle= 0, align=23, **opts_text)
    c.latex("    Less sculpting #rightarrow",    1.1, midpointy,  angle=90, align=23,**opts_text)
    c.latex("    Greater separtion #rightarrow", midpointx, 1.1,  angle= 0, align=21,**opts_text)

    c.text(["#sqrt{s} = 13 TeV",
            "Testing dataset",
            "Baseline selection",
            ],
        xmin=0.24,
        qualifier=QUALIFIER)

    # Reset styles
    ROOT.gStyle.SetTitleOffset(ref_title_offset_x, 'x')
    ROOT.gStyle.SetTitleOffset(ret_title_offset_y, 'y')

    # Output
    path = 'figures/summary.pdf'

    return c, args, path
