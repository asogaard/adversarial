#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ROOT import(s)
import ROOT

# Project import(s)
from .common import *
from adversarial.utils import wpercentile, wmean, latex, bootstrap_metrics, signal_low
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


@showsave
def robustness (data, args, features, var, bins, masscut=False, num_bootstrap=5):
    """
    Perform study of robustness wrt. `var`.

    Saves plot `figures/robustness_{var}.pdf`

    Arguments:
        data: Pandas data frame from which to read data.
        args: Namespace holding command-line arguments.
        features: Features for which to ...
        var: ...
        bins: ...
        masscut: ...
        num_bootstrap: ...
    """

    # Compute metrics for all features
    rejs, jsds = {feat: [] for feat in features}, {feat: [] for feat in features}
    meanx = list()

    # Scan `var`
    for bin in zip(bins[:-1], bins[1:]):

        # Perform selection
        msk_bin  = (data[var] >= bin[0]) & (data[var] < bin[1])
        msk_bkg  = data['signal'] == 0

        # Compute weighted mean if x-axis variable
        meanx.append(wmean(data.loc[msk_bin & msk_bkg, var], data.loc[msk_bin & msk_bkg, 'weight_test']))

        # Compute bootstrapped metrics for all features
        for feat in features:
            mean_std_rej, mean_std_jsd = bootstrap_metrics(data[msk_bin], feat, num_bootstrap=num_bootstrap, masscut=masscut)

            # @TODO: Compute JSD limit
            #frac = ...
            #jsds = jsd_limit(data[msk_bin], frac)
            #np.mean(1/jsds) ± np.std(1/jsds)
            # ...

            # Store in output containers
            rejs[feat].append(mean_std_rej)
            jsds[feat].append(mean_std_jsd)
            pass
        pass

    # Format array
    meanx = np.array(meanx).astype(float)

    # Perform plotting
    c = plot(data, args, features, bins, rejs, jsds, meanx, masscut, var)

    # Output
    path = 'figures/robustness_{}{}.pdf'.format(var, '_masscut' if masscut else '')

    return c, args, path


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, bins, rejs, jsds, meanx, masscut, var = argv

    with TemporaryStyle() as style:

        # Set styles
        scale = 0.9
        style.SetTextSize(scale * style.GetTextSize())
        for coord in ['x', 'y', 'z']:
            style.SetLabelSize(scale * style.GetLabelSize(coord), coord)
            style.SetTitleSize(scale * style.GetTitleSize(coord), coord)
            pass

        # Canvas
        c = rp.canvas(num_pads=2, fraction=0.55, size=(int(800 * 600 / 857.), 600), batch=not args.show)
        c.pads()[0]._bare().SetTopMargin(0.10)
        c.pads()[0]._bare().SetRightMargin(0.23)
        c.pads()[1]._bare().SetRightMargin(0.23)

        # To fix 30.5 --> 30 for NPV
        bins[-1] = np.floor(bins[-1])

        # Plots
        c.pads()[0].hist([0], bins=[bins[0], bins[-1]], linestyle=0, fillstyle=0)
        c.pads()[1].hist([1], bins=[bins[0], bins[-1]], linecolor=ROOT.kGray + 2)
        for is_simple in [True, False]:
            for ifeat, feat in filter(lambda t: is_simple == signal_low(t[1]), enumerate(features)):

                opts = dict(
                    linecolor   = rp.colours[(ifeat // 2)],
                    markercolor = rp.colours[(ifeat // 2)],
                    fillcolor   = rp.colours[(ifeat // 2)],
                    linestyle   = 1 + (ifeat % 2),
                    alpha       = 0.3,
                    option      = 'E2',
                )

                mean_rej, std_rej = map(np.array, zip(*rejs[feat]))
                mean_jsd, std_jsd = map(np.array, zip(*jsds[feat]))

                # Error boxes
                x    = np.array(bins[:-1]) + 0.5 * np.diff(bins)
                xerr = 0.5 * np.diff(bins)
                graph_rej = ROOT.TGraphErrors(len(x), x, mean_rej, xerr, std_rej)
                graph_jsd = ROOT.TGraphErrors(len(x), x, mean_jsd, xerr, std_jsd)

                c.pads()[0].hist(graph_rej, **opts)
                c.pads()[1].hist(graph_jsd, **opts)

                # Markers and lines
                opts['option']      = 'PE2L'
                opts['markerstyle'] = 20 + 4 * (ifeat % 2)

                graph_rej = ROOT.TGraph(len(x), meanx, mean_rej)
                graph_jsd = ROOT.TGraph(len(x), meanx, mean_jsd)

                c.pads()[0].hist(graph_rej, label=latex(feat, ROOT=True) if not is_simple else None, **opts)
                c.pads()[1].hist(graph_jsd, label=latex(feat, ROOT=True) if     is_simple else None, **opts)
                pass

            pass

        # Draw class-specific legend
        width = 0.20
        c.pads()[0].legend(header='MVA:',    width=width, xmin=0.79, ymax=0.92)
        c.pads()[1].legend(header='Simple:', width=width, xmin=0.79, ymax=0.975)

        # Decorations
        for pad in c.pads():
            pad._xaxis().SetNdivisions(504)
            pass

        # -- x-axis label
        if var == 'pt':
            xlabel = "Large-#it{R} jet p_{T} [GeV]"
        elif var == 'npv':
            xlabel = "Number of reconstructed vertices N_{PV}"
        else:
            raise NotImplementedError("Variable {} is not supported.".format(xlabel))

        c.xlabel(xlabel)
        c.pads()[0].ylabel("1/#varepsilon_{bkg} @ #varepsilon_{sig} = 50%")
        c.pads()[1].ylabel("1/JSD @ #varepsilon_{sig} = 50%")

        c.text([], qualifier=QUALIFIER, xmin=0.15, ymax=0.93)

        c.text(["#sqrt{s} = 13 TeV",
                "#it{W} jet tagging"] + \
                (['m #in  [60, 100] GeV'] if masscut else []),
                 ATLAS=False, ymax=0.76)

        c.pads()[1].text(["QCD jets"], ATLAS=False)

        #c.pads()[0].padding(0.5)
        c.pads()[0].ymin(1)

        c.pads()[1].ymin(0.5)
        c.pads()[1].padding(0.2)

        c.pads()[0].logy()
        c.pads()[1].logy()
        pass  # Temporary style scope

    return c
