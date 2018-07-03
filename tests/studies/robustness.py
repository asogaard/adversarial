#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import itertools

# ROOT import(s)
import ROOT

# Project import(s)
from .common import *
from adversarial.utils import wpercentile, wmean, latex, bootstrap_metrics, signal_low
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp



@showsave
def robustness_full (data, args, features, masscut=False, num_bootstrap=5):

    # Compute relevant quantities
    bins, effs, rejs, jsds, meanx, jsd_limits = dict(), dict(), dict(), dict(), dict(), dict()

    # -- pt
    var = 'pt'
    bins[var] = [200, 260, 330, 430, 560, 720, 930, 1200, 1550, 2000]
    effs[var], rejs[var], jsds[var], meanx[var], jsd_limits[var] = compute(data, args, features, var, bins[var], masscut, num_bootstrap)

    # -- npv
    var = 'npv'
    bins[var] = [0,  5.5, 10.5, 15.5, 20.5, 25.5, 30.5]
    effs[var], rejs[var], jsds[var], meanx[var], jsd_limits[var] = compute(data, args, features, var, bins[var], masscut, num_bootstrap)

    # Perform plotting
    c = plot_full(data, args, features, bins, effs, rejs, jsds, meanx, jsd_limits, masscut)
    plot_individual(data, args, features, bins, effs, rejs, jsds, meanx, jsd_limits, masscut)

    # Output
    path = 'figures/robustness{}.pdf'.format('_masscut' if masscut else '')

    return c, args, path

    pass


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

    # Compute relevant quantities
    effs, rejs, jsds, meanx, jsd_limits = compute(data, args, features, var, bins, masscut, num_bootstrap)

    # Perform plotting
    c = plot(data, args, features, bins, effs, rejs, jsds, meanx, jsd_limits, masscut, var)

    # Output
    path = 'figures/robustness_{}{}.pdf'.format(var, '_masscut' if masscut else '')

    return c, args, path


def compute (data, args, features, var, bins, masscut, num_bootstrap):
    """
    ...
    """

    # Compute metrics for all features
    effs, rejs, jsds = {feat: [] for feat in features}, \
                       {feat: [] for feat in features}, \
                       {feat: [] for feat in features}
    meanx = list()

    # For reproducibility of bootstrap sampling
    np.random.seed(7)

    # Scan `var`
    jsd_limits = list()
    for bin in zip(bins[:-1], bins[1:]):

        # Perform selection
        msk_bin  = (data[var] >= bin[0]) & (data[var] < bin[1])
        msk_bkg  = data['signal'] == 0

        # Compute weighted mean if x-axis variable
        meanx.append(wmean(data.loc[msk_bin & msk_bkg, var], data.loc[msk_bin & msk_bkg, 'weight_test']))

        # Compute bootstrapped metrics for all features
        for feat in set(features):  # Ensure no duplicate features

            mean_std_eff, mean_std_rej, mean_std_jsd = bootstrap_metrics(data[msk_bin], feat, num_bootstrap=num_bootstrap, masscut=masscut)

            # Store in output containers
            effs[feat].append(mean_std_eff)
            rejs[feat].append(mean_std_rej)
            jsds[feat].append(mean_std_jsd)
            pass

        # Compute meaningful limit on JSD
        bin_bkgeffs = [1./rejs[feat][-1][0] for feat in features]
        limits_min  = map(lambda jsd: 1./jsd, jsd_limit(data[msk_bin & msk_bkg], np.min (bin_bkgeffs), num_bootstrap=num_bootstrap))
        limits_mean = map(lambda jsd: 1./jsd, jsd_limit(data[msk_bin & msk_bkg], np.mean(bin_bkgeffs), num_bootstrap=num_bootstrap))
        limits_max  = map(lambda jsd: 1./jsd, jsd_limit(data[msk_bin & msk_bkg], np.max (bin_bkgeffs), num_bootstrap=num_bootstrap))
        jsd_limits.append((meanx[-1], np.mean(limits_mean), np.std(limits_max), np.abs(np.mean(limits_min) - np.mean(limits_max)) / 2.))
        pass

    # Format array
    meanx = np.array(meanx).astype(float)

    return effs, rejs, jsds, meanx, jsd_limits


def plot_full (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, bins, effs, rejs, jsds, meanx, jsd_limits, masscut = argv

    with TemporaryStyle() as style:

        # Set styles
        scale      = 1.0
        scale_axis = 0.7
        margin_squeeze = 0.035
        margin_vert    = 0.20
        margin_hori    = 0.35
        size = (800, 600)

        style.SetTextSize(scale_axis * style.GetTextSize())
        for coord in ['x', 'y', 'z']:
            style.SetLabelSize(scale_axis * style.GetLabelSize(coord), coord)
            style.SetTitleSize(scale_axis * style.GetTitleSize(coord), coord)
            pass
        style.SetLegendTextSize(style.GetLegendTextSize() * scale)
        style.SetTickLength(0.05,                                                               'x')
        style.SetTickLength(0.07 * (float(size[0])/float(size[1])) * (margin_hori/margin_vert), 'y')

        # Canvas
        c = rp.canvas(num_pads=(2,2), size=size, batch=not args.show)

        # Margins
        c.pads()[0]._bare().SetTopMargin   (margin_vert)
        c.pads()[1]._bare().SetTopMargin   (margin_vert)
        c.pads()[2]._bare().SetBottomMargin(margin_vert)
        c.pads()[3]._bare().SetBottomMargin(margin_vert)

        c.pads()[0]._bare().SetLeftMargin  (margin_hori)
        c.pads()[2]._bare().SetLeftMargin  (margin_hori)
        c.pads()[1]._bare().SetRightMargin (margin_hori)
        c.pads()[3]._bare().SetRightMargin (margin_hori)

        c.pads()[1]._bare().SetLeftMargin  (margin_squeeze)
        c.pads()[3]._bare().SetLeftMargin  (margin_squeeze)
        c.pads()[0]._bare().SetRightMargin (margin_squeeze)
        c.pads()[2]._bare().SetRightMargin (margin_squeeze)

        c.pads()[0]._bare().SetBottomMargin(margin_squeeze)
        c.pads()[1]._bare().SetBottomMargin(margin_squeeze)
        c.pads()[2]._bare().SetTopMargin   (margin_squeeze)
        c.pads()[3]._bare().SetTopMargin   (margin_squeeze)

        # To fix 30.5 --> 30 for NPV
        bins['npv'][-1] = np.floor(bins['npv'][-1])

        # Plots
        # -- References
        boxopts  = dict(fillcolor=ROOT.kBlack, alpha=0.05, linecolor=ROOT.kGray + 2, linewidth=1, option='HIST')
        c.pads()[0].hist([2], bins=[bins['pt'] [0], bins['pt'] [-1]], **boxopts)
        c.pads()[1].hist([2], bins=[bins['npv'][0], bins['npv'][-1]], **boxopts)
        c.pads()[2].hist([1], bins=[bins['pt'] [0], bins['pt'] [-1]], **boxopts)
        c.pads()[3].hist([1], bins=[bins['npv'][0], bins['npv'][-1]], **boxopts)

        nb_col = 2
        for col, var in enumerate(['pt', 'npv']):
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

                    mean_rej, std_rej = map(np.array, zip(*rejs[var][feat]))  # @TEMP
                    mean_jsd, std_jsd = map(np.array, zip(*jsds[var][feat]))

                    # Only _show_ mass-decorrelated features for `npv`
                    if (col == 1) and (ifeat % 2 == 0):
                        mean_rej *= -9999.
                        mean_jsd *= -9999.
                        pass

                    # Error boxes
                    x    = np.array(bins[var][:-1]) + 0.5 * np.diff(bins[var])
                    xerr = 0.5 * np.diff(bins[var])
                    graph_rej = ROOT.TGraphErrors(len(x), x, mean_rej, xerr, std_rej)
                    graph_jsd = ROOT.TGraphErrors(len(x), x, mean_jsd, xerr, std_jsd)

                    c.pads()[col + 0 * nb_col].hist(graph_rej, **opts)
                    c.pads()[col + 1 * nb_col].hist(graph_jsd, **opts)

                    # Markers and lines
                    opts['option']      = 'PE2L'
                    opts['markerstyle'] = 20 + 4 * (ifeat % 2)

                    graph_rej = ROOT.TGraph(len(x), meanx[var], mean_rej)
                    graph_jsd = ROOT.TGraph(len(x), meanx[var], mean_jsd)

                    c.pads()[col + 0 * nb_col].hist(graph_rej, label=latex(feat, ROOT=True) if not is_simple else None, **opts)
                    c.pads()[col + 1 * nb_col].hist(graph_jsd, label=latex(feat, ROOT=True) if     is_simple else None, **opts)
                    pass
                pass

            # Meaningful limits on JSD
            x, y, ey_stat, ey_syst  = map(np.array, zip(*jsd_limits[var]))
            ex = np.zeros_like(x)
            x[0]  = bins[var][0]
            x[-1] = bins[var][-1]
            format = lambda arr: arr.flatten('C').astype(float)
            gr_stat = ROOT.TGraphErrors(len(x), *list(map(format, [x, y, ex, ey_stat])))
            gr_comb = ROOT.TGraphErrors(len(x), *list(map(format, [x, y, ex, np.sqrt(np.square(ey_stat) + np.square(ey_syst))])))
            smooth_tgrapherrors(gr_stat, ntimes=2)
            smooth_tgrapherrors(gr_comb, ntimes=2)
            c.pads()[col + 1 * nb_col].graph(gr_comb,                                        fillcolor=ROOT.kBlack, alpha=0.03, option='3')
            c.pads()[col + 1 * nb_col].graph(gr_stat, linestyle=2, linecolor=ROOT.kGray + 1, fillcolor=ROOT.kBlack, alpha=0.03, option='L3')

            if col == 0:
                x_, y_, ex_, ey_ = ROOT.Double(0), ROOT.Double(0), ROOT.Double(0), ROOT.Double(0)
                idx = gr_comb.GetN() - 1
                gr_comb.GetPoint(idx, x_,  y_)
                ey_ = gr_comb.GetErrorY(idx)
                x_, y_ = map(float, (x_, y_))
                c.pads()[col + 1 * nb_col].latex("Mean stat. #oplus #varepsilon_{bkg}^{rel} var. limit     ", x_, y_ + 0.75 * ey_, align=31, textsize=11 * scale, angle=0, textcolor=ROOT.kGray + 2)
                pass

            # Decorations
            # -- offsets
            c.pads()[0]._xaxis().SetLabelOffset(9999.)
            c.pads()[0]._xaxis().SetTitleOffset(9999.)
            c.pads()[1]._xaxis().SetLabelOffset(9999.)
            c.pads()[1]._xaxis().SetTitleOffset(9999.)

            c.pads()[2]._xaxis().SetTitleOffset(2.3)
            c.pads()[3]._xaxis().SetTitleOffset(2.3)

            c.pads()[1]._yaxis().SetLabelOffset(9999.)
            c.pads()[1]._yaxis().SetTitleOffset(9999.)
            c.pads()[3]._yaxis().SetLabelOffset(9999.)
            c.pads()[3]._yaxis().SetTitleOffset(9999.)

            # -- x-axis label
            if   var == 'pt':
                xlabel = "Large-#it{R} jet p_{T} [GeV]"
            elif var == 'npv':
                xlabel = "Number of reconstructed vertices N_{PV}"
            else:
                raise NotImplementedError("Variable {} is not supported.".format(var))

            c.pads()[col + 1 * nb_col].xlabel(xlabel)
            if col == 0:
                pattern = "#splitline{#splitline{#splitline{%s}{}}{#splitline{}{}}}{#splitline{#splitline{}{}}{#splitline{}{}}}"
                c.pads()[col + 0 * nb_col].ylabel(pattern % "1/#varepsilon_{bkg}^{rel} @ #varepsilon_{sig}^{rel} = 50%")
                c.pads()[col + 1 * nb_col].ylabel(pattern % "1/JSD @ #varepsilon_{sig}^{rel} = 50%")
                pass

            xmid = (bins[var][0] + bins[var][-1]) * 0.5
            c.pads()[col + 0 * nb_col].latex("Random guessing",   xmid, 2 * 0.9, align=23, textsize=11 * scale, angle=0, textcolor=ROOT.kGray + 2)
            c.pads()[col + 1 * nb_col].latex("Maximal sculpting", xmid, 1 * 0.8, align=23, textsize=11 * scale, angle=0, textcolor=ROOT.kGray + 2)

            c.pads()[col + 0 * nb_col].ylim(1,   70)  # 500
            c.pads()[col + 1 * nb_col].ylim(0.2, 7E+04)  # 2E+05

            c.pads()[col + 0 * nb_col].logy()
            c.pads()[col + 1 * nb_col].logy()

            pass  # end: loop `col`

        # Draw class-specific legend
        width = margin_hori - 0.03
        c.pads()[col + 0 * nb_col].legend(header='MVA:',        width=width, xmin=1. - margin_hori + 0.03, ymax=1. - margin_vert    + 0.02)
        c.pads()[col + 1 * nb_col].legend(header='Analytical:', width=width, xmin=1. - margin_hori + 0.03, ymax=1. - margin_squeeze + 0.02)
        c.pads()[col + 0 * nb_col]._legends[-1].SetTextSize(style.GetLegendTextSize())
        c.pads()[col + 1 * nb_col]._legends[-1].SetTextSize(style.GetLegendTextSize())

        # Common decorations
        for pad in c.pads():
            pad._xaxis().SetNdivisions(504)
            pass

        c.text([], qualifier=QUALIFIER, xmin=margin_hori, ymax=1. - margin_vert + 0.03)

        c.pads()[1].text(["#sqrt{s} = 13 TeV,  #it{W} jet tagging"] + \
                        (['m #in  [60, 100] GeV'] if masscut else []),
                        ATLAS=False, ymax=1. - margin_vert - 0.10)

        c.pads()[3].text(["Multijets"],
                         ATLAS=False, ymax=1. - margin_squeeze - 0.10)

        # Arrows
        c._bare().cd()
        opts_text = dict(textsize=11, textcolor=ROOT.kGray + 2)
        tlatex = ROOT.TLatex()
        tlatex.SetTextAngle(90)
        tlatex.SetTextAlign(22)
        tlatex.SetTextSize(11)
        tlatex.SetTextColor(ROOT.kGray + 2)
        tlatex.DrawLatexNDC(0.5, 0. + 0.5 * (margin_vert + 0.5 * (1.0 - margin_squeeze - margin_vert)), "    Less sculpting #rightarrow")
        tlatex.DrawLatexNDC(0.5, 1. - 0.5 * (margin_vert + 0.5 * (1.0 - margin_squeeze - margin_vert)), "     Greater separation #rightarrow")

        pass  # Temporary style scope

    return c


def plot_individual (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, bins, effs, rejs, jsds, meanx, jsd_limits, masscut = argv

    # To fix 30.5 --> 30 for NPV
    bins['npv'][-1] = np.floor(bins['npv'][-1])

    # Loop combinations
    for var, metric in itertools.product(['pt', 'npv', None], ['rej', 'jsd']):

        with TemporaryStyle() as style:

            # Set styles
            scale      = 1.0
            scale_axis = 0.7
            margin_squeeze = 0.07
            margin_vert    = 0.15
            margin_hori    = 0.17
            size = (350, 300)

            style.SetTextSize(scale_axis * style.GetTextSize())
            for coord in ['x', 'y', 'z']:
                style.SetLabelSize(scale_axis * style.GetLabelSize(coord), coord)
                style.SetTitleSize(scale_axis * style.GetTitleSize(coord), coord)
                pass
            style.SetTitleOffset(1.8, 'y')
            style.SetLegendTextSize(style.GetLegendTextSize() * scale)
            style.SetTickLength(0.05, 'x')
            style.SetTickLength(0.05, 'y')

            # Canvas
            c = rp.canvas(size=size if var is not None else (150, 300), batch=not args.show)

            # Margins
            tpad = c.pad()._bare()
            tpad.SetBottomMargin(margin_vert    if var is not None else 0.49)
            tpad.SetLeftMargin  (margin_hori    if var is not None else 0.49)
            tpad.SetRightMargin (margin_squeeze if var is not None else 0.49)
            tpad.SetTopMargin   (margin_vert    if var is not None else 0.49)

            # Plots
            # -- References
            if var is not None:
                boxopts  = dict(fillcolor=ROOT.kBlack, alpha=0.05, linecolor=ROOT.kGray + 2, linewidth=1, option='HIST')
                c.hist([2 if metric == 'rej' else 1], bins=[bins[var] [0], bins[var] [-1]], **boxopts)

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

                        mean_rej, std_rej = map(np.array, zip(*rejs[var][feat]))  # @TEMP
                        mean_jsd, std_jsd = map(np.array, zip(*jsds[var][feat]))

                        # Only _show_ mass-decorrelated features for `npv`
                        if (var == 'npv') and (ifeat % 2 == 0):
                            mean_rej *= -9999.
                            mean_jsd *= -9999.
                            pass

                        # Error boxes
                        x    = np.array(bins[var][:-1]) + 0.5 * np.diff(bins[var])
                        xerr = 0.5 * np.diff(bins[var])
                        graph_rej = ROOT.TGraphErrors(len(x), x, mean_rej, xerr, std_rej)
                        graph_jsd = ROOT.TGraphErrors(len(x), x, mean_jsd, xerr, std_jsd)

                        if metric == 'rej':
                            c.hist(graph_rej, **opts)
                        else:
                            c.hist(graph_jsd, **opts)
                            pass

                        # Markers and lines
                        opts['option']      = 'PE2L'
                        opts['markerstyle'] = 20 + 4 * (ifeat % 2)

                        graph_rej = ROOT.TGraph(len(x), meanx[var], mean_rej)
                        graph_jsd = ROOT.TGraph(len(x), meanx[var], mean_jsd)

                        if metric == 'rej':
                            c.hist(graph_rej, label=latex(feat, ROOT=True) if not is_simple else None, **opts)
                        else:
                            c.hist(graph_jsd, label=latex(feat, ROOT=True) if     is_simple else None, **opts)
                            pass
                        pass
                    pass   # end loop: `is_simple`

                # Meaningful limits on JSD
                if metric == 'jsd':
                    x, y, ey_stat, ey_syst  = map(np.array, zip(*jsd_limits[var]))
                    ex = np.zeros_like(x)
                    x[0]  = bins[var][0]
                    x[-1] = bins[var][-1]
                    format = lambda arr: arr.flatten('C').astype(float)
                    gr_stat = ROOT.TGraphErrors(len(x), *list(map(format, [x, y, ex, ey_stat])))
                    gr_comb = ROOT.TGraphErrors(len(x), *list(map(format, [x, y, ex, np.sqrt(np.square(ey_stat) + np.square(ey_syst))])))
                    smooth_tgrapherrors(gr_stat, ntimes=2)
                    smooth_tgrapherrors(gr_comb, ntimes=2)
                    c.graph(gr_comb,                                        fillcolor=ROOT.kBlack, alpha=0.03, option='3')
                    c.graph(gr_stat, linestyle=2, linecolor=ROOT.kGray + 1, fillcolor=ROOT.kBlack, alpha=0.03, option='L3')

                    x_, y_, ex_, ey_ = ROOT.Double(0), ROOT.Double(0), ROOT.Double(0), ROOT.Double(0)
                    idx = (gr_comb.GetN() - 1) if var == 'pt' else (gr_comb.GetN() // 2)
                    gr_comb.GetPoint(idx, x_,  y_)
                    ey_ = gr_comb.GetErrorY(idx)
                    x_, y_ = map(float, (x_, y_))
                    if var == 'pt':
                        c.latex("Mean stat. #oplus #varepsilon_{bkg}^{rel} var. limit     ", x_, y_ - 1.0 * ey_, align=31, textsize=11 * scale, angle=0, textcolor=ROOT.kGray + 2)
                        pass
                    pass

                # Decorations
                # -- offsets
                #c.pads()[2]._xaxis().SetTitleOffset(2.3)

                # -- x-axis label
                if   var == 'pt':
                    xlabel = "Large-#it{R} jet p_{T} [GeV]"
                elif var == 'npv':
                    xlabel = "Number of reconstructed vertices N_{PV}"
                elif var is not None:
                    raise NotImplementedError("Variable {} is not supported.".format(var))

                c.xlabel(xlabel)

                # -- y-axis label
                if   metric == 'rej':
                    ylabel = "1/#varepsilon_{bkg}^{rel} @ #varepsilon_{sig}^{rel} = 50%"
                elif metric == 'jsd':
                    ylabel = "1/JSD @ #varepsilon_{sig}^{rel} = 50%"
                else:
                    raise NotImplementedError("Metric {} is not supported.".format(metric))

                c.ylabel(ylabel)

                xmid = (bins[var][0] + bins[var][-1]) * 0.5
                if metric == 'rej':
                    c.latex("Random guessing",   xmid, 2 * 0.9, align=23, textsize=11 * scale, angle=0, textcolor=ROOT.kGray + 2)
                    c.ylim(1,   100)  # 500
                else:
                    c.latex("Maximal sculpting", xmid, 1 * 0.8, align=23, textsize=11 * scale, angle=0, textcolor=ROOT.kGray + 2)
                    c.ylim(0.2, 7E+04)  # 2E+05
                    pass

                c.logy()

                # Common decorations
                c.pad()._xaxis().SetNdivisions(504)

                c.text([], qualifier=QUALIFIER, xmin=margin_hori, ymax=1. - margin_vert + 0.03)

                c.text( ["#sqrt{s} = 13 TeV,  #it{W} jet tagging"] + \
                       (['m #in  [60, 100] GeV'] if masscut else []) + \
                       (['Multijets'] if metric == 'jsd' else []),
                       ATLAS=False, ymax=0.40 if (masscut and (var == 'pt') and (metric == 'rej')) else None)
                       #, ymax=1. - margin_vert - 0.10)

            else:

                # Draw dummy histogram
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
                        opts['option']      = 'PE2L'
                        opts['markerstyle'] = 20 + 4 * (ifeat % 2)

                        label = latex(feat, ROOT=True) if is_simple == (metric == 'jsd') else None
                        h = c.hist([0.5], bins=[0,1], label=label, **opts)
                        pass
                    pass

                # "Remove" axes
                pad = c.pad()
                tpad = pad._bare()
                white = ROOT.kWhite
                pad._xaxis().SetLabelOffset(9999.)
                pad._xaxis().SetTitleOffset(9999.)
                pad._yaxis().SetLabelOffset(9999.)
                pad._yaxis().SetTitleOffset(9999.)
                pad._xaxis().SetAxisColor  (white)  # Remove "double ticks"
                pad._yaxis().SetAxisColor  (white)  # Remove "double ticks"
                tpad.SetFillColor          (white)
                tpad.SetFrameFillColor     (white)
                c._bare().SetFillColor     (white)
                c._bare().SetFrameFillColor(white)

                # Draw class-specific legend
                width = 0.90 #margin_hori - 0.03
                if var is None:
                    if metric == 'rej':
                        c.legend(header='MVA:',        width=width, xmin=0.05, ymax=1. - margin_vert + 0.02)  # xmin = margin_hori + 0.03
                    else:
                        c.legend(header='Analytical:', width=width, xmin=0.05, ymax=1. - margin_vert + 0.02)
                        pass
                    c.pad()._legends[-1].SetTextSize(style.GetLegendTextSize())
                    pass
                pass
            pass

            # Arrows
            '''
            c._bare().cd()
            opts_text = dict(textsize=11, textcolor=ROOT.kGray + 2)
            tlatex = ROOT.TLatex()
            tlatex.SetTextAngle(90)
            tlatex.SetTextAlign(22)
            tlatex.SetTextSize(11)
            tlatex.SetTextColor(ROOT.kGray + 2)
            tlatex.DrawLatexNDC(0.5, 0. + 0.5 * (margin_vert + 0.5 * (1.0 - margin_squeeze - margin_vert)), "    Less sculpting #rightarrow")
            tlatex.DrawLatexNDC(0.5, 1. - 0.5 * (margin_vert + 0.5 * (1.0 - margin_squeeze - margin_vert)), "     Greater separation #rightarrow")
            '''

            # Save
            c.save('figures/robustness__{}_{}{}.pdf'.format(var if var is not None else 'legend', metric if var is not None else ('mva' if metric == 'rej' else 'analytical'), '_masscut' if masscut else ''))

            pass  # Temporary style scope

        pass
    return


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    data, args, features, bins, effs, rejs, jsds, meanx, jsd_limits, masscut, var = argv

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
        # -- References
        boxopts  = dict(fillcolor=ROOT.kBlack, alpha=0.05, linecolor=ROOT.kGray + 2, linewidth=1, option='HIST')
        c.pads()[0].hist([2], bins=[bins[0], bins[-1]], **boxopts)
        c.pads()[1].hist([1], bins=[bins[0], bins[-1]], **boxopts)


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

                mean_rej, std_rej = map(np.array, zip(*rejs[feat]))  # @TEMP
                #mean_rej, std_rej = map(np.array, zip(*effs[feat]))  # @TEMP
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
        c.pads()[1].legend(header='Analytical:', width=width, xmin=0.79, ymax=0.975)

        # Meaningful limits on JSD
        x, y, ey_stat, ey_syst  = map(np.array, zip(*jsd_limits))
        ex = np.zeros_like(x)
        x[0] = bins[0]
        x[-1] = bins[-1]
        format = lambda arr: arr.flatten('C').astype(float)
        gr_stat = ROOT.TGraphErrors(len(x), *list(map(format, [x, y, ex, ey_stat])))
        gr_comb = ROOT.TGraphErrors(len(x), *list(map(format, [x, y, ex, np.sqrt(np.square(ey_stat) + np.square(ey_syst))])))
        smooth_tgrapherrors(gr_stat, ntimes=2)
        smooth_tgrapherrors(gr_comb, ntimes=2)
        c.pads()[1].graph(gr_comb,                                        fillcolor=ROOT.kBlack, alpha=0.03, option='3')
        c.pads()[1].graph(gr_stat, linestyle=2, linecolor=ROOT.kGray + 1, fillcolor=ROOT.kBlack, alpha=0.03, option='L3')

        x_, y_, ex_, ey_ = ROOT.Double(0), ROOT.Double(0), ROOT.Double(0), ROOT.Double(0)
        idx = gr_comb.GetN() - 1
        gr_comb.GetPoint(idx, x_,  y_)
        ey_ = gr_comb.GetErrorY(idx)
        x_, y_ = map(float, (x_, y_))
        c.pads()[1].latex("Mean stat. #oplus #varepsilon_{bkg}^{rel} var. limit     ", x_, y_ + ey_, align=31, textsize=11, angle=0, textcolor=ROOT.kGray + 2)

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
        c.pads()[0].ylabel("1/#varepsilon_{bkg}^{rel} @ #varepsilon_{sig}^{rel} = 50%")
        c.pads()[1].ylabel("1/JSD @ #varepsilon_{sig}^{rel} = 50%")

        xmid = (bins[0] + bins[-1]) * 0.5
        c.pads()[0].latex("Random guessing",   xmid, 2 * 0.9, align=23, textsize=11, angle=0, textcolor=ROOT.kGray + 2)
        c.pads()[1].latex("Maximal sculpting", xmid, 1 * 0.8, align=23, textsize=11, angle=0, textcolor=ROOT.kGray + 2)

        c.text([], qualifier=QUALIFIER, xmin=0.15, ymax=0.93)

        c.text(["#sqrt{s} = 13 TeV,  #it{W} jet tagging"] + \
                (['m #in  [60, 100] GeV'] if masscut else []),
                 ATLAS=False, ymax=0.76)

        c.pads()[1].text(["Multijets"], ATLAS=False)

        c.pads()[0].ylim(1, 500)
        c.pads()[1].ylim(0.2, 2E+05)

        c.pads()[0].logy()
        c.pads()[1].logy()

        pass  # Temporary style scope

    return c
