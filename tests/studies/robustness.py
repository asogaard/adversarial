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
    effs, rejs, jsds = {feat: [] for feat in features}, \
                       {feat: [] for feat in features}, \
                       {feat: [] for feat in features}
    meanx = list()

    # For reproducibility of bootstrap sampling
    np.random.seed(7)

    # Get fixed, inclusive effsig = 50% cut(s)
    '''
    cuts = dict()
    for feat in features:
        # @NOTE: This is _horrible_ code duplication...
        # (Opt.) mass cut mask
        if masscut:
            print "metrics: Applying mass cut."
            pass
        msk = (data['m'] > 60.) & (data['m'] < 100.) if masscut else np.ones_like(data['signal']).astype(bool)
        
        # scikit-learn assumes signal towards 1, background towards 0
        pred = data[feat].values.copy()
        if signal_low(feat):
            pred *= -1.
            pass
        
        # Compute ROC curve efficiencies
        fpr, tpr, thresholds = roc_curve(data.loc[msk, 'signal'], pred[msk], sample_weight=data.loc[msk, 'weight_test'])
        
        if masscut:
            tpr_mass = np.mean(msk[data['signal'] == 1])
            fpr_mass = np.mean(msk[data['signal'] == 0])
            
            tpr *= tpr_mass
            fpr *= fpr_mass
            pass
        
        # Get background rejection factor
        idx = np.argmin(np.abs(tpr - 0.5))
        cuts[feat] = thresholds[idx]
        pass
        '''

    # Scan `var`
    jsd_limits = list()
    for bin in zip(bins[:-1], bins[1:]):

        # Perform selection
        msk_bin  = (data[var] >= bin[0]) & (data[var] < bin[1])
        msk_bkg  = data['signal'] == 0

        # Compute weighted mean if x-axis variable
        meanx.append(wmean(data.loc[msk_bin & msk_bkg, var], data.loc[msk_bin & msk_bkg, 'weight_test']))

        # Compute bootstrapped metrics for all features
        for feat in features:
            mean_std_eff, mean_std_rej, mean_std_jsd = bootstrap_metrics(data[msk_bin], feat, num_bootstrap=num_bootstrap, masscut=masscut)

            # Store in output containers
            effs[feat].append(mean_std_eff)
            rejs[feat].append(mean_std_rej)
            jsds[feat].append(mean_std_jsd)
            pass

        # Compute meaningful limit on JSD
        bin_bkgeffs = [1./rejs[feat][-1][0] for feat in features]
        sigmoid = lambda x: 1. / (1. + np.exp(-x))
        limits_min  = map(lambda jsd: 1./jsd, jsd_limit(data[msk_bin & msk_bkg], np.min (bin_bkgeffs), num_bootstrap=num_bootstrap))
        limits_mean = map(lambda jsd: 1./jsd, jsd_limit(data[msk_bin & msk_bkg], np.mean(bin_bkgeffs), num_bootstrap=num_bootstrap))
        limits_max  = map(lambda jsd: 1./jsd, jsd_limit(data[msk_bin & msk_bkg], np.max (bin_bkgeffs), num_bootstrap=num_bootstrap))
        jsd_limits.append((meanx[-1], np.mean(limits_mean), np.std(limits_max), np.abs(np.mean(limits_min) - np.mean(limits_max)) / 2.))
        pass

    # Format array
    meanx = np.array(meanx).astype(float)

    # Perform plotting
    c = plot(data, args, features, bins, effs, rejs, jsds, meanx, jsd_limits, masscut, var)

    # Output
    path = 'figures/robustness_{}{}.pdf'.format(var, '_masscut' if masscut else '')

    return c, args, path


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
        c.pads()[1].legend(header='Simple:', width=width, xmin=0.79, ymax=0.975)

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
        c.pads()[1].latex("Mean stat. #oplus #varepsilon_{bkg} var. limit     ", x_, y_ + ey_, align=31, textsize=11, angle=0, textcolor=ROOT.kGray + 2)

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

        xmid = (bins[0] + bins[-1]) * 0.5
        c.pads()[0].latex("Random guessing",   xmid, 2 * 0.9, align=23, textsize=11, angle=0, textcolor=ROOT.kGray + 2)
        c.pads()[1].latex("Maximal sculpting", xmid, 1 * 0.8, align=23, textsize=11, angle=0, textcolor=ROOT.kGray + 2)

        c.text([], qualifier=QUALIFIER, xmin=0.15, ymax=0.93)

        c.text(["#sqrt{s} = 13 TeV,  #it{W} jet tagging"] + \
                (['m #in  [60, 100] GeV'] if masscut else []),
                 ATLAS=False, ymax=0.76)

        c.pads()[1].text(["QCD jets"], ATLAS=False)

        c.pads()[0].ylim(1, 500)
        c.pads()[1].ylim(0.2, 2E+05)

        c.pads()[0].logy()
        c.pads()[1].logy()

        pass  # Temporary style scope

    return c
