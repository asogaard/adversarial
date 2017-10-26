#!/usr/bin/env python

"""..."""

# Basic import(s)
# ...

# Scientific import(s)
import numpy as np
import root_numpy
import ROOT
import matplotlib.pyplot as plt

# Custom import(s)
ROOTPLOTTING=False
try:
    from rootplotting import ap
    ROOTPLOTTING=True
except:
    # RootPlotting not available
    pass

# Project import(s)
from adversarial.profile import *
from adversarial.utils   import *

# Global variables
linestyles = ['-', '--', '-.', ':']
colours    = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
root_colours = [ROOT.kPink - 2, ROOT.kAzure + 7, ROOT.kGreen - 7, ROOT.kOrange + 7, ROOT.kViolet + 7, ROOT.kGray + 3]


def plot_jetmass (data, args, var, cut_value=None, cut_eff=None, name='tagger_jetmass', title=''):
    """..."""

    # Check(s)
    assert (cut_value is None) != (cut_eff is None), "Please specify exactly one of `cut_value` and `cut_eff`"

    # Get sequential blues
    blues = plt.get_cmap('Blues')

    # Get cut direction
    if wmean(data.signal[var], data.signal.weights) > wmean(data.background[var], data.background.weights):
        direction = ">"
    else:
        direction = "<"
        pass

    if cut_value is None:
        if isinstance(cut_eff, (list,tuple)):
            cut_value = list()
            for eff in cut_eff:
                assert eff < 1.
                assert eff > 0.
                print "Computing cut value for {:.1f}% signal efficiency".format(eff * 100.)
                if direction == ">":
                    cut_value.append(wpercentile(data.signal[var], (1. - eff) * 100., data.signal.weights))
                else:
                    cut_value.append(wpercentile(data.signal[var],       eff  * 100., data.signal.weights))
                    pass
                pass
        else:
            assert cut_eff < 1.
            assert cut_eff > 0.
            print "Computing cut value for {:.1f}% signal efficiency".format(cut_eff * 100.)
            if direction == ">":
                cut_value = [wpercentile(data.signal[var], (1. - cut_eff) * 100., data.signal.weights)]
            else:
                cut_value = [wpercentile(data.signal[var],       cut_eff  * 100., data.signal.weights)]
                pass
            pass
    else:
        if isinstance(cut_value, (list,tuple)):
            cut_eff = list()
            for value in cut_value:
                if direction == ">":
                    cut_eff.append(np.sum(data.signal.weights[var] > value) / np.sum(data.signal.weights))
                else:
                    cut_eff.append(np.sum(data.signal.weights[var] < value) / np.sum(data.signal.weights))
                    pass
                pass
        else:
            if direction == ">":
                cut_eff = [np.sum(data.signal.weights[var] > cut_value) / np.sum(data.signal.weights)]
            else:
                cut_eff = [np.sum(data.signal.weights[var] < cut_value) / np.sum(data.signal.weights)]
                pass
            pass
        pass

    # Create figure
    fig, ax = plt.subplots()

    # Get axis limits
    edges = np.linspace(0, 300, 60 + 1, endpoint=True)


    # Plot pre-cut distribution
    plt.hist(data.background['m'], bins=edges, weights=data.background.weights,
             alpha=1.0, normed=True, label='Before cut')

    # Loop cut values.
    for icut, (value, eff) in enumerate(zip(cut_value, cut_eff)):
        print "Using {} cut value {:.3f}".format(var, value)

        # Select background jets passing substructure cut
        if direction == ">":
            msk_pass = data.background[var] > value
        else:
            msk_pass = data.background[var] < value
            pass

        # Plot post-cut distribution
        color = blues(1 - float(icut + 1)/(len(cut_value) + 1))
        plt.hist(data.background['m'][msk_pass], bins=edges, weights=data.background.weights[msk_pass], histtype='step', color=color, # ... colours[1],
                 alpha=0.7, linewidth=1.2, normed=True, label=r'After cut (%s %s %.2f; $\varepsilon_{sig.} = %.0f$%% )' % (latex(var), direction, value, eff * 100.))
        pass

    # Decorations
    plt.xlabel(r"Large-radius jet mass [GeV]",
               horizontalalignment='right', x=1.0)

    plt.ylabel("Jets / {:.1f} GeV (normalised)".format(np.diff(edges)[0]),
               horizontalalignment='right', y=1.0)

    plt.yscale('log')
    plt.ylim(1E-05, 1E+00)
    plt.title(r"Jet mass spectra for successive cuts on {}".format(latex(var)), fontweight='medium')
    plt.legend()

    # Save figure
    plt.savefig(args.output + '{}__{}.pdf'.format(name, var))

    # Close figure
    plt.close()

    return


def plot_jetmass_comparison (data, args, cut_eff=0.5, name='tagger_jetmass_comparison', title=''):
    """..."""

    # Create figure
    fig, ax = plt.subplots()

    # Get axis limits
    edges = np.linspace(50, 300, 50 + 1, endpoint=True)

    # Plot pre-cut distribution
    plt.hist(data.background['m'], bins=edges, weights=data.background.weights,
             alpha=1.0, normed=True, label='Before cut')

    # Plot post-cut distributionsa
    for ivar, var in enumerate(['Tau21', 'Tau21DDT_1', 'NN', 'ANN']):

        # Get cut direction
        if wmean(data.signal[var], data.signal.weights) > wmean(data.background[var], data.background.weights):
            direction = ">"
        else:
            direction = "<"
            pass

        # Get cut value
        if direction == ">":
            value = wpercentile(data.signal[var], (1. - cut_eff) * 100., data.signal.weights)
        else:
            value = wpercentile(data.signal[var],       cut_eff  * 100., data.signal.weights)
            pass

        # Select background jets passing substructure cut
        if direction == ">":
            msk_pass = data.background[var] > value
        else:
            msk_pass = data.background[var] < value
            pass

        # Plot post-cut distribution
        icol = ivar + 1 if ivar < 2 else ivar + 2 # @TEMP: ... + 3
        plt.hist(data.background['m'][msk_pass], bins=edges, weights=data.background.weights[msk_pass], histtype='step', color=colours[icol],
                 linewidth=2.0, normed=True, label=r'After cut (%s %s %.2f; $\varepsilon_{sig.} = %.0f$%% )' % (latex(var), direction, value, cut_eff * 100.))
        pass

    # Decorations
    plt.plot([80.5, 80.5], [2E-05, 2E-01], '--', color='gray', linewidth=1.0)
    plt.text(83., 2E-01, "W pole mass", ha='left', va='top', fontsize='small', color='gray', fontstyle='italic', fontweight='medium')
    plt.xlabel(r"Large-radius jet mass [GeV]",
               horizontalalignment='right', x=1.0)

    plt.ylabel("Jets / {:.1f} GeV (normalised)".format(np.diff(edges)[0]),
               horizontalalignment='right', y=1.0)

    plt.yscale('log')
    plt.ylim(2E-05, 2E-01)
    plt.legend()

    # Save figure
    plt.savefig(args.output + '{}.pdf'.format(name))

    # Close figure
    plt.close()


    # RootPlotting version
    # --------------------------------------------------------------------------
    if ROOTPLOTTING:

        # Create canvas
        c = ap.canvas(batch=True, size=(700, 500))
        
        # Plot pre-cut distribution
        c.hist(data.background['m'], bins=edges, weights=data.background.weights, fillcolor=root_colours[0], normalise=True, label="Before cut")
        
        
        # Plot post-cut distributionsa
        for ivar, var in enumerate(['Tau21', 'Tau21DDT_1', 'NN', 'ANN']):
            
            # Get cut direction
            if wmean(data.signal[var], data.signal.weights) > wmean(data.background[var], data.background.weights):
                direction = ">"
            else:
                direction = "<"
                pass
            
            # Get cut value
            if direction == ">":
                value = wpercentile(data.signal[var], (1. - cut_eff) * 100., data.signal.weights)
            else:
                value = wpercentile(data.signal[var],       cut_eff  * 100., data.signal.weights)
                pass
            
            # Select background jets passing substructure cut
            if direction == ">":
                msk_pass = data.background[var] > value
            else:
                msk_pass = data.background[var] < value
                pass
            
            # Plot post-cut distribution
            icol = ivar // 2 + 1 # if ivar < 2 else ivar + 2
            c.hist(data.background['m'][msk_pass], bins=edges, weights=data.background.weights[msk_pass],
                   linecolor=root_colours[icol], linewidth=2, linestyle=ivar % 2 + 2,
                   markercolor=root_colours[icol], markerstyle=20+ivar, markersize=1.0,
                   option="HIST HP", legend_option="LP",
                   normalise=True, label=(r'%s %s %.2f' % (latex(var), direction, value)).replace("\\", "#").replace("$", ""))
            pass
        
        # Decorations
        c.xlabel("Large-#it{R} jet mass [GeV]")
        c.ylabel("Fraction of jets")
        c.ymin(1E-05)
        c.logy()
        c.xline(80.5, ymin=1E-05, linestyle=2, linewidth=1, linecolor=ROOT.kGray + 3, text='#it{W} pole mass', text_align='BR', textsize=17)
        c.text(["#sqrt{s} = 13 TeV",
                "W-tagging MC",
                "Fixed 50% signal efficiency cuts"],
               qualifier="Work in progress")
        c.legend(xmin=0.65, width=0.25)
        
        # Save figure
        c.save(args.output + 'rp__' + '{}.pdf'.format(name))
        pass

    return


def plot_distribution (data, args, var, name='tagger_distribution', title=''):
    """..."""

    # Create figure
    fig, ax = plt.subplots()

    # Get axis limits
    if var == 'D2':
        edges = np.linspace(0, 5, 50 + 1, endpoint=True)
    else:
        edges = np.linspace(0, 1, 50 + 1, endpoint=True)
        pass

    # Plot distributions
    msk = np.isfinite(data.background[var])
    plt.hist(data.background[var][msk], bins=edges, weights=data.background.weights[msk],
             alpha=0.5, normed=True, label='Background')

    msk = np.isfinite(data.signal[var])
    plt.hist(data.signal    [var][msk], bins=edges, weights=data.signal    .weights[msk],
             alpha=0.5, normed=True, label='Signal')


    # Decorations
    plt.xlabel(r"Jet {}".format(latex(var)),
               horizontalalignment='right', x=1.0)

    plt.ylabel("Jets / {:.3f} (normalised)".format(np.diff(edges)[0]),
               horizontalalignment='right', y=1.0)

    plt.legend()

    # Save figure
    plt.savefig(args.output + '{}__{}.pdf'.format(name, var))

    # Close figure
    plt.close()

    return


def plot_roc (data, args, vars, name='tagger_ROCs', title=''):
    """...."""

    # Create figure
    fig, ax = plt.subplots(figsize=(5,5))

    # Plot random-guess line
    plt.plot([0,1], [0,1], 'k--', linewidth=1.0, alpha=0.2)

    # Get and plot ROC curves
    for ivar, var in enumerate(vars):

        # Get format
        linestyle, color = None, None
        if var.startswith('Tau21DDT'):
            idx_lst = int(var[-1]) - 1
            idx_col = len(filter(lambda var: not var.startswith('Tau21DDT'), vars)) # -1
            linestyle = linestyles[idx_lst]
            color     = colours   [idx_col]
            pass

        # Compute selection efficiencies
        eff_sig, eff_bkg = roc_efficiencies(data.signal    [var],
                                            data.background[var],
                                            data.signal    .weights,
                                            data.background.weights)

        # Compute ROC AUC
        try:
            auc = roc_auc(eff_sig, eff_bkg)
        except: # Efficiencies not monotonically increasing
            auc = 0.
            pass

        # Plot ROC curve
        ax.plot(eff_bkg, eff_sig, linestyle=linestyle, color=color, label='{} (AUC: {:.3f})'.format(latex(var), auc))
        pass

    # Decorations
    plt.xlabel("Background efficiency", horizontalalignment='right', x=1.0)
    plt.ylabel("Signal efficiency",     horizontalalignment='right', y=1.0)
    plt.legend()

    # Save figure
    plt.savefig(args.output + '{}.pdf'.format(name))

    # Cloase figure
    plt.close()

    return



def plot_profiles (data, args, var, name='tagger_profile', title=''):
    """Plot percentile profiles of tagger varaibles versus jet mass.

    Args:
        data: Dict containing all relevant data: `X`, `Y`, `P`, `W`, `sig`, and
            `bkg`.
        args: Namespace containing command-line arguments.
        var: Name of tagger variable, to be found in `data['bkg']`, _or_ a Keras
            model assumed to be the classifier to be profiled and to take
            `data['X']` as input.
        name: Name of file to which to save the figure (.pdf suffix appended
            automatically).
        title: Figure (sub-)title.
    """

    # Create figure
    fig, ax = plt.subplots()

    direction = ">" if 'NN' in var else "<"

    # Plotting variables
    edges = np.linspace(0, 300, 60 + 1, endpoint=True)
    bins  = edges[:-1] + 0.5 * np.diff(edges)
    step = 10.
    percentiles = np.linspace(step, 100 - step, int(100 / step) - 1, endpoint=True)
    profiles = [[] for _ in percentiles]
    num_bootstrap = 10
    classifier = None

    # Get tagger variable array
    if isinstance(var, str):
        tagger = data[var]
    else:
        # Assume `var` is a Keras model describing a classifier taking
        # `data['X']` as input
        classifier, var = var, var.name
        tagger = classifier.predict(data.inputs, batch_size=2048).flatten()
        pass

    msk = np.isfinite(tagger)

    tagger = tagger[msk]
    masses = data['m'][msk]
    weight = data['weight'][msk]

    rho = wcorr(masses, tagger, weight)
    print "Linear correlation coeff. for {} vs. jet mass: {:3f}".format(var, rho)

    # Loop mass bins
    errorbands = list()
    for (mass_down, mass_up) in zip(edges[:-1], edges[1:]):

        # Get array of `var` within the current jet-mass band
        msk = (masses >= mass_down) & (masses < mass_up)
        arr_tagger = tagger[msk]
        arr_weight = weight[msk]

        # Perform bootstrapping of the tagger variable array to estimate error
        # bands on percentile contours.
        # Cf. e.g. [https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf]
        bootstrap_percentiles = [[] for _ in percentiles]
        num_samples = len(arr_tagger)
        for bts in range(num_bootstrap):

            if num_samples == 0:
                indices = []
            else:
                indices = np.random.choice(num_samples, num_samples, replace=True)
                pass

            bts_tagger = arr_tagger[indices]
            bts_weight = arr_weight[indices]

            # Loop percentiles for current mass bins
            for idx, perc in enumerate(percentiles):
                if len(bts_tagger) == 0:
                    bootstrap_percentiles[idx].append(np.nan)
                else:
                    if direction == "<":
                        bootstrap_percentiles[idx].append(wpercentile(bts_tagger,         perc,  bts_weight))
                    else:
                        bootstrap_percentiles[idx].append(wpercentile(bts_tagger, (100. - perc), bts_weight))
                        pass
                    pass
                pass
            pass

        errorbands.append(np.std(bootstrap_percentiles, axis=1))

        # Loop percentiles
        for idx, perc in enumerate(percentiles):
            if len(arr_tagger) == 0:
                profiles[idx].append(np.nan)
            else:
                if direction == "<":
                    profiles[idx].append(wpercentile(arr_tagger,         perc,  arr_weight))
                else:
                    profiles[idx].append(wpercentile(arr_tagger, (100. - perc), arr_weight))
                    pass
                pass
            pass # end: loop percentiles
        pass # end: loop mass bins

    errorbands = list(np.array(errorbands).T)

    # Plot profile
    for profile, error, perc in zip(profiles, errorbands, percentiles):
        plt.plot(bins, profile, color=colours[0], linewidth=2 if perc == 50 else 1, label='Median' if perc == 50 else None)
        plt.fill_between(bins, profile - error, profile + error, color=colours[0], alpha=0.1 if perc == 50 else 0.05) #, label='Bootstr. RMS' if perc == 50 else None)
        pass

    # Plot mean profile with error bars
    profile = ROOT.TProfile('profile', "", len(bins), edges)
    root_numpy.fill_profile(profile, np.vstack((masses, tagger)).T, weight)

    means, rmses = list(), list()
    for i in range(1, 1 + len(bins)):
        mean = profile.GetBinContent(i)
        rms  = profile.GetBinError  (i)
        if mean == 0 and rms == 0:
            mean = np.nan
            rms  = np.nan
            pass
        means.append(mean)
        rmses.append(rms)
        pass

    binwidths = np.diff(edges) / 2.
    plt.errorbar(bins, means, xerr=binwidths, yerr=rmses, fmt='k.', color='black', label=r'Mean $\pm$ RMS')

    # Text
    mid = len(percentiles) // 2

    arr_profiles = np.array(profiles).flatten()
    arr_profiles = arr_profiles[~np.isnan(arr_profiles)]
    diff = np.max(arr_profiles) - np.min(arr_profiles)

    opts = dict(horizontalalignment='center', verticalalignment='bottom', fontsize='x-small')
    text_string = r"$\varepsilon_{bkg.}$ = %d%%"

    idx = -1 if direction == "<" else 0
    plt.text(edges[-1], profiles[idx] [-1] + 0.02 * diff, text_string % percentiles[idx],  **opts) # 90%

    opts = dict(horizontalalignment='left', verticalalignment='center', fontsize='x-small')
    text_string = "%d%%"

    idx = mid
    plt.text(edges[-1], profiles[idx][-1], text_string % percentiles[idx], **opts) # 50%

    idx = 0 if direction == "<" else -1
    plt.text(edges[-1], profiles[idx][-1], text_string % percentiles[idx],   **opts) # 10%

    plt.plot([80.5, 80.5], [-0.05, 1.05], '--', color='gray', linewidth=1.0)
    plt.text(83., 1.05, "W pole mass", ha='left', va='top', fontsize='small', color='gray', fontstyle='italic', fontweight='medium')

    # Decorations
    plt.xlabel("Large-radius jet mass [GeV]",  horizontalalignment='right', x=1.0)
    plt.ylabel(r"{}".format(latex(var)),  horizontalalignment='right', y=1.0)
    #plt.title(r'Percentile profiles for {}{}'.format(latex(var), (': ' + title) if title else ''), fontweight='medium')
    plt.legend()
    #plt.xlim(edges[0], edges[-1])
    plt.xlim(0, 320)
    if classifier is not None:
        plt.ylim(-0.05, 1.05)
        pass
    plt.ylim(-0.05, 1.05) # @TEMP

    # Save figure
    plt.savefig(args.output + '{}__{}.pdf'.format(name, var))

    # Close figure
    plt.close()


    # RootPlotting version
    # --------------------------------------------------------------------------
    if ROOTPLOTTING:
        
        # Create canvas
        c = ap.canvas(batch=True, size=(700, 500))
        
        # Plot percentile profiles
        first = True
        for profile, error, perc in zip(profiles, errorbands, percentiles):
            msk = np.isfinite(profile) & np.isfinite(error)
            graph = ROOT.TGraphErrors(sum(msk),
                                      np.array(bins)     .flatten('C').astype(float)[msk],
                                      np.array(profile)  .flatten('C').astype(float)[msk],
                                      np.array(binwidths).flatten('C').astype(float)[msk],
                                      np.array(error)    .flatten('C').astype(float)[msk])
            
            c.graph(graph, linecolor=root_colours[0], fillcolor=root_colours[0], markerstyle=0, alpha=0.2 if perc == 50 else 0.1, linewidth=2 if perc == 50 else 1, option=('A' if first else '') + 'L3', label="Median background eff." if perc == 50 else ("Background efficiency deciles" if perc == 10 else None), legend_option="FL")
            
            padx, pady = 2, 0.01
            idx = list(bins).index(287.5)
            if   ((perc == 10) and (direction == ">")) or ((perc == 90) and (direction == "<")):
                c.latex("#varepsilon_{bkg.} = %d%%" % perc,
                        287.5, profile[idx] + 0.02, align=31, textsize=14, textcolor=ROOT.kGray+3)
                pass
            # ...
            
            first = False
            pass
        
        # Plot mean profile
        graph = ROOT.TGraphErrors(len(bins),
                                  np.array(bins)     .flatten('C').astype(float),
                                  np.array(means)    .flatten('C').astype(float),
                                  np.array(binwidths).flatten('C').astype(float),
                                  np.array(rmses)    .flatten('C').astype(float))
        c.graph(graph, markerstyle=20, label="Mean #pm RMS")
        
        # Decoration
        c.xlabel("Large-#it{R} jet mass [GeV]")
        c.ylabel(latex(var).replace("$", "").replace("\\", "#") + " classifier output")
        c.xlim(50, 300)
        c.ylim(0, 1.6)
        c.legend(xmin=0.45)
        c.xline(80.5, ymax=1.0, linestyle=2, linewidth=1, linecolor=ROOT.kGray + 3, text='#it{W} pole mass', text_align='TR', textsize=17)
        
        c.text(["#sqrt{s} = 13 TeV",
                "W-tagging MC"],
               qualifier="Work in progress")
        
        # Save figure
        c.save(args.output + 'rp__{}__{}.pdf'.format(name, var))
        pass

    return


def plot_posterior (data, args, adversary, name='posterior', title=''):
    """..."""

    # @TODO:
    # - Documentation
    # - Proper treatment of > 1 de-correlation variables

    # Create figure
    fig, ax = plt.subplots()

    # Variable definitions
    edges  = np.linspace(-0.2, 1.2, 2 * 70 + 1, endpoint=True)

    z_slices  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #P2_slices = [0, 1]
    P1        = np.linspace(0, 1, 1000 + 1, endpoint=True)

    # Plot prior
    msk = (data.targets == 0)
    plt.hist(data.decorrelation[:,0][msk], bins=edges, weights=data.weights[msk], normed=True, color='gray', alpha=0.5, label='Prior')

    # Plot adversary posteriors
    # for i, P2_slice in enumerate(P2_slices):
    # P2 = np.ones_like(P1) * P2_slice
    # P_test = np.vstack((P1, P2)).T
    P_test = P1
    for j, z_slice in enumerate(z_slices):
        z_test = np.ones_like(P1) * z_slice
        posterior = adversary.predict([z_test, P_test])
        plt.plot(P1, posterior, color=colours[j], label='clf. = %.1f' % z_slices[j])
        pass

    # Decorations
    plt.xlabel("Normalised jet log(m)", horizontalalignment='right', x=1.0)
    plt.ylabel("Jets",                  horizontalalignment='right', y=1.0)
    plt.title("De-correlation p.d.f.'s{}".format((': ' + title) if title else ''), fontweight='medium')
    plt.ylim(0, 5.)
    plt.legend()

    # Save figure
    plt.savefig(args.output + name + '.pdf')

    # Close figure
    plt.close(fig)

    return


def plot_decorrelation (data, args, name='decorrelation_profile', title='', fit_range=None):
    """..."""

    # Define rhoDDT bin edges
    edges = np.linspace(-2, 6, 64 + 1, endpoint=True)

    # Fill ROOT TProfile
    prof = ROOT.TProfile('tau21_profile', "", len(edges) - 1, edges)
    root_numpy.fill_profile(prof, np.vstack((data['rhoDDT'], data['Tau21'])).T, data.weights)

    # Perform linear fit
    if fit_range is None:
        intercept, slope = None, None
    else:
        fit = ROOT.TF1('fit', 'pol1', *fit_range)
        prof.Fit('fit', 'RQ0')
        intercept, slope = fit.GetParameter(0), fit.GetParameter(1)
        pass

    # Fill transformed ROOT TProfile
    tau21DDT = data['Tau21'] - (data['rhoDDT'] - fit_range[0]) * slope
    profDDT = ROOT.TProfile('tau21DDT_profile', "", len(edges) - 1, edges)
    root_numpy.fill_profile(profDDT, np.vstack((data['rhoDDT'], tau21DDT)).T, data.weights)

    # Create arrays from profile
    arr_x, arr_y1, arr_y2, arr_ex, arr_ey1, arr_ey2 = list(), list(), list(), list(), list(), list()
    for ibin in range(1, prof.GetXaxis().GetNbins() + 1):
        if prof.GetBinContent(ibin) != 0. or prof.GetBinError(ibin) != 0.:
            arr_x  .append(prof   .GetBinCenter (ibin))
            arr_y1 .append(prof   .GetBinContent(ibin))
            arr_y2 .append(profDDT.GetBinContent(ibin))
            arr_ex .append(prof   .GetBinWidth  (ibin) / 2.)
            arr_ey1.append(prof   .GetBinError  (ibin))
            arr_ey2.append(profDDT.GetBinError  (ibin))
            pass
        pass

    xmin, xmax = fit_range

    # Create figure
    fig, ax = plt.subplots()

    # Plot profile
    plt.errorbar(arr_x, arr_y1, xerr=arr_ex, yerr=arr_ey1, fmt='.', label=r"Original profile, $\tau_{21}$")
    plt.errorbar(arr_x, arr_y2, xerr=arr_ex, yerr=arr_ey2, fmt='s', label=r"Transformed profile, $\tau_{21}^{DDT}$")

    # (Opt.) plot fit
    if intercept is not None:
        x1, y1 = xmin, intercept + xmin * slope
        x2, y2 = xmax, intercept + xmax * slope
        plt.plot([x1,x2], [y1,y2], color=colours[-1], label='Linear fit', zorder=10)
        pass

    # Decoration
    plt.xlabel(r"Large-radius jet $\rho^{DDT}$ = log($m^{2}$ / $p_{T}$ / 1 GeV)", horizontalalignment='right', x=1.0)
    plt.ylabel(r"$\langle\tau_{21}\rangle, \langle\tau_{21}^{DDT}\rangle$",     horizontalalignment='right', y=1.0)
    plt.xlim(-0.5, 5.5)
    plt.ylim(0.2, .8)
    #plt.title(title, fontweight='medium')
    if intercept is not None:
        plt.legend(loc=8)
        pass

    plt.text(0.03, 0.95, "ATLAS",
             weight='bold', style='italic', size='large',
             ha='left', va='top',
             transform=ax.transAxes)
    plt.text(0.17, 0.9425, "Simulation",
             ha='left', va='top',
             transform=ax.transAxes)

    # Save figure
    plt.savefig(args.output + '{}.pdf'.format(name))

    # Close figure
    plt.close()


    # RootPlotting version
    # --------------------------------------------------------------------------
    if ROOTPLOTTING:
        
        # Create canvas
        c = ap.canvas(batch=True, size=(700, 500))
        
        # Compute TGraphs with error bars
        g1 = ROOT.TGraphErrors(len(arr_x),
                               np.array(arr_x)  .flatten('C').astype(float),
                               np.array(arr_y1) .flatten('C').astype(float),
                               np.array(arr_ex) .flatten('C').astype(float),
                               np.array(arr_ey1).flatten('C').astype(float))
        g2 = ROOT.TGraphErrors(len(arr_x),
                               np.array(arr_x)  .flatten('C').astype(float),
                               np.array(arr_y2) .flatten('C').astype(float),
                               np.array(arr_ex) .flatten('C').astype(float),
                               np.array(arr_ey2).flatten('C').astype(float))
        
        # Plot profile
        c.graph(g1, linecolor=root_colours[0], markercolor=root_colours[0], markerstyle=20, label="Original profile, #tau_{21}")
        c.graph(g2, linecolor=root_colours[1], markercolor=root_colours[1], markerstyle=21, label="Transformed profile, #tau_{21}^{DDT}")
        
        # (Opt.) plot fit
        if intercept is not None:
            x1, y1 = xmin, intercept + xmin * slope
            x2, y2 = xmax, intercept + xmax * slope
            c.plot([y1,y2], bins=[x1, x2], linecolor=root_colours[-1], markerstyle=0, linewidth=2, linestyle=1, label='Linear fit', option="L")
            c.plot([y1,y1], bins=[x1, x2], linecolor=ROOT.kGray + 3,   markerstyle=0, linewidth=2, linestyle=2, option="L")
            pass
        
        # Decoration
        c.xlabel("Large-#it{R} jet #rho^{DDT} = log(m^{2} / p_{T} / 1 GeV)")
        c.ylabel("#LT#tau_{21}#GT, #LT#tau_{21}^{DDT}#GT")
        c.xlim(-0.5, 5.5)
        c.ylim(0.2, 1.0)
        c.legend(xmin=0.50)
        
        c.text(["#sqrt{s} = 13 TeV",
                "W-tagging MC"],
               qualifier="Work in progress")
        
        # Save figure
        c.save(args.output + 'rp__{}.pdf'.format(name))
        pass
    
    return intercept, slope


"""
    # Plotting: Re-weighting
    # --------------------------------------------------------------------------
    with Profile("Plotting: Re-weighting"):


        fig, ax = plt.subplots(2, 4, figsize=(12,6))

        w_bkg  = bkg['weight']
        rw_bkg = bkg['reweight']
        w_tar  = np.ones((N_tar,)) * np.sum(bkg['weight']) / float(N_tar)

        for row, var in enumerate(['m', 'pt']):
            edges = np.linspace(0, np.max(bkg[var]), 60 + 1, endpoint=True)
            nbins  = len(edges) - 1

            v_bkg  = bkg[var]     # Background  mass/pt values for the background
            rv_bkg = P_bkg[:,row] # Transformed mass/pt values for the background
            rv_tar = P_tar[:,row] # Transformed mass/pt values for the targer

            ax[row,0].hist(v_bkg,  bins=edges, weights=w_bkg,  alpha=0.5, label='Background')
            ax[row,1].hist(v_bkg,  bins=edges, weights=rw_bkg, alpha=0.5, label='Background')
            ax[row,2].hist(rv_bkg, bins=nbins, weights=w_bkg,  alpha=0.5, label='Background') # =rw_bkg
            ax[row,2].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')
            ax[row,3].hist(rv_bkg, bins=nbins, weights=rw_bkg, alpha=0.5, label='Background')
            ax[row,3].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')

            for col in range(4):
                if col < 4: # 3
                    ax[row,col].set_yscale('log')
                    ax[row,col].set_ylim(1E+01, 1E+06)
                    if row == 1:
                        ax[row,col].set_ylim(1E-01, 1E+05)
                        pass
                    pass
                ax[row,col].set_xlabel("Jet %s%s%s" % (var, " (transformed)" if col > 1 else '', " (re-weighted)" if (col + 1) % 2 == 0 else ''))
                if col == 0:
                    ax[row,col].set_ylabel("Jets / {:.1f} GeV".format(np.diff(edges)[0]))
                    pass
                pass
            pass

        plt.legend()
        plt.savefig(args.output + 'priors_1d.pdf')

        # Plot 2D prior before and after re-weighting
        log.debug("Plotting 2D prior before and after re-weighting")
        fig, ax = plt.subplots(1,2,figsize=(11,5), sharex=True, sharey=True)
        h = ax[0].hist2d(P_bkg[:,0], P_bkg[:,1], bins=40, weights=bkg['weight'],   vmin=0, vmax=5, normed=True)
        h = ax[1].hist2d(P_bkg[:,0], P_bkg[:,1], bins=40, weights=bkg['reweight'], vmin=0, vmax=5, normed=True)
        ax[0].set_xlabel("Scaled log(m)")
        ax[1].set_xlabel("Scaled log(m)")
        ax[0].set_ylabel("Scaled log(pt)")

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
        fig.colorbar(h[3], cax=cbar_ax)
        plt.savefig(args.output + 'priors_2d.pdf')
        pass


    # Plotting: Cost log for classifier-only fit
    # --------------------------------------------------------------------------
    # Optimal number of training epochs
    opt_epochs = None

    with Profile("Plotting: Cost log, cross-val."):

        fig, ax = plt.subplots()
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))

        # @NOTE: Assuming no early stopping
        epochs = 1 + np.arange(len(histories[0]['loss']))

        for fold, hist in enumerate(histories):
            plt.plot(epochs, hist['val_loss'], color=colours[1], linewidth=0.6, alpha=0.3,
                     label='Validation (fold)' if fold == 0 else None)
            pass

        val_avg = np.mean([hist['val_loss'] for hist in histories], axis=0)
        plt.plot(epochs, val_avg,   color=colours[1], label='Validation (avg.)')

        # Store the optimal number of training epochs
        opt_epochs = epochs[np.argmin(val_avg)]
        log.info("Using optimal number of {:d} training epochs".format(opt_epochs))

        for fold, hist in enumerate(histories):
            plt.plot(epochs, hist['loss'],     color=colours[0], linewidth=1.0, alpha=0.3,
                     label='Training (fold)'   if fold == 0 else None)
            pass

        train_avg = np.mean([hist['loss'] for hist in histories], axis=0)
        plt.plot(epochs, train_avg, color=colours[0], label='Train (avg.)')

        plt.title('Classifier-only, stratified {}-fold training'.format(args.folds), fontweight='medium')
        plt.xlabel("Training epochs",    horizontalalignment='right', x=1.0)
        plt.ylabel("Objective function", horizontalalignment='right', y=1.0)

        epochs = [0] + list(epochs)
        step = max(int(np.floor(len(epochs) / 10.)), 1)

        plt.xticks(filter(lambda x: x % step == 0, epochs))
        plt.legend()
        plt.savefig(args.output + 'costlog.pdf')
        pass


    # Plotting: Cost log for adversarial fit
    # --------------------------------------------------------------------------
    with Profile("Plotting: Cost log, adversarial, full"):

        fig, ax = plt.subplots()
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
        history = result['history']
        print "history keys:", history.keys()
        epochs = 1 + np.arange(len(history['loss']))
        lambda_reg = cfg['adversary']['model']['lambda_reg']
        lr_ratio   = cfg['adversary']['model']['lr_ratio']


        classifier_loss = np.mean([loss for key,loss in history.iteritems() if key.startswith('adversary') and int(key.split('_')[-1]) % 2 == 1 ], axis=0)
        adversary_loss  = np.mean([loss for key,loss in history.iteritems() if key.startswith('adversary') and int(key.split('_')[-1]) % 2 == 0 ], axis=0) * lambda_reg
        #combined_loss   = np.array(history['loss']) * lambda_reg /
        #float(lr_ratio)
        combined_loss   = classifierloss + adversary_loss

        plt.plot(epochs, classifier_loss, color=colours[0],  linewidth=1.4,  label='Classifier')
        plt.plot(epochs, adversary_loss,  color=colours[1],  linewidth=1.4,  label=r'Adversary (\lambda = {})'.format(lambda_reg))
        plt.plot(epochs, combined_loss,   color=colours[-1], linestyle='--', label='Combined')

        plt.title('Adversarial training', fontweight='medium')
        plt.xlabel("Training epochs", horizontalalignment='right', x=1.0)
        plt.ylabel("Objective function",   horizontalalignment='right', y=1.0)
        ax.set_yscale('log')

        epochs = [0] + list(epochs)
        step = max(int(np.floor(len(epochs) / 10.)), 1)

        plt.xticks(filter(lambda x: x % step == 0, epochs))
        plt.legend()
        plt.savefig(args.output + 'adversary_costlog.pdf')
        pass


    # Plotting: Distributions/ROC
    # --------------------------------------------------------------------------
    with Profile("Plotting: Distributions/ROC"):

        # Tagger variables
        variables = ['tau21', 'D2', 'NN', 'ANN']

        # Plotted 1D tagger variable distributions
        fig, ax = plt.subplots(1, len(variables), figsize=(len(variables) * 4, 4))


        for ivar, var in enumerate(variables):

            # Get axis limits
            if var == 'D2':
                edges = np.linspace(0, 5, 50 + 1, endpoint=True)
            else:
                edges = np.linspace(0, 1, 50 + 1, endpoint=True)
                pass

            # Get value- and weight arrays
            v_sig = np.array(sig[var])
            v_bkg = np.array(bkg[var])

            w_sig = np.array(sig['weight'])
            w_bkg = np.array(bkg['weight'])

            # Mask out NaN's
            msk = ~np.isnan(sig[var])
            v_sig = v_sig[msk]
            w_sig = w_sig[msk]

            msk = ~np.isnan(bkg[var])
            v_bkg = v_bkg[msk]
            w_bkg = w_bkg[msk]

            # Plot distributions
            ax[ivar].hist(v_bkg, bins=edges, weights=w_bkg, alpha=0.5, normed=True, label='Background')
            ax[ivar].hist(v_sig, bins=edges, weights=w_sig, alpha=0.5, normed=True, label='Signal')

            ax[ivar].set_xlabel("Jet {}".format(var),
                                horizontalalignment='right', x=1.0)

            ax[ivar].set_ylabel("Jets / {:.3f}".format(np.diff(edges)[0]),
                                horizontalalignment='right', y=1.0)
            pass

        plt.legend()
        plt.savefig(args.output + 'tagger_distributions.pdf')

        # Plotted ROCs
        fig, ax = plt.subplots(figsize=(5,5))

        ax.plot([0,1],[0,1], 'k--', linewidth=1.0, alpha=0.2)
        for ivar, var in enumerate(reversed(variables)):
            eff_sig, eff_bkg = roc_efficiencies(sig[var], bkg[var], sig['weight'], bkg['weight'])
            try:
                auc = roc_auc(eff_sig, eff_bkg)
            except: # Efficiencies not monotonically increasing
                auc = 0.
                pass
            ax.plot(eff_bkg, eff_sig, label='{} (AUC: {:.3f})'.format(var, auc))
            pass

        plt.xlabel("Background efficiency", horizontalalignment='right', x=1.0)
        plt.ylabel("Signal efficiency",     horizontalalignment='right', y=1.0)
        plt.legend()
        plt.savefig(args.output + 'tagger_ROCs.pdf')
        pass

"""
