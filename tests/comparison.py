#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing comparison studies."""

# Basic import(s)
import re
import gzip

# Scientific import(s)
import ROOT
import numpy as np
import pandas as pd
import pickle
import root_numpy
from array import array
from scipy.stats import entropy
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Project import(s)
from adversarial.utils import initialise_backend, wpercentile, latex, parse_args, initialise, load_data, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *
from .studies.common import *

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Argument-dependent setup
    # --------------------------------------------------------------------------
    # Initialise Keras backend
    initialise_backend(args)

    # Keras import(s)
    import keras.backend as K
    from keras.models import load_model

    # Project import(s)
    from adversarial.models import classifier_model, adversary_model, combined_model, decorrelation_model


    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data(args.input + 'data.h5')
    data = data[data['train'] == 0]
    #data = data.sample(frac=0.01)  # @TEMP!


    # Common definitions
    # --------------------------------------------------------------------------
    """
    configs = [
        {'type', 'DDT',
         'name': '#tau_{21}',
         'params': {
             'transformed:' False,
            },
        },

        {'type', 'DDT',
         'name': '#tau_{21}^{DDT}',
         'params': {
             'transformed:' True,
            },
        },

        {'type', 'kNN',
         'name': 'D_{2}',
         'params': {
             'transformed': False,
             'efficiency':  None
            },
        },

        {'type', 'kNN',
         'name': 'D_{2}^{kNN(%d%%)}' % kNN_eff,
         'params': {
             'transformed:' True,
             'efficiency':  kNN_eff,
            },
        },

        {'type': 'ANN',
         'name': "ANN(#lambda={:.0f})".format(lambda_reg) ,
         'params': {
             'lambda': lambda_reg,
            }

        },
        # ...
    ]
    #"""

    eps = np.finfo(float).eps
    msk_mass = (data['m'] > 60.) & (data['m'] < 100.)  # W mass window
    msk_sig  = data['signal'] == 1
    kNN_eff = 10
    uboost_eff = 20
    uboost_uni = 1.0
    D2_kNN_var = 'D2-kNN({:d}%)'.format(kNN_eff)
    uboost_var = 'uBoost(#varepsilon={:d}%,#alpha={:.1f})'.format(uboost_eff, uboost_uni)

    lambda_reg  = 60
    lambda_regs = sorted([3, 10, 30, 60, 100])
    ann_vars    = list()
    lambda_strs = list()
    for lambda_reg_ in lambda_regs:
        digits = int(np.ceil(max(-np.log10(lambda_reg_), 0)))
        lambda_str = '{l:.{d:d}f}'.format(d=digits,l=lambda_reg_).replace('.', 'p')
        lambda_strs.append(lambda_str)

        ann_var_ = "ANN(#lambda={:s})".format(lambda_str.replace('p', '.'))
        ann_vars.append(ann_var_)
        pass

    ann_var = ann_vars[lambda_regs.index(lambda_reg)]

    nn_mass_var = "NN(m-weight)"
    nn_linear_vars = list()
    for lambda_reg_ in lambda_regs:
        nn_linear_var_ = "NN(rho,#lambda={:.0f})".format(lambda_reg_)
        nn_linear_vars.append(nn_linear_var_)
        pass
    nn_linear_var = nn_linear_vars[lambda_regs.index(lambda_reg)]

    tagger_features = ['Tau21','Tau21DDT', 'D2', D2_kNN_var, 'D2CSS', 'NN', ann_var, 'Adaboost', uboost_var]

    # DDT variables
    fit_range = (1.5, 4.0)
    intercept, slope = 0.774633, -0.111879


    # Adding variables
    # --------------------------------------------------------------------------
    with Profile("Adding variables"):

        # Tau21DDT
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Tau21DDT"):
            with gzip.open('models/ddt/ddt.pkl.gz', 'r') as f:
                ddt = pickle.load(f)
                pass
            data['rhoDDT']   = pd.Series(np.log(np.square(data['m'])/(data['pt'] * 1.)), index=data.index)
            data['Tau21DDT'] = pd.Series(data['Tau21'] - ddt.predict(data['rhoDDT'].as_matrix().reshape((-1,1))), index=data.index)
            pass

        # D2-kNN
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("D2-kNN"):
            # Common, classifier-specific import(s)
            from run.knn.common import AXIS as knn_axis
            from run.knn.common import VAR  as knn_var
            from run.knn.common import VARX as knn_varx
            from run.knn.common import VARY as knn_vary
            from run.knn.common import add_variables as knn_add_variables

            # Add necessary variables
            knn_add_variables(data)

            X = data[[knn_varx, knn_vary]].as_matrix().astype(np.float)
            X[:,0] -= knn_axis[knn_varx][1]
            X[:,0] /= knn_axis[knn_varx][2] - knn_axis[knn_varx][1]
            X[:,1] -= knn_axis[knn_vary][1]
            X[:,1] /= knn_axis[knn_vary][2]  - knn_axis[knn_vary][1]

            with gzip.open('models/knn/knn_{}_{}.pkl.gz'.format(knn_var, kNN_eff), 'r') as f:
                knn = pickle.load(f)
                pass

            kNN_percentile = knn.predict(X).flatten()
            data[D2_kNN_var] = pd.Series(data[knn_var] - kNN_percentile, index=data.index)
            pass

        # D2-CSS
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("D2-CSS"):
            # Common, classifier-specific import(s)
            from run.css.common import AddCSS

            # Add necessary variables
            AddCSS("D2", data)
            pass

        # NN
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("NN"):
            classifier = load_model('models/adversarial/classifier/full/classifier.h5')

            data['NN'] = pd.Series(classifier.predict(data[features].as_matrix().astype(K.floatx()), batch_size=2048 * 8).flatten().astype(K.floatx()), index=data.index)
            pass

        # ANN
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("ANN"):
            adversary = adversary_model(gmm_dimensions=1,
                                        **cfg['adversary']['model'])

            combined = combined_model(classifier, adversary,
                                      **cfg['combined']['model'])

            for ann_var_, lambda_str_ in zip(ann_vars, lambda_strs):
                print "== Loading model for {}".format(ann_var_)
                combined.load_weights('models/adversarial/combined/full/combined_lambda{}.h5'.format(lambda_str_))
                data[ann_var_] = pd.Series(classifier.predict(data[features].as_matrix().astype(K.floatx()), batch_size=2048 * 8).flatten().astype(K.floatx()), index=data.index)
                pass
            print "== Done loading ANN models"
            pass

        # Adaboost
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Adaboost"):
            # @TODO:
            #  - gzip
            with open('models/uboost/adaboost.pkl', 'r') as f:
                adaboost = pickle.load(f)
                pass
            data['Adaboost'] = pd.Series(adaboost.predict_proba(data)[:,1].flatten().astype(K.floatx()), index=data.index)
            pass

        # uBoost
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("uBoost"):
            # @TODO: Add uniforming rate, `uboost_uni`
            # @TODO: gzip
            with open('models/uboost/uboost_{:d}.pkl'.format(100 - uboost_eff), 'r') as f:
                uboost = pickle.load(f)
                pass
            data[uboost_var] = pd.Series(uboost.predict_proba(data)[:,1].flatten().astype(K.floatx()), index=data.index)
            pass

        pass


    # Perform summary plot study
    # --------------------------------------------------------------------------
    regex_nn = re.compile('\#lambda=[\d\.]+')
    regex_ub = re.compile('\#alpha=[\d\.]+')

    scan_features = {'NN': map(lambda feat: (feat, regex_nn.search(feat).group(0)), ann_vars),
                     'Adaboost': [(uboost_var, regex_ub.search(uboost_var).group(0))]}

    from .studies import study_summary
    study_summary(data, args, tagger_features, scan_features)


    # Perform distributions study
    # --------------------------------------------------------------------------
    from .studies import study_distribution
    for feat in tagger_features:
        study_distribution(data, args, feat)
        pass


    return
    '''
    with Profile("Study: Distributions"):
        for feat in tagger_features:

            # Filter out NaNs (outside restricted phase-space)
            valid = ~np.isnan(data[feat])

            # Define bins
            if 'knn' in feat.lower():
                xmin, xmax = -1, 2
            elif 'NN' in feat or 'tau21' in feat.lower() or 'boost' in feat.lower():
                xmin, xmax = 0., 1.
            elif feat == 'D2':
                xmin, xmax = 0, 3.5
            else:
                xmin = wpercentile (data.loc[valid, feat].as_matrix().flatten(),  1, weights=data.loc[valid, 'weight'].as_matrix().flatten())
                xmax = wpercentile (data.loc[valid, feat].as_matrix().flatten(), 99, weights=data.loc[valid, 'weight'].as_matrix().flatten())
                pass

            bins = np.linspace(xmin, xmax, 50 + 1, endpoint=True)

            # Canvas
            c = rp.canvas(batch=not args.show)

            # Plots
            ROOT.gStyle.SetHatchesLineWidth(3)
            c.hist(data.loc[valid & (data['signal'] == 0), feat].as_matrix().flatten(), bins=bins,
                   weights=data.loc[valid & (data['signal'] == 0), 'weight'].as_matrix().flatten(),
                   alpha=0.5, fillcolor=rp.colours[1], label="QCD jets", normalise=True,
                   fillstyle=3445, linewidth=3, linecolor=rp.colours[1])
            c.hist(data.loc[valid & (data['signal'] == 1), feat].as_matrix().flatten(), bins=bins,
                   weights=data.loc[valid & (data['signal'] == 1), 'weight'].as_matrix().flatten(),
                   alpha=0.5, fillcolor=rp.colours[5], label="#it{W} jets", normalise=True,
                   fillstyle=3454, linewidth=3, linecolor=rp.colours[5])

            # Decorations
            ROOT.gStyle.SetTitleOffset(1.6, 'y')
            c.xlabel("Large-#it{R} jet " + latex(feat, ROOT=True))
            c.ylabel("1/N dN/d{}".format(latex(feat, ROOT=True)))
            c.text(["#sqrt{s} = 13 TeV",
                    "Testing dataset",
                    "Baseline selection",
                    ],
                qualifier=QUALIFIER)
            c.ylim(2E-03, 2E+00)
            c.logy()
            c.legend()

            # Save
            if args.save:
                mkdir('figures/')
                c.save('figures/dist_{}.pdf'.format(filename(feat)))
                pass

            # Show
            if args.show:
                c.show()
                pass
            pass
        pass
        '''


    # Perform jet mass distributions study
    # --------------------------------------------------------------------------
    with Profile("Study: Jet mass distributions"):
        for feat in tagger_features:

            # Define masks; fixed signal efficiency cut
            eff_sig = 50
            valid = ~np.isnan(data[feat])
            msk_sig = data['signal'] == 1
            msk_bkg = ~msk_sig
            eff_cut = eff_sig if signal_high(feat) else 100 - eff_sig
            cut = wpercentile(data.loc[valid & msk_sig, feat].as_matrix().flatten(), eff_cut, weights=data.loc[valid & msk_sig, 'weight'].as_matrix().flatten())
            msk_pass = data[feat] > cut

            # Ensure correct cut direction
            if signal_high(feat):
                msk_pass = ~msk_pass
                pass

            # Define bins
            bins = MASSBINS

            # Canvas
            c = rp.canvas(num_pads=2, size=(int(800 * 600 / 857.), 600), batch=not args.show)

            # Plots
            ROOT.gStyle.SetHatchesLineWidth(3)
            h_fail = c.hist(data.loc[valid & msk_bkg & ~msk_pass, 'm'].as_matrix().flatten(), bins=bins,
                            weights=data.loc[valid & msk_bkg & ~msk_pass, 'weight'].as_matrix().flatten(),
                            alpha=0.3, fillcolor=rp.colours[1], normalise=True,
                            fillstyle=3445, linewidth=3, label="Failing cut",
                            linecolor=rp.colours[1])
            h_pass = c.hist(data.loc[valid & msk_bkg &  msk_pass, 'm'].as_matrix().flatten(), bins=bins,
                            weights=data.loc[valid & msk_bkg &  msk_pass, 'weight'].as_matrix().flatten(),
                            alpha=0.3, fillcolor=rp.colours[5], normalise=True,
                            fillstyle=3454, linewidth=3, label="Passing cut",
                            linecolor=rp.colours[5])

            # Ratio plots
            c.pads()[1].hist([1], bins=[bins[0], bins[-1]], linecolor=ROOT.kGray + 1, linewidth=1, linestyle=1)
            h_ratio = c.ratio_plot((h_pass, h_fail), option='E2',   fillstyle=1001, fillcolor=rp.colours[0], linecolor=rp.colours[0], alpha=0.3)
            c.ratio_plot((h_pass, h_fail), option='HIST', fillstyle=0, linewidth=3, linecolor=rp.colours[0])

            # Out-of-bounds indicators
            ymin, ymax = 1E-01, 1E+01
            ratio = root_numpy.hist2array(h_ratio)
            centres = bins[:-1] + 0.5 * np.diff(bins)
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
            c.ylabel("1/N dN/d{}".format('m'))
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

            # Save
            if args.save:
                mkdir('figures/')
                c.save('figures/jetmass_{}__eff_sig_{:d}.pdf'.format(standardise(feat), int(eff_sig)))
                pass

            # Show
            if args.show:
                c.show()
                pass
            pass
        pass



    # Perform robustness study
    # --------------------------------------------------------------------------
    with Profile("Study: Robustness"):

        # Define common variables
        msk = data['signal'] == 0
        effs = np.linspace(0, 100, 10 * 2, endpoint=False)[1:].astype(int)
        ROOT.gStyle.SetTitleOffset(2.0, 'y')

        # Loop tagger features
        jsd = {feat: [] for feat in tagger_features}
        c = rp.canvas(batch=not args.show)
        for feat in tagger_features:

            # Filter out NaNs (outside restricted phase-space)
            valid = ~np.isnan(data[feat])

            # Define cuts
            cuts = list()
            for eff in effs:
                cut = wpercentile(data.loc[valid & msk, feat].as_matrix().flatten(), eff, weights=data.loc[valid & msk, 'weight'].as_matrix().flatten())
                cuts.append(cut)
                pass

            # Ensure correct direction of cut
            if not signal_high(feat):
                cuts = list(reversed(cuts))
                pass

            # Compute KL divergence for successive cuts
            for cut, eff in zip(cuts, effs):
                # Create ROOT histograms
                msk_pass = data[feat] > cut
                h_pass = c.hist(data.loc[valid &  msk_pass & msk, 'm'].as_matrix().flatten(), bins=bins, weights=data.loc[valid &  msk_pass & msk, 'weight'].as_matrix().flatten(), normalise=True, display=False)
                h_fail = c.hist(data.loc[valid & ~msk_pass & msk, 'm'].as_matrix().flatten(), bins=bins, weights=data.loc[valid & ~msk_pass & msk, 'weight'].as_matrix().flatten(), normalise=True, display=False)

                # Convert to numpy arrays
                p = root_numpy.hist2array(h_pass)
                f = root_numpy.hist2array(h_fail)

                # Compute Jensen-Shannon divergence
                jsd[feat].append(JSD(p, f, base=2))
                pass
            pass

        # Canvas
        c = rp.canvas(batch=not args.show)

        # Plots
        ref = ROOT.TH1F('ref', "", 10, 0., 1.)
        for i in range(ref.GetXaxis().GetNbins()):
            ref.SetBinContent(i + 1, 1)
            pass
        c.hist(ref, linecolor=ROOT.kGray + 2, linewidth=1)
        for ifeat, feat in enumerate(tagger_features):
            colour = rp.colours[(ifeat // 2) % len(rp.colours)]
            linestyle = 1 + (ifeat % 2)
            markerstyle = 20 + (ifeat // 2)
            c.plot(jsd[feat], bins=np.array(effs) / 100., linecolor=colour, markercolor=colour, linestyle=linestyle, markerstyle=markerstyle, label=latex(feat, ROOT=True), option='PL')

            # Split legend
            if   ifeat == 3:
                c.legend(xmin=0.56, width=0.18)
            elif ifeat == len(tagger_features) - 1:
                c.legend(xmin=0.73, width=0.18)
                pass
            pass


        # Decorations
        c.xlabel("Background efficiency #varepsilon_{bkg.}")
        #c.ylabel("JSD(1/N_{pass} dN_{pass}/dm #parallel 1/N_{fail}
        #dN_{fail}/dm)")
        c.ylabel("Mass correlation, JSD")
        c.text(["#sqrt{s} = 13 TeV,  QCD jets",
                "Testing dataset",
                "Baseline selection",
                ],
            qualifier=QUALIFIER)
        c.latex("Maximal sculpting", 0.065, 1.2, align=11, textsize=11, textcolor=ROOT.kGray + 2)
        c.xlim(0, 1)
        c.ymin(5E-05)
        c.padding(0.45)
        c.logy()

        # Save
        if args.save:
            mkdir('figures/')
            c.save('figures/jsd.pdf')
            pass

        # Show
        if args.show:
            c.show()
            pass
        pass


    # Perform efficiency study
    # --------------------------------------------------------------------------
    with Profile("Study: Efficiency"):

        # Define common variables
        msk  = data['signal'] == 0
        effs = np.linspace(0, 100, 10, endpoint=False)[1:].astype(int)
        ROOT.gStyle.SetTitleOffset(1.6, 'y')

        # Loop tagger features
        c = rp.canvas(batch=not args.show)
        for feat in tagger_features:

            # Filter out NaNs (outside restricted phase-space)
            valid = ~np.isnan(data[feat])

            # Define cuts
            cuts = list()
            for eff in effs:
                cut = wpercentile(data.loc[valid & msk, feat].as_matrix().flatten(), eff, weights=data.loc[valid & msk, 'weight'].as_matrix().flatten())
                cuts.append(cut)
                pass

            # Ensure correct direction of cut
            if not signal_high(feat):
                cuts = list(reversed(cuts))
                pass

            # Define mass bins
            bins = MASSBINS

            # Compute cut efficiency vs. mass
            profiles = list()
            for cut, eff in zip(cuts, effs):
                # Get correct pass-cut mask
                msk_pass = data[feat] > cut
                if signal_high(feat):
                    msk_pass = ~msk_pass
                    pass

                # Fill efficiency profile  # @TODO TEfficiency?
                profile = ROOT.TProfile('profile_{}_{}'.format(feat, cut), "",
                                        len(bins) - 1, bins)

                M = np.vstack((data.loc[valid & msk, 'm'].as_matrix().flatten(), msk_pass[valid & msk])).T
                weights = data.loc[valid & msk, 'weight'].as_matrix().flatten()

                root_numpy.fill_profile(profile, M, weights=weights)

                # Add to list
                profiles.append(profile)
                pass

            # Canvas
            c = rp.canvas(batch=not args.show)

            # Plots
            for idx, (profile, cut, eff) in enumerate(zip(profiles, cuts, effs)):
                colour = rp.colours[1]
                linestyle = 1
                c.hist(profile, linecolor=colour, linestyle=linestyle, option='HIST L')
                c.hist(profile, fillcolor=colour, alpha=0.3, option='E3')
                pass

            # Text
            for idx, (profile, cut, eff) in enumerate(zip(profiles, cuts, effs)):
                if int(eff) in [10, 50, 90]:
                    c.latex('#bar{#varepsilon}_{bkg.} = %d%%' % eff,
                            260., profile.GetBinContent(np.argmin(np.abs(bins - 270.)) + 1) + 0.025,
                            textsize=13,
                            textcolor=ROOT.kGray + 2, align=11)
                    pass
                pass


            # Decorations
            c.xlabel("Large-#it{R} jet mass [GeV]")
            c.ylabel("Background efficiency #varepsilon_{bkg.}")
            c.text(["#sqrt{s} = 13 TeV,  QCD jets",
                    "Testing dataset",
                    "Baseline selection",
                    "Sequential cuts on {}".format(latex(feat, ROOT=True)),
                    ],
                   qualifier=QUALIFIER)
            c.ylim(0, 1.9)

            # Save
            if args.save:
                mkdir('figures/')
                c.save('figures/eff_{}.pdf'.format(standardise(feat)))
                pass

            # Show
            if args.show:
                c.show()
                pass
            pass
        pass


    # Perform ROC study
    # --------------------------------------------------------------------------
    with Profile("Study: ROC"):

        # Computing ROC curves
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Computing ROC curves"):
            ROCs = dict()
            for feat in tagger_features:

                # Filter out NaNs (outside restricted phase-space)
                valid = ~np.isnan(data[feat])

                sign = 1. if ('D2' in feat or 'Tau21' in feat or 'N2' in feat) else -1.

                eff_sig, eff_bkg, thresholds = roc_curve(data.loc[valid & msk_mass, 'signal'].values,
                                                         data.loc[valid & msk_mass, feat]    .values * sign,
                                                         sample_weight=data.loc[valid & msk_mass, 'weight'].values)

                #### eff_sig, eff_bkg = roc_efficiencies (data.loc[valid & msk_mass &  msk_sig, feat].as_matrix().flatten().astype(float),
                ####                                      data.loc[valid & msk_mass & ~msk_sig, feat].as_matrix().flatten().astype(float),
                ####                                      sig_weight=data.loc[valid & msk_mass &  msk_sig, 'weight'].as_matrix().flatten().astype(float),
                ####                                      bkg_weight=data.loc[valid & msk_mass & ~msk_sig, 'weight'].as_matrix().flatten().astype(float))

                # Filter, to advoid background rejection blowing up
                indices = np.where((eff_bkg > 0) & (eff_sig > 0))
                eff_sig = eff_sig[indices]
                eff_bkg = eff_bkg[indices]

                # Subsample to 1% steps
                targets = np.linspace(0, 1, 100 + 1, endpoint=True)
                indices = np.array([np.argmin(np.abs(eff_sig - t)) for t in targets])
                eff_sig = eff_sig[indices]
                eff_bkg = eff_bkg[indices]

                # Store
                ROCs[feat] = (eff_sig,eff_bkg)
                pass
            pass


        # Computing ROC AUCs
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Computing ROC AUCs"):
            AUCs = dict()
            for feat in tagger_features:
                #AUCs[feat] = roc_auc (*ROCs[feat])
                sign = 1. if ('D2' in feat or 'Tau21' in feat or 'N2' in feat) else -1.
                AUCs[feat] = roc_auc_score(data.loc[valid & msk_mass, 'signal'].values,
                                           data.loc[valid & msk_mass, feat]    .values * sign,
                                           sample_weight=data.loc[valid & msk_mass, 'weight'].values)
            pass


        # Creating figure
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Creating figure"):

            # Canvas
            c = rp.canvas(batch=not args.show)

            # Plots
            # -- Random guessing
            c.graph(np.power(eff_sig, -1.), bins=eff_sig, linecolor=ROOT.kGray + 2, linewidth=1, option='AL')

            # -- AUCs
            categories = list()
            for feat in tagger_features:
                 line = "#scale[0.6]{#color[13]{AUC: %.3f}}" % AUCs[feat]
                 categories += [(line, {'linestyle': 0, 'fillstyle': 0, 'markerstyle': 0, 'option': ''})]
                 pass
            c.legend(categories=categories, xmin=0.80, width=0.04)

            # -- ROCs
            for ifeat, feat in enumerate(tagger_features):
                eff_sig, eff_bkg = ROCs[feat]
                c.graph(np.power(eff_bkg, -1.), bins=eff_sig, linestyle=1 + (ifeat % 2), linecolor=rp.colours[(ifeat // 2) % len(rp.colours)], linewidth=2, label=latex(feat, ROOT=True), option='L')
                pass
            c.legend(xmin=0.58, width=0.22)

            # Decorations
            c.xlabel("Signal efficiency #varepsilon_{sig.}")
            c.ylabel("Background rejection 1/#varepsilon_{bkg.}")
            c.text(["#sqrt{s} = 13 TeV",
                    "Testing dataset",
                    "Baseline selection",
                    "m #in  [60, 100] GeV",
                    ],
                qualifier=QUALIFIER)
            c.latex("Random guessing", 0.3, 1./0.3 * 0.9, align=23, angle=-12, textsize=13, textcolor=ROOT.kGray + 2)
            c.xlim(0., 1.)
            c.ylim(1E+00, 1E+05)
            c.logy()
            c.legend()

            # Save
            if args.save:
                mkdir('figures/')
                c.save('figures/roc.pdf')
                pass

            # Show
            if args.show:
                c.show()
                pass
            pass

        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass
