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
import studies

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
    with Profile("Study: Summary plot"):
        regex_nn = re.compile('\#lambda=[\d\.]+')
        regex_ub = re.compile('\#alpha=[\d\.]+')

        scan_features = {'NN': map(lambda feat: (feat, regex_nn.search(feat).group(0)), ann_vars),
                         'Adaboost': [(uboost_var, regex_ub.search(uboost_var).group(0))]}

        studies.summary(data, args, tagger_features, scan_features)
        pass

    # Perform distributions study
    # --------------------------------------------------------------------------
    with Profile("Study: Substructure tagger distributions"):
        for feat in tagger_features:
            studies.distribution(data, args, feat)
            pass
        pass

    # Perform jet mass distributions study
    # --------------------------------------------------------------------------
    with Profile("Study: Jet mass distributions"):
        for feat in tagger_features:
            studies.jetmass(data, args, feat)
            pass
        pass


    # Perform robustness study
    # --------------------------------------------------------------------------
    with Profile("Study: Robustness"):
        studies.jsd(data, args, tagger_features)
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
