#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing comparison studies."""

# Basic import(s)
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
from adversarial.utils import initialise_backend, roc_efficiencies, roc_auc, wpercentile, latex
from adversarial.profile import profile, Profile
from adversarial.new_utils import parse_args, initialise, load_data, mkdir
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp

def filename (name):
    """Method to standardise a given filename, and remove special characters."""
    return name.lower() \
           .replace('#', '') \
           .replace(',', '__') \
           .replace('.', 'p') \
           .replace('(', '__') \
           .replace(')', '') \
           .replace('%', '') \
           .replace('=', '_')


def signal_high (feat):
    """Method to determine whether the signal distribution is towards higher values."""
    return ('tau21' in feat.lower() or 'd2' in feat.lower())

# Define mass bins
MASSBINS = np.linspace(40, 300, (300 - 40) // 10 + 1, endpoint=True)

def JSD(P, Q, base=2):
    """Compute Jensen-Shannon divergence of two distribtions.
    From: [https://stackoverflow.com/a/27432724]
    """
    _P = P / np.sum(P)
    _Q = Q / np.sum(Q)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M, base=base) + entropy(_Q, _M, base=base))


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
    #data = data.sample(frac=0.2)  # @TEMP!


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
    kNN_eff = 20
    uboost_eff = 20
    uboost_uni = 1.0
    D2_kNN_var = 'D2-kNN({:d}%)'.format(kNN_eff)
    uboost_var = 'uBoost(#varepsilon={:d}%,#alpha={:.1f})'.format(uboost_eff, uboost_uni)

    lambda_reg  = 1
    lambda_regs = sorted([0.1, 10, 1, 100])
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

    tagger_features = ['Tau21','Tau21DDT', 'D2', D2_kNN_var, 'NN', ann_var, 'Adaboost', uboost_var] #, nn_mass_var, nn_linear_var]  # D2CSS, N2KNN

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
            data['rho'] = pd.Series(np.log(np.square(data['m']) / np.square(data['pt'])), index=data.index)

            X = data[['rho', 'pt']].as_matrix().astype(np.float)
            X[:,0] -= -5.5
            X[:,0] /= -2.0 - (-5.5)
            X[:,1] -= 200.
            X[:,1] /= 2000. - 200.

            with gzip.open('models/knn/knn_D2_{}.pkl.gz'.format(kNN_eff), 'r') as f:
                knn = pickle.load(f)
                pass
            kNN_percentile = knn.predict(X).flatten()
            data[D2_kNN_var] = pd.Series(data['D2'] - kNN_percentile, index=data.index)
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
                combined.load_weights('models/adversarial/combined/full/combined_lambda{}.h5'.format(lambda_str_))
                data[ann_var_] = pd.Series(classifier.predict(data[features].as_matrix().astype(K.floatx()), batch_size=2048 * 8).flatten().astype(K.floatx()), index=data.index)
                pass
            pass


        """ @TEMP
        # NN: mass-reweighted
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("NN (mass-reweighted)"):
            classifier = load_model('models/adversarial/classifier_massreweighted/full/classifier_massreweighted.h5')

            data[nn_mass_var] = pd.Series(classifier.predict(data[features].as_matrix().astype(K.floatx()), batch_size=2048 * 8).flatten().astype(K.floatx()), index=data.index)
            pass

        # NN: linearly decorrelated
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("NN (linearly decorrelated)"):
            decorrelator = decorrelation_model(classifier, 1, **cfg['combined']['model'])
            for nn_linear_var_, lambda_str_ in zip(nn_linear_vars, lambda_strs):
                decorrelator.load_weights('models/adversarial/combined/full/classifier_decorrelator_lambda{}.h5'.format(lambda_str_))
                data[nn_linear_var_] = pd.Series(classifier.predict(data[features].as_matrix().astype(K.floatx()), batch_size=2048 * 8).flatten().astype(K.floatx()), index=data.index)
                pass
            pass
        #"""


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


    # Perform generalised ROC study
    # --------------------------------------------------------------------------
    target_tpr = 0.5
    axisrange = (0.5, 500.)
    with Profile("Study: Generalised ROC"):

        points = list()
        for feat in tagger_features + ann_vars:  # + nn_linear_vars:
            print  "-- {}".format(feat)

            # Check for duplicates
            if feat in map(lambda t: t[2], points):
                print "    Skipping (already encounted)"
                continue

            # scikit-learn assumes signal towards 1, background towards 0
            pred = data[feat].values.copy()
            if 'Tau' in feat or 'D2' in feat:
                print "   Reversing cut direction for {}".format(feat)
                pred *= -1.
                pass
            fpr, tpr, thresholds = roc_curve(data['signal'], pred, sample_weight=data['weight'])

            # Get background rejection factor @ eff_sig. = 50%
            idx = np.argmin(np.abs(tpr - target_tpr))
            rej = 1. / fpr[idx]
            cut = thresholds[idx]

            # Get JSD(pass || fail) @ eff_sig. = 50%
            msk      = data['signal'] == False
            msk_pass = pred > cut

            p, _ = np.histogram(data.loc[ msk_pass & msk, 'm'].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk, 'weight'].values, density=True)
            f, _ = np.histogram(data.loc[~msk_pass & msk, 'm'].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk, 'weight'].values, density=True)

            jsd = JSD(p, f)

            # Add point to array
            points.append((rej, 1. / jsd, feat))
            pass

        # Canvas
        c = rp.canvas(batch=not args.show, size=(600,600))

        # Reference lines
        c.plot([.1, .1, 9999, 9999], bins=[axisrange[0] + 0.05, 9999, 9999, axisrange[0] + 0.048], linecolor=0,              linewidth=0, markerstyle=0, markersize=0, option='AP')
        c.plot([1, 9999],            bins=[1, 1],                                                  linecolor=ROOT.kGray + 2, linewidth=1, option='L')
        c.plot([1, 1],               bins=[1, 9999],                                               linecolor=ROOT.kGray + 2, linewidth=1, option='L')

        # Markers
        for ipoint, feat in enumerate(tagger_features):
            # Coordinates, label
            idx = map(lambda t: t[2], points).index(feat)
            x, y, label = points[idx]

            if label.startswith('ANN'):
                label = 'ANN'
                pass

            # Style
            colour      = rp.colours[(ipoint // 2) % len(rp.colours)]
            markerstyle = 20 + (ipoint % 2) * 4

            # Draw
            c.graph([y], bins=[x], markercolor=colour, markerstyle=markerstyle, label=latex(label, ROOT=True), option='P')
            pass

        # Markers, paramerised decorrelation
        for feat in ann_vars:
            #if feat in tagger_features:
            #    continue
            idx = map(lambda t: t[2], points).index(feat)
            x, y, label = points[idx]

            # Style
            ipoint = 5
            colour      = rp.colours[(ipoint // 2) % len(rp.colours)]
            markerstyle = 20 + (ipoint % 2) * 4

            # Draw
            c.graph([y], bins=[x], markercolor=colour, markerstyle=markerstyle, option='P')
            c.latex("   " + label[4:-1], x, y, textsize=11, align=12, textcolor=ROOT.kGray + 2)
            pass

        # Connecting lines
        for i in [0,1,3]:
            x1, y1, _ = points[2 * i + 0]
            x2, y2, _ = points[2 * i + 1]
            colour = rp.colours[i]
            c.graph([y1, y2], bins=[x1, x2], linecolor=colour, linestyle=2, option='L')
            pass

        # Connecting lines
        feats = [tagger_features[4]] + ann_vars
        for feat1, feat2 in zip(feats[:-1], feats[1:]):
            idx1 = map(lambda t: t[2], points).index(feat1)
            idx2 = map(lambda t: t[2], points).index(feat2)

            x1, y1, _ = points[idx1]
            x2, y2, _ = points[idx2]
            colour = rp.colours[2]
            c.graph([y1, y2], bins=[x1, x2], linecolor=colour, linestyle=2, option='L')
            pass

        # Decorations
        c.legend(xmin=0.65, xmax=0.90)
        c.xlabel("Background rejection, 1 / #varepsilon_{bkg.} @ #varepsilon_{sig.} = 50%")
        c.ylabel("1 / JSD(P_{pass}(m) #parallel P_{fail}(m) ) @ #varepsilon_{sig.} = 50%")
        c.xlim(*axisrange)
        c.ylim(axisrange[0], axisrange[1] * 10)
        c.logx()
        c.logy()
        c.latex("No separation",                   0.95, 420.,   angle=90, align=31, textsize=11, textcolor=ROOT.kGray + 2)
        c.latex("Maximal dissimilarity",         400.,     0.95, angle= 0, align=33, textsize=11, textcolor=ROOT.kGray + 2)
        c.latex("Less sculpting #rightarrow",      1.1,   25.,   angle=90, align=23, textsize=11, textcolor=ROOT.kGray + 2)
        c.latex("Greater separtion #rightarrow",  25.,     1.1,  angle= 0, align=21, textsize=11, textcolor=ROOT.kGray + 2)
        ROOT.gStyle.SetTitleOffset(3.0, 'x')
        ROOT.gStyle.SetTitleOffset(1.6, 'y')

        c.text(["#sqrt{s} = 13 TeV,  QCD jets",
                "Testing dataset",
                "Baseline selection",
                ],
            xmin=0.27,
            qualifier=QUALIFIER)

        # Save
        if args.save:
            mkdir('figures/')
            c.save('figures/genroc.pdf')
            pass

        # Show
        if args.show:
            c.show()
            pass

        # ...

        pass


    exit()  # @TEMP

    # ...


    # Perform distributions study
    # --------------------------------------------------------------------------
    with Profile("Study: Distributions"):
        for feat in tagger_features:

            # Define bins
            if 'knn' in feat.lower():
                xmin, xmax = -1, 2
            elif 'NN' in feat or 'tau21' in feat.lower() or 'boost' in feat.lower():
                xmin, xmax = 0., 1.
            elif feat == 'D2':
                xmin, xmax = 0, 3.5
            else:
                xmin = wpercentile (data[feat].as_matrix().flatten(),  1, weights=data['weight'].as_matrix().flatten())
                xmax = wpercentile (data[feat].as_matrix().flatten(), 99, weights=data['weight'].as_matrix().flatten())
                pass

            bins = np.linspace(xmin, xmax, 50 + 1, endpoint=True)

            # Canvas
            c = rp.canvas(batch=not args.show)

            # Plots
            ROOT.gStyle.SetHatchesLineWidth(3)
            c.hist(data.loc[data['signal'] == 0, feat].as_matrix().flatten(), bins=bins,
                   weights=data.loc[data['signal'] == 0, 'weight'].as_matrix().flatten(),
                   alpha=0.5, fillcolor=rp.colours[1], label="QCD jets", normalise=True,
                   fillstyle=3445, linewidth=3, linecolor=rp.colours[1])
            c.hist(data.loc[data['signal'] == 1, feat].as_matrix().flatten(), bins=bins,
                   weights=data.loc[data['signal'] == 1, 'weight'].as_matrix().flatten(),
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


    # Perform jet mass distributions study
    # --------------------------------------------------------------------------
    with Profile("Study: Jet mass distributions"):
        for feat in tagger_features:

            # Define masks; fixed signal efficiency cut
            eff_bkg = 20
            msk_sig = data['signal'] == 1
            msk_bkg = ~msk_sig
            eff_cut = eff_bkg if signal_high(feat) else 100 - eff_bkg
            cut = wpercentile(data.loc[msk_bkg, feat].as_matrix().flatten(), eff_cut, weights=data.loc[msk_bkg, 'weight'].as_matrix().flatten())
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
            h_fail = c.hist(data.loc[msk_bkg & ~msk_pass, 'm'].as_matrix().flatten(), bins=bins,
                            weights=data.loc[msk_bkg & ~msk_pass, 'weight'].as_matrix().flatten(),
                            alpha=0.3, fillcolor=rp.colours[1], normalise=True,
                            fillstyle=3445, linewidth=3, label="Failing cut",
                            linecolor=rp.colours[1])
            h_pass = c.hist(data.loc[msk_bkg &  msk_pass, 'm'].as_matrix().flatten(), bins=bins,
                            weights=data.loc[msk_bkg &  msk_pass, 'weight'].as_matrix().flatten(),
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
                    "Fixed #varepsilon_{bkg.} = %d%% cut on %s" % (eff_bkg, latex(feat, ROOT=True)),
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
                c.save('figures/jetmass_{}__eff_bkg_{:d}.pdf'.format(filename(feat), int(eff_bkg)))
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

            # Define cuts
            cuts = list()
            for eff in effs:
                cut = wpercentile(data.loc[msk, feat].as_matrix().flatten(), eff, weights=data.loc[msk, 'weight'].as_matrix().flatten())
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
                h_pass = c.hist(data.loc[ msk_pass & msk, 'm'].as_matrix().flatten(), bins=bins, weights=data.loc[ msk_pass & msk, 'weight'].as_matrix().flatten(), normalise=True, display=False)
                h_fail = c.hist(data.loc[~msk_pass & msk, 'm'].as_matrix().flatten(), bins=bins, weights=data.loc[~msk_pass & msk, 'weight'].as_matrix().flatten(), normalise=True, display=False)

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
        c.ylabel("JSD(1/N_{pass} dN_{pass}/dm #parallel 1/N_{fail} dN_{fail}/dm)")
        c.text(["#sqrt{s} = 13 TeV,  QCD jets",
                "Testing dataset",
                "Baseline selection",
                ],
            qualifier=QUALIFIER)
        c.latex("Maximal dissimilarity", 0.065, 1.2, align=11, textsize=11, textcolor=ROOT.kGray + 2)
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

            # Define cuts
            cuts = list()
            for eff in effs:
                cut = wpercentile(data.loc[msk, feat].as_matrix().flatten(), eff, weights=data.loc[msk, 'weight'].as_matrix().flatten())
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

                M = np.vstack((data.loc[msk, 'm'].as_matrix().flatten(), msk_pass[msk])).T
                weights = data.loc[msk, 'weight'].as_matrix().flatten()

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
                c.save('figures/eff_{}.pdf'.format(filename(feat)))
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
                eff_sig, eff_bkg = roc_efficiencies (data.loc[msk_mass &  msk_sig, feat].as_matrix().flatten().astype(float),
                                                     data.loc[msk_mass & ~msk_sig, feat].as_matrix().flatten().astype(float),
                                                     sig_weight=data.loc[msk_mass &  msk_sig, 'weight'].as_matrix().flatten().astype(float),
                                                     bkg_weight=data.loc[msk_mass & ~msk_sig, 'weight'].as_matrix().flatten().astype(float))

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
                AUCs[feat] = roc_auc (*ROCs[feat])
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
