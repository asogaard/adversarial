#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing fixed-efficiency kNN regressor."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import ROOT
import numpy as np
import root_numpy

# Project import(s)
from adversarial.utils import latex, parse_args, initialise, load_data, mkdir, loadclf  #, initialise_backend
from adversarial.profile import profile, Profile
from adversarial.constants import *
#from run.adversarial.common import initialise_config

# Local import(s)
from .common import *

# Custom import(s)
import rootplotting as rp

# Global definitions
BOUNDS = [
    ROOT.TF1('bounds_0', "TMath::Sqrt( TMath::Power( 50, 2) * TMath::Exp(-x) )", AXIS[VARX][1], AXIS[VARX][2]),
    ROOT.TF1('bounds_1', "TMath::Sqrt( TMath::Power(300, 2) * TMath::Exp(-x) )", AXIS[VARX][1], AXIS[VARX][2])
    ]

NB_CONTOUR = 13 * 16

# Shout out to Cynthia Brewer and Mark Harrower
# [http://colorbrewer2.org]. Palette is colorblind-safe.
rgbs = [
    (247/255., 251/255., 255/255.),
    (222/255., 235/255., 247/255.),
    (198/255., 219/255., 239/255.),
    (158/255., 202/255., 225/255.),
    (107/255., 174/255., 214/255.),
    ( 66/255., 146/255., 198/255.),
    ( 33/255., 113/255., 181/255.),
    (  8/255.,  81/255., 156/255.),
    (  8/255.,  48/255., 107/255.)
    ]

red, green, blue = map(np.array, zip(*rgbs))
nb_cols = len(rgbs)
stops = np.linspace(0, 1, nb_cols, endpoint=True)

ROOT.TColor.CreateGradientColorTable(nb_cols, stops, red, green, blue, NB_CONTOUR)

BOUNDS[0].SetLineColor(ROOT.kGray + 3)
BOUNDS[1].SetLineColor(ROOT.kGray + 3)
for bound in BOUNDS:
    bound.SetLineWidth(1)
    bound.SetLineStyle(2)
    pass

#ZRANGE = (0.125, 0.325)
ZRANGE = (0., 2.5)


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, _, _ = load_data(args.input + 'data.h5', train=True)
    msk_sig = data['signal'] == 1
    msk_bkg = ~msk_sig

    # -------------------------------------------------------------------------
    ####
    #### # Initialise Keras backend
    #### initialise_backend(args)
    ####
    #### # Neural network-specific initialisation of the configuration dict
    #### initialise_config(args, cfg)
    ####
    #### # Keras import(s)
    #### from keras.models import load_model
    ####
    #### # NN
    #### from run.adversarial.common import add_nn
    #### with Profile("NN"):
    ####     classifier = load_model('models/adversarial/classifier/full/classifier.h5')
    ####     add_nn(data, classifier, 'NN')
    ####     pass
    # -------------------------------------------------------------------------

    # Fill measured profile
    profile_meas, _ = fill_profile(data[msk_bkg])

    # Add k-NN variable
    knnfeat = 'knn'
    add_knn(data, newfeat=knnfeat, path='models/knn/knn_{}_{}.pkl.gz'.format(VAR, EFF))

    # Loading KNN classifier
    knn = loadclf('models/knn/knn_{:s}_{:.0f}.pkl.gz'.format(VAR, EFF))

    # Filling fitted profile
    with Profile("Filling fitted profile"):
        rebin = 8
        edges, centres = dict(), dict()
        for ax, var in zip(['x', 'y'], [VARX, VARY]):

            # Short-hands
            vbins, vmin, vmax = AXIS[var]

            # Re-binned bin edges  @TODO: Make standardised right away?
            edges[ax] = np.interp(np.linspace(0,    vbins, vbins * rebin + 1, endpoint=True),
                                  range(vbins + 1),
                                  np.linspace(vmin, vmax,  vbins + 1,         endpoint=True))

            # Re-binned bin centres
            centres[ax] = edges[ax][:-1] + 0.5 * np.diff(edges[ax])
            pass


        # Get predictions evaluated at re-binned bin centres
        g = dict()
        g['x'], g['y'] = np.meshgrid(centres['x'], centres['y'])
        g['x'], g['y'] = standardise(g['x'], g['y'])

        X = np.vstack((g['x'].flatten(), g['y'].flatten())).T
        fit = knn.predict(X).reshape(g['x'].shape).T

        # Fill ROOT "profile"
        profile_fit = ROOT.TH2F('profile_fit', "", len(edges['x']) - 1, edges['x'].flatten('C'),
                                                   len(edges['y']) - 1, edges['y'].flatten('C'))
        root_numpy.array2hist(fit, profile_fit)
        pass


    # Plotting
    with Profile("Plotting"):
        for fit in [False, True]:

            # Select correct profile
            profile = profile_fit if fit else profile_meas

            # Plot
            plot(profile, fit)
            pass
        pass

    # Plotting local selection efficiencies for D2-kNN < 0
    # -- Compute signal efficiency
    for sig, msk in zip([True, False], [msk_sig, msk_bkg]):

        if sig:
            rgbs = [
                (247/255., 251/255., 255/255.),
                (222/255., 235/255., 247/255.),
                (198/255., 219/255., 239/255.),
                (158/255., 202/255., 225/255.),
                (107/255., 174/255., 214/255.),
                ( 66/255., 146/255., 198/255.),
                ( 33/255., 113/255., 181/255.),
                (  8/255.,  81/255., 156/255.),
                (  8/255.,  48/255., 107/255.)
                ]

            red, green, blue = map(np.array, zip(*rgbs))
            nb_cols = len(rgbs)
            stops = np.linspace(0, 1, nb_cols, endpoint=True)
        else:
            rgbs = [
                (255/255.,  51/255.,   4/255.),
                (247/255., 251/255., 255/255.),
                (222/255., 235/255., 247/255.),
                (198/255., 219/255., 239/255.),
                (158/255., 202/255., 225/255.),
                (107/255., 174/255., 214/255.),
                ( 66/255., 146/255., 198/255.),
                ( 33/255., 113/255., 181/255.),
                (  8/255.,  81/255., 156/255.),
                (  8/255.,  48/255., 107/255.)
                ]

            red, green, blue = map(np.array, zip(*rgbs))
            nb_cols = len(rgbs)
            stops = np.array([0] + list(np.linspace(0, 1, nb_cols - 1, endpoint=True) * (1. - EFF / 100.) + EFF / 100.))
            pass

        ROOT.TColor.CreateGradientColorTable(nb_cols, stops, red, green, blue, NB_CONTOUR)

        # Define arrays
        shape   = (AXIS[VARX][0], AXIS[VARY][0])
        bins    = [np.linspace(AXIS[var][1], AXIS[var][2], AXIS[var][0] + 1, endpoint=True) for var in VARS]
        x, y, z = (np.zeros(shape) for _ in range(3))

        # Create `profile` histogram
        profile = ROOT.TH2F('profile', "", len(bins[0]) - 1, bins[0].flatten('C'), len(bins[1]) - 1, bins[1].flatten('C'))

        # Compute inclusive efficiency in bins of `VARY`
        effs = list()
        for edges in zip(bins[1][:-1], bins[1][1:]):
            msk_bin  = (data[VARY] > edges[0]) & (data[VARY] < edges[1])
            msk_pass =  data[knnfeat] < 0
            num = data.loc[msk & msk_bin & msk_pass, 'weight_test'].values.sum()
            den = data.loc[msk & msk_bin,            'weight_test'].values.sum()
            effs.append(num/den)
            pass

        # Fill profile
        for i,j in itertools.product(*map(range, shape)):

            # Bin edges in x and y
            edges = [bin[idx:idx+2] for idx, bin in zip([i,j],bins)]

            # Masks
            msks = [(data[var] > edges[dim][0]) & (data[var] <= edges[dim][1]) for dim, var in enumerate(VARS)]
            msk_bin = reduce(lambda x,y: x & y, msks)
            data_ = data[msk & msk_bin]

            # Set non-zero bin content
            if np.sum(msk & msk_bin):
                msk_pass = data_[knnfeat] < 0
                num = data.loc[msk & msk_bin & msk_pass, 'weight_test'].values.sum()
                den = data.loc[msk & msk_bin,            'weight_test'].values.sum()
                eff = num/den
                profile.SetBinContent(i + 1, j + 1, eff)
                pass
            pass

        c = rp.canvas(batch=True)
        pad = c.pads()[0]._bare()
        pad.cd()
        pad.SetRightMargin(0.20)
        pad.SetLeftMargin(0.15)
        pad.SetTopMargin(0.10)

        # Styling
        profile.GetXaxis().SetTitle("Large-#it{R} jet " + latex(VARX, ROOT=True) + " = log(m^{2}/p_{T}^{2})")
        profile.GetYaxis().SetTitle("Large-#it{R} jet " + latex(VARY, ROOT=True) + " [GeV]")
        profile.GetZaxis().SetTitle("Selection efficiency for %s^{(%s%%)}" % (latex(VAR, ROOT=True), EFF))

        profile.GetYaxis().SetNdivisions(505)
        profile.GetZaxis().SetNdivisions(505)
        profile.GetXaxis().SetTitleOffset(1.4)
        profile.GetYaxis().SetTitleOffset(1.8)
        profile.GetZaxis().SetTitleOffset(1.3)
        zrange = (0., 1.)
        if zrange:
            profile.GetZaxis().SetRangeUser(*zrange)
            pass
        profile.SetContour(NB_CONTOUR)

        # Draw
        profile.Draw('COLZ')

        # Decorations
        c.text(qualifier=QUALIFIER, ymax=0.92, xmin=0.15)
        c.text(["#sqrt{s} = 13 TeV", "#it{W} jets" if sig else "Multijets"], ATLAS=False)

        # -- Efficiencies
        xaxis = profile.GetXaxis()
        yaxis = profile.GetYaxis()
        tlatex = ROOT.TLatex()
        tlatex.SetTextColor(ROOT.kGray + 2)
        tlatex.SetTextSize(0.023)
        tlatex.SetTextFont(42)
        tlatex.SetTextAlign(32)
        xt = xaxis.GetBinLowEdge(xaxis.GetNbins())
        for eff, ibin in zip(effs,range(1, yaxis.GetNbins() + 1)):
            yt = yaxis.GetBinCenter(ibin)
            tlatex.DrawLatex(xt, yt, "%s%.1f%%" % ("#bar{#varepsilon}^{rel}_{%s} = " % ('sig' if sig else 'bkg') if ibin == 1 else '', eff * 100.))
            pass

        # -- Bounds
        BOUNDS[0].DrawCopy("SAME")
        BOUNDS[1].DrawCopy("SAME")
        c.latex("m > 50 GeV",  -4.5, BOUNDS[0].Eval(-4.5) + 30, align=21, angle=-37, textsize=13, textcolor=ROOT.kGray + 3)
        c.latex("m < 300 GeV", -2.5, BOUNDS[1].Eval(-2.5) - 30, align=23, angle=-57, textsize=13, textcolor=ROOT.kGray + 3)

        # Save
        mkdir('figures/knn/')
        c.save('figures/knn/knn_eff_{}_{:s}_{:.0f}.pdf'.format('sig' if sig else 'bkg', VAR, EFF))
        pass

    return


def plot (profile, fit):
    """
    Method for delegating plotting.
    """

    # rootplotting
    c = rp.canvas(batch=True)
    pad = c.pads()[0]._bare()
    pad.cd()
    pad.SetRightMargin(0.20)
    pad.SetLeftMargin(0.15)
    pad.SetTopMargin(0.10)

    # Styling
    profile.GetXaxis().SetTitle("Large-#it{R} jet " + latex(VARX, ROOT=True) + " = log(m^{2}/p_{T}^{2})")
    profile.GetYaxis().SetTitle("Large-#it{R} jet " + latex(VARY, ROOT=True) + " [GeV]")
    profile.GetZaxis().SetTitle("%s %s^{(%s%%)}" % ("#it{k}-NN fitted" if fit else "Measured", latex(VAR, ROOT=True), EFF))

    profile.GetYaxis().SetNdivisions(505)
    profile.GetZaxis().SetNdivisions(505)
    profile.GetXaxis().SetTitleOffset(1.4)
    profile.GetYaxis().SetTitleOffset(1.8)
    profile.GetZaxis().SetTitleOffset(1.3)
    if ZRANGE:
        profile.GetZaxis().SetRangeUser(*ZRANGE)
        pass
    profile.SetContour(NB_CONTOUR)

    # Draw
    profile.Draw('COLZ')
    BOUNDS[0].DrawCopy("SAME")
    BOUNDS[1].DrawCopy("SAME")
    c.latex("m > 50 GeV",  -4.5, BOUNDS[0].Eval(-4.5) + 30, align=21, angle=-37, textsize=13, textcolor=ROOT.kGray + 3)
    c.latex("m < 300 GeV", -2.5, BOUNDS[1].Eval(-2.5) - 30, align=23, angle=-57, textsize=13, textcolor=ROOT.kGray + 3)

    # Decorations
    c.text(qualifier=QUALIFIER, ymax=0.92, xmin=0.15)
    c.text(["#sqrt{s} = 13 TeV", "Multijets"], ATLAS=False, textcolor=ROOT.kWhite)

    # Save
    mkdir('figures/knn/')
    c.save('figures/knn/knn_{}_{:s}_{:.0f}.pdf'.format('fit' if fit else 'profile', VAR, EFF))
    pass


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()  # (adversarial=True)

    # Call main function
    main(args)
    pass
