#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing fixed-efficiency kNN regressor."""

# Basic import(s)
import gzip
import pickle

# Scientific import(s)
import ROOT
import numpy as np

# Project import(s)
from adversarial.utils import latex
from adversarial.profile import profile, Profile
from adversarial.new_utils import parse_args, initialise, load_data, mkdir
from adversarial.constants import *

# Local import(s)
from .common import *

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    data, _, _ = load_data(args.input + 'data.h5')
    data = data[(data['train'] == 0) & (data['signal'] == 0)]


    # Common definitions
    # --------------------------------------------------------------------------
    bounds = [
        ROOT.TF1('bounds_0', "TMath::Sqrt( TMath::Power( 40, 2) * TMath::Exp(-x) )", AXIS[VARX][1], AXIS[VARX][2]),
        ROOT.TF1('bounds_1', "TMath::Sqrt( TMath::Power(300, 2) * TMath::Exp(-x) )", AXIS[VARX][1], AXIS[VARX][2])
        ]
    ROOT.gStyle.SetPalette(53)
    nb_contour = 13 * 16

    for bound in bounds:
        bound.SetLineColor(ROOT.kGray + 2)
        bound.SetLineWidth(1)
        pass


    # Adding variable(s)
    # --------------------------------------------------------------------------
    add_variables(data)


    # Filling profile
    # --------------------------------------------------------------------------
    profile_meas, _ = fill_profile(data)


    # Plotting profile
    # --------------------------------------------------------------------------
    with Profile("Plotting profile"):

        # rootplotting
        c = rp.canvas(batch=True)
        pad = c.pads()[0]._bare()
        pad.cd()
        pad.SetRightMargin(0.20)
        pad.SetLeftMargin(0.15)
        pad.SetTopMargin(0.10)

        # Styling
        profile_meas.GetXaxis().SetTitle("Large-#it{R} jet " + latex(VARX, ROOT=True) + " = log(m^{2}/p_{T}^{2})")
        profile_meas.GetYaxis().SetTitle("Large-#it{R} jet " + latex(VARY, ROOT=True) + " [GeV]")
        profile_meas.GetZaxis().SetTitle("Measured, weighted {}-percentile of {}".format(EFF, latex(VAR, ROOT=True)))
        profile_meas.GetYaxis().SetTitleOffset(1.6)
        profile_meas.GetZaxis().SetTitleOffset(1.4)
        profile_meas.GetZaxis().SetRangeUser(0.7, 2.0)
        profile_meas.SetContour(nb_contour)

        # Draw
        profile_meas.Draw('COLZ')
        bounds[0].DrawCopy("SAME")
        bounds[1].DrawCopy("SAME")
        c.latex("m > 40 GeV",  -4.5, bounds[0].Eval(-4.5) + 30, align=21, angle=-20, textsize=13, textcolor=ROOT.kGray + 2)
        c.latex("m < 300 GeV", -3.2, bounds[1].Eval(-3.2) - 30, align=23, angle=-53, textsize=13, textcolor=ROOT.kGray + 2)

        # Decorations
        c.text(qualifier=QUALIFIER,
               ymax=0.92,
               xmin=0.15)
        c.text(["#sqrt{s} = 13 TeV,  QCD jets",
                "Training dataset",
                "Baseline selection",
                ],
            ATLAS=False,
            ymax=0.83,
            xmin=0.18,
            textcolor=ROOT.kWhite)

        # Save
        mkdir('figures/')
        c.save('figures/knn_profile.pdf')
        pass


    # Loading KNN classifier
    # --------------------------------------------------------------------------
    with Profile("Loading KNN classifier"):
        with gzip.open('models/knn/knn_{:s}_{:.0f}.pkl.gz'.format(VAR, EFF), 'r') as f:
            knn = pickle.load(f)
            pass
        pass


    # Plotting fit
    # --------------------------------------------------------------------------
    with Profile("Plotting fit"):
        rebin = 10
        binsx_fine = np.interp(np.linspace(0, AXIS[VARX][0], AXIS[VARX][0] * rebin + 1, endpoint=True), range(AXIS[VARX][0] + 1), np.linspace(AXIS[VARX][1], AXIS[VARX][2], AXIS[VARX][0] + 1, endpoint=True))
        binsy_fine = np.interp(np.linspace(0, AXIS[VARY][1], AXIS[VARY][1] * rebin + 1, endpoint=True), range(AXIS[VARY][0] + 1), np.linspace(AXIS[VARY][1], AXIS[VARY][2], AXIS[VARY][0] + 1, endpoint=True))

        gx, gy = np.meshgrid(binsx_fine, binsy_fine)
        gx -= AXIS[VARX][1]
        gx /= AXIS[VARX][2] - AXIS[VARX][1]
        gy -= AXIS[VARY][1]
        gy /= AXIS[VARY][2] - AXIS[VARY][1]
        X_fine = np.vstack((gx.flatten(), gy.flatten())).T
        fit = knn.predict(X_fine).reshape(gx.shape).T

        # Fill ROOT "profile"
        profile_fit = ROOT.TH2F('profile_fit', "", len(binsx_fine) - 1, binsx_fine.flatten('C'), len(binsy_fine) - 1, binsy_fine.flatten('C'))
        for i,j in itertools.product(*map(range, map(len,[binsx_fine, binsy_fine]))):
            x,y = binsx_fine[i], binsy_fine[j]
            x -= AXIS[VARX][1]
            x /= AXIS[VARX][2] - AXIS[VARX][1]
            y -= AXIS[VARY][1]
            y /= AXIS[VARY][2] - AXIS[VARY][1]
            pred = knn.predict([[x,y]])
            profile_fit.SetBinContent(i + 1, j + 1, pred)
            pass

        # rootplotting
        c = rp.canvas(batch=True)
        pad = c.pads()[0]._bare()
        pad.cd()
        pad.SetRightMargin(0.20)
        pad.SetLeftMargin(0.15)
        pad.SetTopMargin(0.10)

        # Styling
        profile_fit.GetXaxis().SetTitle("Large-#it{R} jet " + latex(VARX, ROOT=True) + " = log(m^{2}/p_{T}^{2})")  # @TODO: Improve...
        profile_fit.GetYaxis().SetTitle("Large-#it{R} jet " + latex(VARY, ROOT=True) + " [GeV]")
        profile_fit.GetZaxis().SetTitle("kNN-fitted, weighted {}-percentile of {}".format(EFF, latex(VAR, ROOT=True)))
        profile_fit.GetYaxis().SetTitleOffset(1.6)
        profile_fit.GetZaxis().SetTitleOffset(1.4)
        profile_fit.GetZaxis().SetRangeUser(0.7, 2.0)
        profile_fit.SetContour(nb_contour)

        # Draw
        profile_fit.Draw('COLZ')
        bounds[0].DrawCopy("SAME")
        bounds[1].DrawCopy("SAME")
        c.latex("m > 40 GeV",  -4.5, bounds[0].Eval(-4.5) + 30, align=21, angle=-20, textsize=13, textcolor=ROOT.kGray + 2)
        c.latex("m < 300 GeV", -3.2, bounds[1].Eval(-3.2) - 30, align=23, angle=-53, textsize=13, textcolor=ROOT.kGray + 2)

        # Decorations
        c.text(qualifier=QUALIFIER,
               ymax=0.92,
               xmin=0.15)
        c.text(["#sqrt{s} = 13 TeV,  QCD jets",
                "Training dataset",
                "Baseline selection",
                ],
            ATLAS=False,
            ymax=0.83,
            xmin=0.18,
            textcolor=ROOT.kWhite)

        # Save
        mkdir('figures/')
        c.save('figures/knn_fit.pdf')
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
