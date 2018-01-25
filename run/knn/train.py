#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training fixed-efficiency KNN classifier for de-correlated jet tagging."""

# Basic import(s)
import pickle
import itertools

# Scientific import(s)
import ROOT
import root_numpy
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# Project import(s)
from adversarial.utils import apply_patch, wpercentile, latex
from adversarial.profile import profile, Profile
from adversarial.new_utils import parse_args, initialise, load_data, mkdir
from adversarial.constants import *

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
    data = data[data['train'] == 1]


    # Common definitions
    # --------------------------------------------------------------------------
    var = 'D2'  # Substructure variable to decorrelate
    eff = 20  # Fixed backround efficiency at which to perform decorrelation
    varx = 'rho'
    vary = 'pt'
    msk_bkg = data['signal'] == 0
    bounds = [
        ROOT.TF1('bounds_0', "TMath::Sqrt( TMath::Power( 40, 2) * TMath::Exp(-x) )", -5.5, -2.0),
        ROOT.TF1('bounds_1', "TMath::Sqrt( TMath::Power(300, 2) * TMath::Exp(-x) )", -5.5, -2.0)
        ]
    ROOT.gStyle.SetPalette(53)
    nb_contour = 13 * 16  # ... * 4
    
    for bound in bounds:
        bound.SetLineColor(ROOT.kGray + 2)
        bound.SetLineWidth(1)
        pass
    

    # Adding rho variable
    # --------------------------------------------------------------------------
    with Profile("Adding rho variable"):
        data['rho'] = pd.Series(np.log(np.square(data['m']) / np.square(data['pt'])), index=data.index)
        pass


    # Filling profile
    # --------------------------------------------------------------------------
    with Profile("Filling profile"):

        # Fill numpy array with weighted percentiles
        shape = (20,20)
        binsx = np.linspace(-5.5,  -2.0, shape[0] + 1, endpoint=True)  # rho bins
        binsy = np.linspace(200., 2000., shape[1] + 1, endpoint=True)  # pt bins
        x = np.zeros(shape)
        y = np.zeros_like(x)
        z = np.zeros_like(x)

        profile_meas = ROOT.TH2F('profile_meas', "", len(binsx) - 1, binsx.flatten('C'), len(binsy) - 1, binsy.flatten('C'))

        for i,j in itertools.product(*map(range, shape)):
            # Bin edges in x and y
            xmin, xmax = binsx[i:i+2]
            ymin, ymax = binsy[j:j+2]

            # Masks
            mskx = (data[varx] > xmin) & (data[varx] <= xmax)
            msky = (data[vary] > ymin) & (data[vary] <= ymax)
            msk  = msk_bkg & mskx & msky

            # Percentile
            perc = np.nan
            if np.sum(msk) > 20:  # Ensure sufficient statistics for meaningful percentile
                perc = wpercentile(data=   data.loc[msk, var]     .as_matrix().astype(np.float).flatten(), percents=eff,
                                   weights=data.loc[msk, 'weight'].as_matrix().astype(np.float).flatten())
                pass
            x[i,j] = (xmin + xmax) * 0.5
            y[i,j] = (ymin + ymax) * 0.5
            z[i,j] = perc

            if perc == perc:
                profile_meas.SetBinContent(i + 1, j + 1, perc)
                pass
            pass

        # Normalise arrays
        x -= -5.5
        x /= -2.0 - (-5.5)
        y -= 200.
        y /= 2000. - 200.
        pass


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
        profile_meas.GetXaxis().SetTitle("Large-#it{R} jet " + latex(varx, ROOT=True) + " = log(m_{calo}^{2}/p_{T}^{2})")
        profile_meas.GetYaxis().SetTitle("Large-#it{R} jet " + latex(vary, ROOT=True) + " [GeV]")
        profile_meas.GetZaxis().SetTitle("Measured, weighted {}-percentile of {}".format(eff, latex(var, ROOT=True)))
        profile_meas.GetYaxis().SetTitleOffset(1.6)
        profile_meas.GetZaxis().SetTitleOffset(1.4)
        profile_meas.GetZaxis().SetRangeUser(0.7, 2.0)
        profile_meas.SetContour(nb_contour)

        # Draw
        profile_meas.Draw('COLZ')
        bounds[0].DrawCopy("SAME")
        bounds[1].DrawCopy("SAME")
        c.latex("m_{calo} > 40 GeV",  -4.5, bounds[0].Eval(-4.5) + 30, align=21, angle=-20, textsize=13, textcolor=ROOT.kGray + 2)
        c.latex("m_{calo} < 300 GeV", -3.2, bounds[1].Eval(-3.2) - 30, align=23, angle=-53, textsize=13, textcolor=ROOT.kGray + 2)
        
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
        c.save('figures/knn_profile.pdf')
        pass


    # Fit KNN classifier
    # --------------------------------------------------------------------------
    with Profile("Fitting KNN classifier"):
        # Format arrays
        X = np.vstack((x.flatten(), y.flatten())).T
        Y = z.flatten()

        # Remove NaN's
        msk_nan = np.isnan(Y)

        # Fit KNN regressor
        knn = KNeighborsRegressor(weights='distance')
        knn.fit(X[~msk_nan,:], Y[~msk_nan])
        pass


    # Plotting fit
    # --------------------------------------------------------------------------
    with Profile("Plotting fit"):
        Nx = len(binsx)
        Ny = len(binsy)
        rebin = 10
        binsx_fine = np.interp(np.linspace(0, (Nx - 1), (Nx - 1) * rebin + 1, endpoint=True), range(Nx), binsx)
        binsy_fine = np.interp(np.linspace(0, (Ny - 1), (Ny - 1) * rebin + 1, endpoint=True), range(Ny), binsy)

        gx, gy = np.meshgrid(binsx_fine,binsy_fine)
        gx -= -5.5
        gx /= -2.0 - (-5.5)
        gy -= 200.
        gy /= 2000. - 200.
        X_fine = np.vstack((gx.flatten(), gy.flatten())).T
        fit = knn.predict(X_fine).reshape(gx.shape).T
        
        # Fill ROOT "profile"
        profile_fit = ROOT.TH2F('profile_fit', "", len(binsx_fine) - 1, binsx_fine.flatten('C'), len(binsy_fine) - 1, binsy_fine.flatten('C'))
        for i,j in itertools.product(*map(range, map(len,[binsx_fine, binsy_fine]))):
            x,y = binsx_fine[i], binsy_fine[j]
            x = (x - (-5.5))/(-2.0 - (-5.5))
            y = (y - 200.)/(2000.0 - 200.)
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
        profile_fit.GetXaxis().SetTitle("Large-#it{R} jet " + latex(varx, ROOT=True) + " = log(m_{calo}^{2}/p_{T}^{2})")
        profile_fit.GetYaxis().SetTitle("Large-#it{R} jet " + latex(vary, ROOT=True) + " [GeV]")
        profile_fit.GetZaxis().SetTitle("kNN-fitted, weighted {}-percentile of {}".format(eff, latex(var, ROOT=True)))
        profile_fit.GetYaxis().SetTitleOffset(1.6)
        profile_fit.GetZaxis().SetTitleOffset(1.4)
        profile_fit.GetZaxis().SetRangeUser(0.7, 2.0)
        profile_fit.SetContour(nb_contour)

        # Draw
        profile_fit.Draw('COLZ')
        bounds[0].DrawCopy("SAME")
        bounds[1].DrawCopy("SAME")
        c.latex("m_{calo} > 40 GeV",  -4.5, bounds[0].Eval(-4.5) + 30, align=21, angle=-20, textsize=13, textcolor=ROOT.kGray + 2)
        c.latex("m_{calo} < 300 GeV", -3.2, bounds[1].Eval(-3.2) - 30, align=23, angle=-53, textsize=13, textcolor=ROOT.kGray + 2)

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
        c.save('figures/knn_fit.pdf')
        pass


    # Saving KNN classifier
    # --------------------------------------------------------------------------
    with Profile("Saving KNN classifier"):

        # Ensure model directory exists
        mkdir('models/knn/')

        # Save classifier
        with open('models/knn/knn_{:s}_{:.0f}.pkl'.format(var, eff), 'w') as f:
            pickle.dump(knn, f)
            pass
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
