#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training DDT transform."""

# Basic import(s)
import gzip
import pickle
from array import array

# Scientific import(s)
import ROOT
import numpy as np
from scipy.special import gamma

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp

# Local import(s)
from .common import *


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, features, _ = load_data(args.input + 'data.h5', background=True, train=True)

    # Fill substructure profile
    D2BINS = np.linspace(0., 5., 501, endpoint=True)
    perform_optimisation("D2", D2BINS, data)
    return


@profile
def perform_optimisation (var, bins, data):
    """
    ...
    """

    # Fill 2D substructure profile
    profile2d = fill_2d_profile(data, var, bins, "m", MASS_BINS)

    # Get 1D profile for lowest mass bin
    profile0 = profile2d.ProjectionY("%s_lowMass"%profile2d.GetName(), 1, 1)

    # Perform the optimisation
    bestShapeVal = 0
    bestSumChi2 = 1e20
    for shapeVal in SHAPEVAL_RANGE:
        print "Shape value: ", shapeVal
        sumChi2 = 0.

        # Each mass bin needs to be optimized over omega
        for mass in range(len(MASS_BINS)-1):
            print "Mass bin: ", mass

            # Get 1D profile for current mass bin
            profile = profile2d.ProjectionY("%s_bin_%i"%(profile2d.GetName(),mass),mass+1, mass+1)

            # Fit current profile to low-mass profile
            chi2, bestOmega, _, _ = fit(profile, shapeVal, profile0, "%.2f"%mass)

            # Accumulate chi2
            sumChi2 += chi2
            pass

        # Update chi2 for current `shapeVal`
        if sumChi2 < bestSumChi2:
            bestSumChi2  = sumChi2
            bestShapeVal = shapeVal
            pass
        pass

    # Saving CSS transforms
    with Profile("Saving CSS transform"):

        # Ensure model directory exists
        mkdir('models/css/')

        # Write out the optifmal configuration for each mass bin
        for mass in range(len(MASS_BINS)-1):
            profile = profile2d.ProjectionY("%s_bin_%i_final"%(profile2d.GetName(),mass),mass+1, mass+1)
            sumChi2, bestOmega, profile_css, profile0rebin = fit(profile, bestShapeVal, profile0, "%.2f"%mass)
            print mass, bestOmega
            F,Ginv = get_css_fns(bestShapeVal, bestOmega, profile, "")

            rebinProfiles = profile.Clone("%s_%i_rebin"%(profile.GetName(), mass))
            rebinProfiles.Rebin(5)
            normalise(rebinProfiles)

            c1 = rp.canvas(batch=True)

            c1.hist(profile0rebin, label="Low Mass", linecolor=rp.colours[1], markercolor=rp.colours[1])
            c1.hist(profile_css, label="D2 CSS", linecolor=rp.colours[2], markercolor=rp.colours[2])
            c1.hist(rebinProfiles, label="D2", linecolor=rp.colours[3], markercolor=rp.colours[3])
            c1.legend()
            c1.save('figures/css/cssProfile_{}_{}.pdf'.format(var, mass))

            # Save classifier
            saveclf(F,    'models/css/css_%s_F_%i.pkl.gz' % (var,mass))
            saveclf(Ginv, 'models/css/css_%s_Ginv_%i.pkl.gz' % (var,mass))
            pass
        pass

    return 0


def fit (profile, shapeVal, lowMassProfile, name):
    """
    ...
    """

    # Convert 1D profiles to p.d.f.s
    normalise(profile,        density=True)
    normalise(lowMassProfile, density=True)

    # These are free parameters - need to tune these.
    bestOmega = 0.01
    bestChi2 = 1e30
    rebinLowMass = lowMassProfile.Clone("%s_%.2f_clone"%(name, shapeVal))
    rebinLowMass.Rebin(5)
    normalise(rebinLowMass)

    # Find optimal value for omega for this mass bin
    for omega in OMEGA_RANGE:
        jssVar_css = get_css(shapeVal, omega, profile, "_%.2f"%(omega))
        jssVar_css.Rebin(5)
        normalise(jssVar_css)

        chi2 = 0
        for i in range(1, jssVar_css.GetNbinsX()+1):
            # Want to compare to the low-mass region. Obviously this isn't quite
            # how chi2 works, but it works for a basic optimization
            if rebinLowMass.GetBinContent(i) > 0:
                chi2 += np.square(rebinLowMass.GetBinContent(i) - jssVar_css.GetBinContent(i))
                pass
            pass

        if chi2 < bestChi2 :
            bestChi2 = chi2
            bestOmega = omega
            pass
        pass

    # Compute CSS-transformed version of `var`
    jssVar_css = get_css(shapeVal, bestOmega, profile, "_%.2f_best"%(bestOmega))
    jssVar_css.Rebin(5)
    normalise(jssVar_css)

    return bestChi2, bestOmega, jssVar_css, rebinLowMass


def get_css (shapeval, omega, originalHist, name):
    """
    ...
    """
    F, Ginv = get_css_fns(shapeval, omega, originalHist, name)
    jssVar_css = ROOT.TH1D("css%s_%.2f_%.2f_%s"%(name, omega, shapeval,originalHist.GetName()),"low_m_css",originalHist.GetNbinsX(),0,originalHist.GetXaxis().GetXmax())

    # Apply the convolutions to get the new distributions
    for i in range(1, originalHist.GetNbinsX()+1):
        lowMValBkg = apply_css(originalHist.GetBinCenter(i), Ginv, F)
        jssVar_css.Fill(lowMValBkg, originalHist.GetBinContent(i))
        pass

    return jssVar_css


def get_css_fns (shapeval, omega, originalHist, name):
    """
    ...
    """

    # Compute $F_{CSS}( x | \alpha, \Omega_{D} )$
    Fcss = originalHist.Clone("Fcss_%s_%.2f_%.2f_clone" % (name, omega, shapeval))

    alpha = shapeval
    x = np.array(map(Fcss.GetXaxis().GetBinCenter, range(1, Fcss.GetNbinsX() + 1)))
    y = np.power(alpha/omega, alpha) / gamma(x) * np.power(x, alpha - 1) * np.exp(- alpha * x / omega)
    root_numpy.array2hist(y, Fcss)

    # Now, let's do some convolving!
    jssVar_conv = get_convolution(originalHist, Fcss)
    normalise(jssVar_conv)
    normalise(originalHist)

    # Make the CDFs
    jssVar_CDF      = get_cdf(originalHist)
    jssVar_conv_CDF = get_cdf(jssVar_conv)

    # Let F = CDF of original function and G = CDF of convolution
    # Then, our map shoud be G^{-1}(F(X)) (http://math.arizona.edu/~jwatkins/f-transform.pdf page 3)
    fxvals = jssVar_CDF.GetXaxis().GetXbins()
    fyvals = []
    ginvxvals = []
    ginvyvals = jssVar_CDF.GetXaxis().GetXbins()

    for i in range(1,jssVar_CDF.GetNbinsX()+1):
        fyvals +=    [jssVar_CDF.GetBinContent(i)]
        ginvxvals += [jssVar_conv_CDF.GetBinContent(i)]
        pass

    # Cumulative distribution of jssVar for low mass bin
    F    = ROOT.TGraph(len(fxvals),    np.array(fxvals),    np.array(fyvals))
    Ginv = ROOT.TGraph(len(ginvxvals), np.array(ginvxvals), np.array(ginvyvals))

    return F, Ginv


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
