#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for training DDT transform."""

# Basic import(s)
import gzip
import pickle
from array import array
import itertools

# Scientific import(s)
import ROOT
import root_numpy
import numpy as np
from scipy.special import gamma

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir, latex, wpercentile
from adversarial.profile import profile, Profile
from adversarial.constants import *
from tests.studies.common import *

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
    profile0 = kde(profile0)
    normalise(profile0, density=True)

    # Perform the optimisation
    bestShapeVal = 0
    bestSumChi2 = 1e20
    for shapeVal in SHAPEVAL_RANGE:
        print "Shape value: ", shapeVal
        sumChi2 = 0.

        # Each mass bin needs to be optimized over omega
        for mass in range(len(MASS_BINS)-1):
            print "   Mass bin: ", mass

            # Get 1D profile for current mass bin
            profile = profile2d.ProjectionY("%s_bin_%i"%(profile2d.GetName(),mass),mass+1, mass+1)

            # Fit current profile to low-mass profile
            chi2, bestOmega, _, _ = fit(profile, shapeVal, profile0, "%.2f"%mass)

            # Accumulate chi2
            sumChi2 += chi2
            pass

        # Update chi2 for current `shapeVal`
        print "-- sumChi2: {} (cp. {})".format(sumChi2, bestSumChi2)
        if sumChi2 < bestSumChi2:
            bestSumChi2  = sumChi2
            bestShapeVal = shapeVal
            pass
        pass

    # Saving CSS transforms
    with Profile("Saving CSS transform"):

        # Ensure model directory exists
        mkdir('models/css/')

        # Get the optimal, measured `omega`s for each mass-bin
        bestOmegas = list()
        for mass in range(len(MASS_BINS)-1):
            profile = profile2d.ProjectionY("%s_bin_%i_final"%(profile2d.GetName(),mass),mass+1, mass+1)
            sumChi2, bestOmega, profile_css, profile0rebin = fit(profile, bestShapeVal, profile0, "%.2f"%mass)

            # Test-plot distributions used for fitting!
            # -- Canvas
            c = rp.canvas(batch=True)

            # -- Plot
            profile = kde(profile)
            normalise(profile, density=True)

            lowmassbin = "#it{{m}} #in  [{:.1f}, {:.1f}] GeV".format(MASS_BINS[0],    MASS_BINS[1])     .replace('.0', '')
            massbin    = "#it{{m}} #in  [{:.1f}, {:.1f}] GeV".format(MASS_BINS[mass], MASS_BINS[mass+1]).replace('.0', '')
            c.hist(profile0rebin, label=latex(var, ROOT=True) + ",    {}".format(lowmassbin),
                   linecolor=rp.colours[1], fillcolor=rp.colours[1], alpha=0.5, option='HISTL', legend_option='FL')
            c.hist(profile,       label=latex(var, ROOT=True) + ",    {}".format(massbin),
                   linecolor=rp.colours[4], linestyle=2, option='HISTL')
            c.hist(profile_css,   label=latex(var + 'CSS', ROOT=True) + ", {}".format(massbin),
                   linecolor=rp.colours[3], option='HISTL')

            # -- Decorations
            c.xlabel(latex(var, ROOT=True) + ", " + latex(var + 'CSS', ROOT=True))
            c.ylabel("Number of jets p.d.f.")
            c.legend(xmin=0.45, ymax=0.76, width=0.25)
            c.text(["#sqrt{s} = 13 TeV,  Multijets",
                    "KDE smoothed"], qualifier=QUALIFIER)
            c.pad()._xaxis().SetTitleOffset(1.3)
            c.pad()._yaxis().SetNdivisions(105)
            c.pad()._primitives[-1].Draw('SAME AXIS')
            c.padding(0.50)


            # -- Save
            c.save('figures/css/css_test_{}_mass{}.pdf'.format(var, mass))

            # Store best-fit omega in array
            print mass, bestOmega
            bestOmegas.append(bestOmega)
            pass

        # Fit best omega vs. mass
        x = MASS_BINS[:-1] + 0.5 * np.diff(MASS_BINS)
        y = np.array(bestOmegas)

        h = ROOT.TH1F('hfit', "", len(MASS_BINS) - 1, MASS_BINS)
        root_numpy.array2hist(y, h)
        for ibin in range(1, len(x) + 1):
            h.SetBinError(ibin, 0.02)  # Just some value to ensure equal errors on all points
            pass

        m0 = 0.5 * (MASS_BINS[0] + MASS_BINS[1])
        f = ROOT.TF1("fit","[0] * (1./{m0}  - 1./x) + [1] * TMath::Log(x/{m0})".format(m0=m0), m0, 300);
        f.SetLineColor(rp.colours[4])
        f.SetLineStyle(2)
        h.Fit(f)

        # Write out the optimal configuration for each mass bin
        for mass in range(len(MASS_BINS)-1):
            profile = profile2d.ProjectionY("%s_bin_%i_final"%(profile2d.GetName(),mass),mass+1, mass+1)
            profile = kde(profile)
            normalise(profile, density=True)
            bestOmegaFitted_ = f.Eval(h.GetBinCenter(mass + 1)) + np.finfo(float).eps
            bestOmegaFitted = max(bestOmegaFitted_, 1E-04)
            #bestOmegaFitted = h.GetBinContent(mass + 1)
            print "bestOmegaFitted[{}] = {} --> {}".format(mass, bestOmegaFitted_, bestOmegaFitted)
            F,Ginv = get_css_fns(bestShapeVal, bestOmegaFitted, profile, "")

            # Save classifier
            saveclf(F,    'models/css/css_%s_F_%i.pkl.gz' % (var,mass))
            saveclf(Ginv, 'models/css/css_%s_Ginv_%i.pkl.gz' % (var,mass))
            pass

        # Plot best omega vs. mass
        # -- Canvas
        c = rp.canvas(batch=True)

        # -- Plots
        #c.hist(bestOmegas, bins=MASS_BINS, linecolor=rp.colours[1])
        c.hist(h, linecolor=rp.colours[1], option='HIST', label="Measured")
        f.Draw('SAME')

        # -- Decorations
        c.xlabel("Large-#it{R} jet mass [GeV]")
        c.ylabel("Best-fit #Omega_{D}")
        c.text(["#sqrt{s} = 13 TeV,  Multijets",
                "CSS applied to {}".format(latex(var, ROOT=True)),
                "Best-fit #alpha = {:.1f}".format(bestShapeVal)], qualifier=QUALIFIER)
        c.legend(categories=[('Functional fit', {'linewidth': 2, 'linestyle': 2, 'linecolor': rp.colours[4]})])
        # Save
        c.save('figures/css/cssBestOmega_{}.pdf'.format(var))
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
    normalise(rebinLowMass, density=True)

    # Find optimal value for omega for this mass bin
    for omega in OMEGA_RANGE:
        jssVar_css = get_css(shapeVal, omega, profile, "_%.2f"%(omega))
        jssVar_css = kde(jssVar_css)
        normalise(jssVar_css, density=True)

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
    jssVar_css = kde(jssVar_css)
    normalise(jssVar_css, density=True)

    return bestChi2, bestOmega, jssVar_css, rebinLowMass


def get_css (shapeval, omega, originalHist, name):
    """
    ...
    """
    F, Ginv = get_css_fns(shapeval, omega, originalHist, name)
    jssVar_css = ROOT.TH1D("css%s_%.3f_%.3f_%s"%(name, omega, shapeval,originalHist.GetName()),"low_m_css",originalHist.GetNbinsX(),0,originalHist.GetXaxis().GetXmax())

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
