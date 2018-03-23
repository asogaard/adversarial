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
import rootplotting as rp

# Project import(s)
from adversarial.profile import profile, Profile
from adversarial.new_utils import parse_args, initialise, load_data, mkdir
from adversarial.constants import *

# Custom import(s)
#import rootplotting as rp

# Local import(s)
from .common import *

def fit(profile, shapeVal, lowMassProfile, name):
  if(profile.Integral() != 0):
    profile.Scale(1. / profile.Integral("width"))

  if(lowMassProfile.Integral() > 0):
    lowMassProfile.Scale(1. / lowMassProfile.Integral("width"))

  #These are free parameters - need to tune these.
  bestOmega = 0.01
  bestChi2 = 1e30
  rebinLowMass = lowMassProfile.Clone("%s_%.2f_clone"%(name, shapeVal))
  rebinLowMass.Rebin(5)
  rebinLowMass.Scale(1. / rebinLowMass.Integral())

  # Find optimal value for omega for this mass bin
  for omega in OMEGA_RANGE:
    jssVar_css = getCSS(shapeVal, omega, profile, "_%.2f"%(omega))
    jssVar_css.Rebin(5)
    if(jssVar_css.Integral() != 0):
      jssVar_css.Scale(1. / jssVar_css.Integral())
    #c1 = rp.canvas(batch=True)
    #c1.hist(rebinLowMass, label="Low Mass", linecolor=rp.colours[1], markercolor=rp.colours[1])
    #c1.hist(jssVar_css, label="D2", linecolor=rp.colours[2], markercolor=rp.colours[2])
    #c1.legend()
    #c1.save('figures/css/cssProfile_{}_{}_{}.pdf'.format(name, omega, shapeVal))

    chi2 = 0
    for i in range(1, jssVar_css.GetNbinsX()+1):
      #Want to compare to the low-mass region. Obviously this isn't quite how chi2 works, but it works for a basic optimization
      if(rebinLowMass.GetBinContent(i) > 0):
        chi2 += (rebinLowMass.GetBinContent(i) - jssVar_css.GetBinContent(i))*(rebinLowMass.GetBinContent(i) - jssVar_css.GetBinContent(i))

    if chi2 < bestChi2 :
      bestChi2 = chi2
      bestOmega = omega

  jssVar_css = getCSS(shapeVal, bestOmega, profile, "_%.2f_best"%(bestOmega))
  jssVar_css.Rebin(5)
  if(jssVar_css.Integral() != 0):
    jssVar_css.Scale(1. / jssVar_css.Integral())

  return bestChi2, bestOmega, jssVar_css, rebinLowMass

def getCSS(shapeval, omega, originalHist, name):
  F, Ginv = getCSSFns(shapeval, omega, originalHist, name)
  jssVar_css = ROOT.TH1D("css%s_%.2f_%.2f_%s"%(name, omega, shapeval,originalHist.GetName()),"low_m_css",originalHist.GetNbinsX(),0,originalHist.GetXaxis().GetXmax())

  # Apply the convolutions to get the new distributions
  for i in range(1, originalHist.GetNbinsX()+1):
    lowMValBkg = ApplyCSS(originalHist.GetBinCenter(i), Ginv, F)
    jssVar_css.Fill(lowMValBkg, originalHist.GetBinContent(i))

  return jssVar_css

def getCSSFns(shapeval, omega, originalHist, name):
  shapefunc = originalHist.Clone("shapefunc_%s_%.2f_%.2f_clone"%(name, omega, shapeval))

  for i in range(1,shapefunc.GetNbinsX()+1):
    xx = shapefunc.GetXaxis().GetBinCenter(i)
    myval = 0
    if( ROOT.Math.tgamma(shapeval) != 0 and (omega)**shapeval != 0):
      myval = (shapeval**shapeval/ROOT.Math.tgamma(shapeval))*(xx**(shapeval-1))*np.exp(-shapeval*xx/omega) / (omega)**shapeval
    shapefunc.SetBinContent(i,myval)

  #Now, let's do some convolving!
  jssVar_conv = getConvolution(originalHist, shapefunc)
  if jssVar_conv.Integral() > 0:
    jssVar_conv.Scale(1. / jssVar_conv.Integral())

  if originalHist.Integral() > 0:
    originalHist.Scale(1. / originalHist.Integral())
  # Make the CDFs
  jssVar_CDF = getCDF(originalHist)
  jssVar_conv_CDF = getCDF(jssVar_conv)
  
  #Let F = CDF of original function and G = CDF of convolution
  #Then, our map shoud be G^{-1}(F(X)) (http://math.arizona.edu/~jwatkins/f-transform.pdf page 3)
  fxvals = jssVar_CDF.GetXaxis().GetXbins()
  fyvals = []
  ginvxvals = []
  ginvyvals = jssVar_CDF.GetXaxis().GetXbins()

  for i in range(1,jssVar_CDF.GetNbinsX()+1):
    fyvals+=   [jssVar_CDF.GetBinContent(i)]
    ginvxvals+=[jssVar_conv_CDF.GetBinContent(i)]

  #Cumulative distribution of jssVar for low mass bin
  F = ROOT.TGraph(len(fxvals),np.array(fxvals),np.array(fyvals))
  Ginv = ROOT.TGraph(len(ginvxvals),np.array(ginvxvals),np.array(ginvyvals))
  return F, Ginv

def doOptimization(jssVar, jssBins, data):
    profile2d = fill_2d_profile(data, jssVar, jssBins, "m", MASS_BINS)
    profile0 = profile2d.ProjectionY("%s_lowMass"%profile2d.GetName(),1, 1)

    # Do the optimization
    bestShapeVal = 0
    bestSumChi2 = 1e20
    for shapeVal in SHAPEVAL_RANGE:
      print "Shape value: ", shapeVal
      sumChi2 = 0

      # Each mass bin needs to be optimized over omega
      for mass in range(len(MASS_BINS)-1):
        print "Mass bin: ", mass
        profile = profile2d.ProjectionY("%s_bin_%i"%(profile2d.GetName(),mass),mass+1, mass+1)
        sumChi2, bestOmega, _,_ = fit(profile, shapeVal, profile0, "%.2f"%mass)

      if sumChi2 < bestSumChi2:
        bestSumChi2 = sumChi2
        bestShapeVal = shapeVal

    # Saving CSS transforms
    # --------------------------------------------------------------------------
    # Currently need to do everythin in mass bins
    with Profile("Saving CSS transform"):

        # Ensure model directory exists
        mkdir('models/css/')
        # Write out the optimal configuration for each mass bin
        for mass in range(len(MASS_BINS)-1):
            profile = profile2d.ProjectionY("%s_bin_%i_final"%(profile2d.GetName(),mass),mass+1, mass+1)
            sumChi2, bestOmega, profile_css, profile0rebin= fit(profile, bestShapeVal, profile0, "%.2f"%mass)
            print mass, bestOmega
            F,Ginv = getCSSFns(bestShapeVal, bestOmega, profile, "")
            c1 = rp.canvas(batch=True)

            c1.hist(profile0rebin, label="Low Mass", linecolor=rp.colours[1], markercolor=rp.colours[1])
            c1.hist(profile_css, label="D2 CSS", linecolor=rp.colours[2], markercolor=rp.colours[2])
            rebinProfiles = profile.Clone("%s_%i_rebin"%(profile.GetName(), mass))
            rebinProfiles.Rebin(5)
            if(rebinProfiles.Integral() != 0):
              rebinProfiles.Scale(1. / rebinProfiles.Integral())
            c1.hist(rebinProfiles, label="D2", linecolor=rp.colours[3], markercolor=rp.colours[3])
            c1.legend()
            c1.save('figures/css/cssProfile_{}_{}.pdf'.format(jssVar, mass))

            # Save classifier
            with gzip.open('models/css/css_%s_F_%i.pkl.gz'%(jssVar,mass), 'w') as f:
                pickle.dump(F, f)
            with gzip.open('models/css/css_%s_Ginv_%i.pkl.gz'%(jssVar,mass), 'w') as f2:
                pickle.dump(Ginv, f2)

    return 0

# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)

    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data(args.input + 'data.h5')
    #data, features, _ = load_data('data/data.h5')
    
    data = data[(data['train'] == 1) & (data['signal'] == 0)]

    # Filling Tau21 profile
    # --------------------------------------------------------------------------

    D2BINS = np.linspace(0., 5., 501, endpoint=True)
    doOptimization("D2", D2BINS, data)

    #TAU21BINS = np.linspace(0., 2., 501, endpoint=True)
    #doOptimization("Tau21", TAU21BINS, data)


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
