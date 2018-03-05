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

from sklearn.linear_model import LinearRegression

# Project import(s)
from adversarial.profile import profile, Profile
from adversarial.new_utils import parse_args, initialise, load_data, mkdir
from adversarial.constants import *

# Custom import(s)
#import rootplotting as rp

# Local import(s)
from .common import *

def fill_css (data, var, mass, css_f, css_ginv):
    profile = ROOT.TH1F('profile_{}_{}'.format(var,mass), "", len(BINS) - 1, BINS)
    tau21Data = data[var].as_matrix().flatten()
    massData = data['m'].as_matrix().flatten()
    weightData = data['weight'].as_matrix().flatten()

    for cdata,ctau,cweight in zip(massData, tau21Data, weightData):
      if cdata > MASS_BINS[mass] and cdata < MASS_BINS[mass+1]:
        ctauOld = ctau
        ctau = ApplyCSS(ctau, css_ginv, css_f)
        #print ctauOld, ctau, cdata, mass
        profile.Fill(ctau,cweight)

    return profile


def fit(profile, shapeVal, lowMassProfile, name):
  if(profile.Integral() > 0):
    profile.Scale(1. / profile.Integral("width"))

  if(lowMassProfile.Integral() > 0):
    lowMassProfile.Scale(1. / lowMassProfile.Integral("width"))

  #These are free parameters - need to tune these.
  bestOmega = 0.01
  bestChi2 = 1e30
  rebinLowMass = lowMassProfile.Clone("%s_clone"%name)
  rebinLowMass.Rebin(5)
  rebinLowMass.Scale(1. / rebinLowMass.Integral())
  

  # Find optimal value for omega for this mass bin
  for omega in OMEGA_RANGE:
    D2histo_css = getCSS(shapeVal, omega, profile, "_%.2f"%(omega))
    #lowMassProfile.Scale(1. / lowMassProfile.Integral())
    D2histo_css.Rebin(5)
    D2histo_css.Scale(1. / D2histo_css.Integral(1, 300))
    c1 = rp.canvas(batch=True)
    c1.hist(rebinLowMass, label="Low Mass", linecolor=rp.colours[1], markercolor=rp.colours[1])
    c1.hist(D2histo_css, label="D2", linecolor=rp.colours[2], markercolor=rp.colours[2])
    c1.legend()

    c1.save('figures/css/cssProfile_{}_{}_{}.pdf'.format(name, omega, shapeVal))


    chi2 = 0
    for i in range(1, D2histo_css.GetNbinsX()+1):
      #Want to compare to the low-mass region. Obviously this isn't quite how chi2 works, but it works for a basic optimization
      #if(lowMassProfile.GetBinContent(i) > 0):
      if(rebinLowMass.GetBinContent(i) > 0):
        chi2 += (rebinLowMass.GetBinContent(i) - D2histo_css.GetBinContent(i))*(rebinLowMass.GetBinContent(i) - D2histo_css.GetBinContent(i))

    if chi2 < bestChi2 :
      bestChi2 = chi2
      bestOmega = omega

  print "Best omega for shapeVal of ",shapeVal, " = ",bestOmega
  return bestChi2, bestOmega

def getCSS(shapeval, omega, originalHist, name):
  F, Ginv = getCSSFns(shapeval, omega, originalHist, name)
  D2histo_css = ROOT.TH1D("css%s"%(name),"low_m_css",originalHist.GetNbinsX(),0,MAX_D2)

  # Apply the convolutions to get the new distributions
  for i in range(1, originalHist.GetNbinsX()+1):
    lowMValBkg = ApplyCSS(originalHist.GetBinCenter(i), Ginv, F)
    #print omega, shapeval
    D2histo_css.Fill(lowMValBkg, originalHist.GetBinContent(i))

  return D2histo_css


def getCDF(hist):
  hist_CDF = hist.Clone("%s_CDF"%hist.GetName())

  for i in range(1,hist.GetNbinsX()+1):
    hist_CDF.SetBinContent(i, hist.Integral(1,i))
  return hist_CDF

def getConvolution(hist, shapefunc):
  #Now, let's do some convolving!
  hist_conv = hist.Clone("%s_conv"%hist.GetName())
  for i in range(0,shapefunc.GetNbinsX()+1):
    mysum = 0.
    for j in range(0,shapefunc.GetNbinsX()+1):
      if (i-j >= 0):
        mysum+=MAX_D2 / shapefunc.GetNbinsX() * hist.GetBinContent(i-j+1)*shapefunc.GetBinContent(j+1)
    hist_conv.SetBinContent(i+1,mysum)
  return hist_conv


def getCSSFns(shapeval, omega, originalHist, name):
  shapefunc = originalHist.Clone()

  for i in range(1,shapefunc.GetNbinsX()+1):
    xx = shapefunc.GetXaxis().GetBinCenter(i)
    myval = 0
    if( ROOT.Math.tgamma(shapeval) != 0 and (omega)**shapeval != 0):
      myval = (shapeval**shapeval/ROOT.Math.tgamma(shapeval))*(xx**(shapeval-1))*np.exp(-shapeval*xx/omega) / (omega)**shapeval
    shapefunc.SetBinContent(i,myval)

  #Now, let's do some convolving!
  D2histo_conv = getConvolution(originalHist, shapefunc)
  #D2histo_conv.Scale(1. / D2histo_conv.Integral("width"))
  if D2histo_conv.Integral() > 0:
    D2histo_conv.Scale(1. / D2histo_conv.Integral())

  if originalHist.Integral() > 0:
    originalHist.Scale(1. / originalHist.Integral())
  # Make the CDFs
  D2histo_CDF = getCDF(originalHist)
  D2histo_conv_CDF = getCDF(D2histo_conv)
  
  #Let F = CDF of original function and G = CDF of convolution
  #Then, our map shoud be G^{-1}(F(X)) (http://math.arizona.edu/~jwatkins/f-transform.pdf page 3)
  fxvals = []
  fyvals = []
  ginvxvals = []
  ginvyvals = []

  for i in range(1,D2histo_CDF.GetNbinsX()+1):
    xx = D2histo_CDF.GetXaxis().GetBinCenter(i)
    fxvals+=[xx]
    ginvyvals+=[xx]
    fyvals+=   [D2histo_CDF.GetBinContent(i)]
    ginvxvals+=[D2histo_conv_CDF.GetBinContent(i)]

  #Cumulative distribution for low mass bin D2
  F = ROOT.TGraph(len(fxvals),np.array(fxvals),np.array(fyvals))
  Ginv = ROOT.TGraph(len(ginvxvals),np.array(ginvxvals),np.array(ginvyvals))
  return F, Ginv

# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    #data, features, _ = load_data(args.input + 'data.h5')
    data, features, _ = load_data('data/data.h5')
    data = data[(data['train'] == 1) & (data['signal'] == 0)]


    # Adding variable(s)
    # --------------------------------------------------------------------------
    add_variables(data)


    # Filling Tau21 profile
    # --------------------------------------------------------------------------

    var = 'D2'
    #var = 'Tau21'
    profile0 = fill_profile(data, var, 0)
    # Do the optimization
    bestShapeVal = 0
    bestSumChi2 = 1e20
    for shapeVal in SHAPEVAL_RANGE:
      sumChi2 = 0

      # Each mass bin needs to be optimized over omega
      for mass in range(len(MASS_BINS)-1):
        profile = fill_profile(data, var, mass)
         
        name = "%.2f"%mass
        sumChi2, bestOmega = fit(profile, shapeVal, profile0, name)

      if sumChi2 < bestSumChi2:
        bestSumChi2 = sumChi2
        bestShapeVal = shapeVal



    # Saving DDT transform
    # --------------------------------------------------------------------------
    with Profile("Saving CSS transform"):

        # Ensure model directory exists
        mkdir('models/css/')
        # Write out the optimal configuration for each mass bin
        for mass in range(len(MASS_BINS)-1):
            profile = fill_profile(data, var, mass)
            name = "%.2f"%mass
            sumChi2, bestOmega = fit(profile, bestShapeVal, profile0, name)
            print MASS_BINS[mass], MASS_BINS[mass+1], mass, bestOmega, bestShapeVal
            F,Ginv = getCSSFns(bestShapeVal, bestOmega, profile, "")

            # Save classifier
            with gzip.open('models/css/css_%s_F_%i.pkl.gz'%(var,mass), 'w') as f:
                pickle.dump(F, f)
            with gzip.open('models/css/css_%s_Ginv_%i.pkl.gz'%(var,mass), 'w') as f2:
                pickle.dump(Ginv, f2)
                pass
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
