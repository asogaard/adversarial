#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common methods for training and testing CSS transform."""

# Scientific import(s)
import ROOT
import root_numpy
import numpy as np
import pandas as pd
import gzip
import pickle

# Project import(s)
from adversarial.profile import profile


# Common definition(s)

SHAPEVAL_RANGE = np.linspace(1., 3., 3)
OMEGA_RANGE = np.linspace(0.01, 1.4, 40)
MASS_BINS = np.linspace(40., 310., 20)

TAU21BINS = np.linspace(0., 2., 501, endpoint=True)
D2BINS = np.linspace(0., 5., 501, endpoint=True)

# Adds the CSS variable to the data (assuming Ginv, F files exist)
def AddCSS(jssVar, data):
  data['%sCSS'%jssVar] = GetCSSSeries(jssVar, data)

# Applies CSS, given a value, and the convolution functions
def ApplyCSS(jssVar, Ginv, F):
  newJSSVar = Ginv.Eval(F.Eval(jssVar))
  return newJSSVar


@profile
def fill_2d_profile (data, jssVar, jssBins, massVar, massBins):
    """Fill ROOT.TProfile with the average `jssVar` as a function of rhoCSS."""
    profile2d = ROOT.TH2F('profile2d_{}'.format(jssVar), "", len(massBins)-1, massBins, len(jssBins) - 1, jssBins)

    jssVarData = data[jssVar].as_matrix().flatten()
    massData = data[massVar].as_matrix().flatten()
    weightData = data['weight'].as_matrix().flatten()

    for cmass,ctau,cweight in zip(massData, jssVarData, weightData):
      profile2d.Fill(cmass, ctau, cweight)

    return profile2d

def GetCSSSeries(jssVar, data):
  massData = data['m'].as_matrix().flatten()
  jssData = data[jssVar].as_matrix().flatten()
  massbins = np.digitize(massData, MASS_BINS)-1

  F_massbins = getFunction(jssVar, "F")
  Ginv_massbins = getFunction(jssVar, "Ginv")

  newJSSVars = []
  for jssVal, massBin in zip(jssData, massbins):
    if massBin>= len(MASS_BINS):
      massBin = len(MASS_BINS)

    newJSSVal = ApplyCSS(jssVal, Ginv_massbins[massBin], F_massbins[massBin])
    newJSSVars.append(newJSSVal)

  jssSeries = pd.Series(newJSSVars, index=data.index)
  return jssSeries


#Returns the convolution of hist and shapefunc
def getConvolution(hist, shapefunc):
  hist_conv = hist.Clone("%s_conv"%hist.GetName())
  for i in range(0,shapefunc.GetNbinsX()+1):
    mysum = 0.
    for j in range(0,shapefunc.GetNbinsX()+1):
      if (i-j >= 0):
        mysum+=hist.GetXaxis().GetXmax() / shapefunc.GetNbinsX() * hist.GetBinContent(i-j+1)*shapefunc.GetBinContent(j+1)
    hist_conv.SetBinContent(i+1,mysum)
  return hist_conv

# Gets the cumulative distribution of hist
def getCDF(hist):
  hist_CDF = hist.Clone("%s_CDF"%hist.GetName())

  for i in range(1,hist.GetNbinsX()+1):
    hist_CDF.SetBinContent(i, hist.Integral(1,i))
  return hist_CDF

def getFunction(jssVar, funcName):
  css_func_all = []
  for m in range(len(MASS_BINS)-1):
    with gzip.open('models/css/css_%s_%s_%i.pkl.gz'%(jssVar,funcName,m), 'r') as func:
      css_func_all.append(pickle.load(func))

  return css_func_all
