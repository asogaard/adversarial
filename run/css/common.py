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
MAX_D2 = 5.
BINS = np.linspace(0., MAX_D2, 501, endpoint=True)  # Binning in rhoCSS
SHAPEVAL_RANGE = np.linspace(1., 3., 2)
OMEGA_RANGE = np.linspace(0.01, 0.8, 20)
MASS_BINS = np.linspace(40., 300., 12)
RHO_BINS = np.linspace(-7, -0.5, 7 * 8 + 1, endpoint=True)

@profile
def add_variables (data):
    """Add necessary variable(s).
    Modify data container in-place."""

    # rhoCSS
    data['rho'] = pd.Series(np.log(np.square(data['m'])/np.square(data['pt'])), index=data.index)
    #data['rhoCSS'] = pd.Series(data['m'], index=data.index)
    return


@profile
def fill_profile (data, var, mass):
    """Fill ROOT.TProfile with the average `var` as a function of rhoCSS."""
    profile = ROOT.TH1F('profile_{}_{}'.format(var,mass), "", len(BINS) - 1, BINS)

    tau21Data = data[var].as_matrix().flatten()
    massData = data['m'].as_matrix().flatten()
    weightData = data['weight'].as_matrix().flatten()
    i=0
    for cmass,ctau,cweight in zip(massData, tau21Data, weightData):
      if cmass > MASS_BINS[mass] and cmass < MASS_BINS[mass+1]:
        profile.Fill(ctau,cweight)
        i+=1

    print "Number of entries: ", i
    return profile

def fill_mass_profile (data, var):
    """Fill ROOT.TProfile with the average `var` as a function of rhoCSS."""
    profile = ROOT.TH1F('profile_{}_{}'.format(var,mass), "", len(BINS) - 1, BINS)

    for mass in range(len(BINS)-1):
      massProf = fill_profile(data, var, mass)
      profile.SetBinContent(mass,massProf.GetMean())


    return profile

def getGinv():
  css_ginv_all = []
  for m in range(len(MASS_BINS)-1):
    with gzip.open('models/css/css_Ginv_%i.pkl.gz'%m, 'r') as ginv:
      css_ginv = pickle.load(ginv)
      css_ginv_all.append(css_ginv)
  return css_ginv_all

def getF():
  css_f_all = []
  for m in range(len(MASS_BINS)-1):
    with gzip.open('models/css/css_F_%i.pkl.gz'%m, 'r') as f:
      css_f = pickle.load(f)
      css_f_all.append(css_f)
  return css_f_all

def ApplyCSSAgain(d2, massbins, Ginv, F):
  
  newD2s = []
  short = range(0, 3)
  #for cD2,cGinv,cF in zip(d2, Ginv, F):
  #for cD2, massbin, tmp in zip(d2[0:3], massbins[0:3], short):
  for i in range(len(d2)):
    if massbins[i]> 11:
      massbins[i]= 11
    newD2 = Ginv[massbins[i]].Eval(F[massbins[i]].Eval(d2[i]))
    newD2s.append(newD2)
    #print d2, newD2
  
  return newD2s

def GetCSSSeries(jssVar, data):
  massData = data['m'].as_matrix().flatten()
  jssData = data[jssVar].as_matrix().flatten()
  massbins = np.digitize(massData, MASS_BINS)-1
  F_massbins = getF()
  Ginv_massbins = getGinv()

  newJSSVars = []
  for i in range(len(jssData)):
    if massbins[i]> 11:
      massbins[i]= 11
    newJSSVar = Ginv_massbins[massbins[i]].Eval(F_massbins[massbins[i]].Eval(jssData[i]))
    newJSSVars.append(newJSSVar)

  jssSeries = pd.Series(newJSSVars, index=data.index)
  return jssSeries



def ApplyCSS(d2, Ginv, F):
  newD2 = Ginv.Eval(F.Eval(d2))
  return newD2

