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
SHAPEVAL_RANGE = np.linspace(1., 3., 3)
OMEGA_RANGE = np.linspace(0.01, 1.2, 30)
MASS_BINS = np.linspace(40., 310., 30)
RHO_BINS = np.linspace(-7, -0.5, 7 * 8 + 1, endpoint=True)

@profile
def add_variables (data):
    """Add necessary variable(s).
    Modify data container in-place."""

    # rhoCSS
    data['rho'] = pd.Series(np.log(np.square(data['m'])/np.square(data['pt'])), index=data.index)
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

    return profile

def getGinv(var):
  css_ginv_all = []
  for m in range(len(MASS_BINS)-1):
    with gzip.open('models/css/css_%s_Ginv_%i.pkl.gz'%(var,m), 'r') as ginv:
      css_ginv_all.append(pickle.load(ginv))

  return css_ginv_all

def getF(var):
  css_f_all = []
  for m in range(len(MASS_BINS)-1):
    with gzip.open('models/css/css_%s_F_%i.pkl.gz'%(var,m), 'r') as f:
      css_f_all.append(pickle.load(f))

  return css_f_all

def AddCSS(jssVar, data):
  data['%sCSS'%jssVar] = GetCSSSeries(jssVar, data)

def GetCSSSeries(jssVar, data):
  massData = data['m'].as_matrix().flatten()
  jssData = data[jssVar].as_matrix().flatten()
  massbins = np.digitize(massData, MASS_BINS)-1
  F_massbins = getF(jssVar)
  Ginv_massbins = getGinv(jssVar)

  newJSSVars = []
  for i in range(len(jssData)):
    if massbins[i]>= len(MASS_BINS):
      massbins[i] = len(MASS_BINS)
    newJSSVar = Ginv_massbins[massbins[i]].Eval(F_massbins[massbins[i]].Eval(jssData[i]))
    newJSSVars.append(newJSSVar)

  jssSeries = pd.Series(newJSSVars, index=data.index)
  return jssSeries

# Applies CSS, given a value, and the convolution functions
def ApplyCSS(jssVar, Ginv, F):
  newD2 = Ginv.Eval(F.Eval(jssVar))
  return newD2

