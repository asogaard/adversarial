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
from adversarial.utils import loadclf, saveclf
from adversarial.profile import profile


# Common definition(s)
eps = np.finfo(float).eps

SHAPEVAL_RANGE = np.linspace(0, 3.0, 3 * 2 + 1, endpoint=True)[1:]
OMEGA_RANGE = np.linspace(0., 1.0, 2 * 50 + 1, endpoint = True)[1:]
MASS_BINS = np.linspace(50., 300., 25 + 1, endpoint=True)

TAU21BINS = np.linspace(0., 2., 501, endpoint=True)
D2BINS = np.linspace(0., 5., 500 + 1, endpoint=True)
N2BINS = np.linspace(0, 0.6, 500 + 1, endpoint=True)

# Adds the CSS variable to the data (assuming Ginv, F files exist)
def add_css (jssVar, data):
    data['%sCSS'%jssVar] = get_css_series(jssVar, data)
    return

# Applies CSS, given a value, and the convolution functions
def apply_css (jssVar, Ginv, F):
  newJSSVar = Ginv.Eval(F.Eval(jssVar))
  return newJSSVar


@profile
def fill_2d_profile (data, jssVar, jssBins, massVar, massBins):
    """
    Fill ROOT.TProfile with the average `jssVar` as a function of rhoCSS.
    """
    profile2d = ROOT.TH2F('profile2d_{}'.format(jssVar), "", len(massBins)-1, massBins, len(jssBins) - 1, jssBins)

    jssVarData = data[jssVar].as_matrix().flatten()
    massData = data[massVar].as_matrix().flatten()
    weightData = data['weight_test'].as_matrix().flatten()

    for cmass,ctau,cweight in zip(massData, jssVarData, weightData):
        profile2d.Fill(cmass, ctau, cweight)
        pass

    return profile2d


def get_css_series (jssVar, data):
    """
    Perform the CSS transform on substructure variables `jssVar` in Pandas
    Dataframe `data`.

    Arguments:
        jssvar: Name of substructure variable to be transformed.
        data: Pandas.Dataframe holding the substructure variable and jet mass.

    Returns:
        Pandas.Series containing transformed variable.
    """

    massData = data['m'].as_matrix().flatten()
    jssData = data[jssVar].as_matrix().flatten()
    massbins = np.digitize(massData, MASS_BINS)-1

    F_massbins    = get_function(jssVar, "F")
    Ginv_massbins = get_function(jssVar, "Ginv")

    newJSSVars = []
    for jssVal, massBin in zip(jssData, massbins):
        if massBin>= len(MASS_BINS):
            massBin = len(MASS_BINS)
            pass

        newJSSVal = apply_css(jssVal, Ginv_massbins[massBin], F_massbins[massBin])
        newJSSVars.append(newJSSVal)
        pass

    jssSeries = pd.Series(newJSSVars, index=data.index)
    return jssSeries


def get_cdf (h):
    """
    Returns the cumulative distribution of `hist`.
    """

    # Prepare array(s)
    y = root_numpy.hist2array(h)

    # Compute CDF
    cdf = np.cumsum(y)

    # Convert to ROOT.TH1F
    hcdf = h.Clone("%s_CDF" % h.GetName())
    root_numpy.array2hist(cdf, hcdf)

    return hcdf


def get_convolution (h, Fcss):
    """
    Returns the convolution of `h` and `Fcss`.
    """

    # Check(s)
    N = Fcss.GetNbinsX()
    assert N == h.GetNbinsX()

    # Prepare array(s)
    f1 = root_numpy.hist2array(h)
    f2 = root_numpy.hist2array(Fcss)

    # Perform convolution
    conv = np.convolve(f1, f2)[:N]

    # Convert to ROOT.TH1F
    hconv = h.Clone("%s_conv" % h.GetName())
    root_numpy.array2hist(conv, hconv)

    return hconv


def get_function (jssVar, funcName):
    css_func_all = []
    for m in range(len(MASS_BINS)-1):
        clf = loadclf('models/css/css_%s_%s_%i.pkl.gz' % (jssVar, funcName, m))
        css_func_all.append(clf)
        pass

    return css_func_all


def normalise (p, density=False):
    """
    Normalise ROOT.TProfile `p` to p.d.f. in-place
    """
    if p.Integral():
        p.Scale(1. / p.Integral('width' if density else ''))
        pass
    return


def kde (original, scale=0.15):
    """
    Perform kernel density estimation (KDE), -ish, on input histogram `original`
    with gaussian kernel length scale `scale`.
    """

    # Clone histogram, for KDE
    h = original.Clone(original.GetName() + '_redistributed')

    # Rebinning
    if isinstance(scale, int):
        #print "kde: Rebinning instead of KDE-smoothing."
        h.Rebin(scale)
        return h

    # Smoothing
    h.Reset()

    # Define utility method(s)
    gaussian = lambda z: np.exp(-np.power(z,2.)/2.)

    # Perform KDE
    ibins = range(1, original.GetNbinsX() + 1)
    X = np.array(map(h.GetXaxis().GetBinCenter, ibins))
    for ibin in ibins:
        x = h.GetXaxis().GetBinCenter(ibin)
        c = original.GetBinContent(ibin)
        Z = np.abs(X - x) / scale

        weights = c * gaussian(Z)
        root_numpy.fill_hist(h, X, weights)
        pass

    # Normalise
    h.Scale(original.Integral()/h.Integral())

    return h
