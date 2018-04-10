#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common methods for training and testing DDT transform."""

# Basic import(s)
import gzip

# Scientific import(s)
import ROOT
import root_numpy
import numpy as np
import pandas as pd

# Project import(s)
from adversarial.utils import loadclf
from adversarial.profile import profile

# Common definition(s)
BINS = np.linspace(-1, 6, 7 * 4 + 1, endpoint=True)  # Binning in rhoDDT
FIT_RANGE = (1.5, 4.0) # Range in rhoDDT to be fitted


@profile
def add_ddt (data, feat='Tau21', newfeat=None, path='models/ddt/ddt.pkl.gz'):
    """
    Add DDT-transformed `feat` to `data`. Modifies `data` in-place.

    Arguments:
        data: Pandas DataFrame to which to add the DDT-transformed variable.
        feat: Substructure variable to be decorrelated.
        newfeat: Name of output featur. By default, `{feat}DDT`.
        path: Path to trained DDT transform model
    """

    # Check(s)
    if newfeat is None:
        newfeat = '{}DDT'.format(feat)
        pass

    # Load model
    ddt = loadclf(path)

    # Add new classifier to data array
    data[newfeat] = pd.Series(data[feat] - ddt.predict(data[['rhoDDT']].values), index=data.index)
    return


@profile
def fill_profile (data, var):
    """
    Fill ROOT.TProfile with the average `var` as a function of rhoDDT.
    """

    profile = ROOT.TProfile('profile_{}'.format(var), "", len(BINS) - 1, BINS)
    root_numpy.fill_profile(profile, data[['rhoDDT', var]].values, weights=data['weight_test'].values)
    return profile
