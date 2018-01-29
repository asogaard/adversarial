#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common methods for training and testing DDT transform."""

# Scientific import(s)
import ROOT
import root_numpy
import numpy as np
import pandas as pd

# Project import(s)
from adversarial.profile import profile


# Common definition(s)
BINS = np.linspace(-1, 6, 7 * 8 + 1, endpoint=True)  # Binning in rhoDDT
FIT_RANGE = (1.5, 4.0)  # Range in rhoDDT to be fitted


@profile
def add_variables (data):
    """Add necessary variable(s).
    Modify data container in-place."""

    # rhoDDT
    data['rhoDDT'] = pd.Series(np.log(np.square(data['m'])/(data['pt'] * 1.)), index=data.index)
    return


@profile
def fill_profile (data, var):
    """Fill ROOT.TProfile with the average `var` as a function of rhoDDT."""
    profile = ROOT.TProfile('profile_{}'.format(var), "", len(BINS) - 1, BINS)
    root_numpy.fill_profile(profile, data[['rhoDDT', var]].as_matrix(), weights=data['weight'].as_matrix().flatten())
    return profile
