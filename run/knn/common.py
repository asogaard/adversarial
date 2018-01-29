#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common methods for training and testing fixed-efficiency kNN regressor."""

# Basic import(s)
import itertools

# Scientific import(s)
import ROOT
import numpy as np
import pandas as pd

# Project import(s)
from adversarial.utils import wpercentile
from adversarial.profile import profile


# Common definition(s)
VAR  = 'D2'   # Substructure variable to decorrelate
EFF  = 20     # Fixed backround efficiency at which to perform decorrelation
VARX = 'rho'  # X-axis variable from which to decorrelate
VARY = 'pt'   # Y-axis variable from which to decorrelate
AXIS = {      # Dict holding (num_bins, axis_min, axis_max) for axis variables
    'rho': (20, -5.5, -2.0),
    'pt':  (20, 200., 2000.),
}


@profile
def add_variables (data):
    """Add necessary variable(s).
    Modify data container in-place."""

    # rho
    data['rho'] = pd.Series(np.log(np.square(data['m']) / np.square(data['pt'])), index=data.index)
    return


@profile
def fill_profile (data):
    """Fill ROOT.TH2F with the measured, weighted values of the `EFF`-percentile
    of the background `VAR`. """

    # Define arrays
    shape = (AXIS[VARX][0], AXIS[VARY][0])
    binsx = np.linspace(AXIS[VARX][1], AXIS[VARX][2], AXIS[VARX][0] + 1, endpoint=True)
    binsy = np.linspace(AXIS[VARY][1], AXIS[VARY][2], AXIS[VARY][0] + 1, endpoint=True)
    x = np.zeros(shape)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    # Create `profile` histogram
    profile = ROOT.TH2F('profile', "", len(binsx) - 1, binsx.flatten('C'), len(binsy) - 1, binsy.flatten('C'))

    # Fill profile
    for i,j in itertools.product(*map(range, shape)):

        # Bin edges in x and y
        xmin, xmax = binsx[i:i+2]
        ymin, ymax = binsy[j:j+2]

        # Masks
        mskx = (data[VARX] > xmin) & (data[VARX] <= xmax)
        msky = (data[VARY] > ymin) & (data[VARY] <= ymax)
        msk  = mskx & msky

        # Percentile
        perc = np.nan
        if np.sum(msk) > 20:  # Ensure sufficient statistics for meaningful percentile
            perc = wpercentile(data=   data.loc[msk, VAR]     .as_matrix().astype(np.float).flatten(), percents=EFF,
                               weights=data.loc[msk, 'weight'].as_matrix().astype(np.float).flatten())
            pass
        x[i,j] = (xmin + xmax) * 0.5
        y[i,j] = (ymin + ymax) * 0.5
        z[i,j] = perc

        # Set non-zero bin content
        if perc == perc:
            profile.SetBinContent(i + 1, j + 1, perc)
            pass
        pass

    # Normalise arrays
    x -= AXIS[VARX][1]
    x /= AXIS[VARX][2] - AXIS[VARX][1]
    y -= AXIS[VARY][1]
    y /= AXIS[VARY][2] - AXIS[VARY][1]

    return profile, (x,y,z)
