#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common methods for training and testing flatness reweighting."""

# Basic import(s)
from enum import Enum

# Scientific import(s)
import numpy as np
import pandas as  pd

# Project import(s)
from adversarial.profile import profile


# Common definition(s)
# @NOTE: This is the crucial point: If the target is flat in, say, (m, pt) the
# re-weighted background _won't_ be flat in (log m, log pt), and vice versa. It
# should go without saying, but draw target samples from a uniform prior on the
# coordinates which are used for the decorrelation.
DECORRELATION_VARIABLES = ['logm']

# Reweighting scenario
class Scenario (Enum):
    FLATNESS = 1
    MASS     = 2
    pass


@profile
def get_input (data, scenario):
    """Get input array and -weights for flatness- or jet mass reweighting.
    Modify data container in-place"""

    # Check(s)
    assert isinstance(scenario, Scenario), \
        "Please specify a reweighting scenario. Received {}.".format(scenario)

    if scenario == Scenario.FLATNESS:

        # Add necessary variable(s)
        data['logm'] = pd.Series(np.log(data['m']), index=data.index)

        # Initialise and fill coordinate original arrays
        original        = data[DECORRELATION_VARIABLES].as_matrix()
        original_weight = data['weight'].as_matrix().flatten()

        # Scale coordinates to range [0,1]
        original -= np.min(original, axis=0)
        original /= np.max(original, axis=0)

        # Return
        return original, original_weight

    elif scenario == Scenario.MASS:

        # Initialise and fill coordinate original arrays
        original        = data[['pt', 'm']].as_matrix()
        original_weight = data['weight'].as_matrix().flatten()

        # Mask out signal and background
        msk = (data['signal'] == 1)
        sig = original[ msk,:]
        bkg = original[~msk,:]
        sig_weight = original_weight[ msk]
        bkg_weight = original_weight[~msk]

        # Return
        return sig, bkg, sig_weight, bkg_weight

    else:
        raise "Shouldn't be here."
