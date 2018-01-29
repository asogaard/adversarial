#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common methods for training and testing flatness reweighting."""

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


@profile
def get_input (data):
    """Get input array and -weights for flatness reweighting.
    Modify data container in-place"""

    # Add necessary variable(s)
    data['logm'] = pd.Series(np.log(data['m']), index=data.index)

    # Initialise and fill coordinate original arrays
    original = data[DECORRELATION_VARIABLES].as_matrix()
    original_weight = data['weight'].as_matrix().flatten()

    # Scale coordinates to range [0,1]
    original -= np.min(original, axis=0)
    original /= np.max(original, axis=0)

    # Return
    return original, original_weight
