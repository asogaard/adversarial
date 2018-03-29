#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common, math-related utilities."""

# Basic import(s)
import numpy as np


def wmean (x, w):
    """Weighted Mean
    From [https://stackoverflow.com/a/38647581]
    """
    return np.sum(x * w) / np.sum(w)


def wcov (x, y, w):
    """Weighted Covariance
    From [https://stackoverflow.com/a/38647581]
    """
    return np.sum(w * (x - wmean(x, w)) * (y - wmean(y, w))) / np.sum(w)


def wcorr (x, y, w):
    """Weighted Correlation
    From [https://stackoverflow.com/a/38647581]
    """
    return wcov(x, y, w) / np.sqrt(wcov(x, x, w) * wcov(y, y, w))


def wpercentile (data, percents, weights=None):
    """ percents in units of 1%
    weights specifies the frequency (count) of data.
    From [https://stackoverflow.com/a/31539746]
    """
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 100. * w.cumsum() / w.sum()
    y = np.interp(percents, p, d)
    return y
