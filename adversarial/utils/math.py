#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common, math-related utilities."""

# Basic import(s)
import numpy as np
import pandas as pd

# Project import(s)
from .misc import belongs_to


def wmean (x, w):
    """Weighted mean with flexible backend
    From [https://stackoverflow.com/a/38647581]
    """

    # Numpy array-type
    if belongs_to(x, np):
        return np.sum(x * w) / np.sum(w)

    # Pandas DataFrame-type
    if belongs_to(x, pd):
        return wmean(x.values, w)

    # Tensorflow type
    else:
        import tensorflow as tf
        if belongs_to(x, tf):
            import keras.backend as K
            assert K.backend() == 'tensorflow', \
                "The method `correlation_coefficient` is only defined for TensorFlow backend."
            return K.sum(tf.multiply(x, w)) / K.sum(w)
        pass

    # Fallback
    raise ValueError("Input with type {} from module {} not recognised.".format(type(x), type(x).__module__))


def wcov (x, y, w):
    """Weighted covariance with flexible backend
    From [https://stackoverflow.com/a/38647581]
    """

    # Intermediary results
    xm = x - wmean(x, w)
    ym = y - wmean(y, w)

    # Numpy array-type
    if belongs_to(x, np):
        return np.sum(w * xm * ym) / np.sum(w)

    # Tensorflow type
    else:
        import tensorflow as tf
        if belongs_to(x, tf):
            return K.sum(tf.multiply(tf.multiply(w, xm), ym)) / K.sum(w)
        pass

    # Fallback
    raise ValueError("Input with type {} from module {} not recognised.".format(type(x), type(x).__module__))


def wcorr (x, y, w):
    """Weighted linear correlation with flexible backend
    From [https://stackoverflow.com/a/38647581]
    """

    # Numpy array-type
    if belongs_to(x, np):
        backend = np

    # Assuming Keras backend-type
    else:
        import keras.backend as K
        backend = K
        pass
    return wcov(x, y, w) / backend.sqrt(wcov(x, x, w) * wcov(y, y, w))

    # Fallback
    raise ValueError("Input with type {} from module {} not recognised.".format(type(x), type(x).__module__))


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
