#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common, Keras-ops related utilities."""

# Scientific import(s)
import numpy as np

# Keras import(s)
import keras.backend as K


# Implement Error-function using the appropriate backend.
if K.backend() == 'tensorflow':
    import tensorflow as tf
    def erf (x):
        """Error-function from TensorFlow backend."""
        return tf.erf(x)
else:
    import theano.tensor as t
    def erf (x):
        """Error-function from Theano backend."""
        return t.erf(x)
    pass


def cumulative (x):
    """Cumulative distribution function for the unit gaussian."""
    return 0.5 * (1. + erf(x / np.sqrt(2.)))


def gaussian_integral_on_unit_interval (mean, width):
    """Compute the integral of unit gaussians on the unit interval.

    Args:
        mean: Mean(s) of unit gaussian(s).
        width: Width(s) of unit gaussian(s).

    Returns:
        Integral of unit gaussian on [0,1]
    """
    z0 = (0. - mean) / width
    z1 = (1. - mean) / width
    return cumulative(z1) - cumulative(z0)


def gaussian (x, coeff, mean, width):
    """Compute a unit gaussian using Keras-backend methods.

    Args:
        x: Variable value(s) at which to evaluate unit gaussian(s).
        coeff: Normalisation constant(s) for unit gaussian(s).
        mean: Mean(s) of unit gaussian(s).
        width: Width(s) of unit gaussian(s).

    Returns
        Function value of unit gaussian(s) evaluated at `x`.
    """
    return coeff * K.exp( - K.square(x - mean) / 2. / K.square(width)) / K.sqrt( 2. * K.square(width) * np.pi)
