#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from scipy.stats import entropy


# Define global variable(s)
MASSBINS = np.linspace(40, 300, (300 - 40) // 10 + 1, endpoint=True)


def signal_high (feat):
    """Method to determine whether the signal distribution is towards higher values."""
    return ('tau21' in feat.lower() or 'd2' in feat.lower() or 'n2' in feat.lower())


def JSD (P, Q, base=2):
    """Compute Jensen-Shannon divergence (JSD) of two distribtions.
    From: [https://stackoverflow.com/a/27432724]

    Arguments:
        P: First distribution of variable as a numpy array.
        Q: Second distribution of variable as a numpy array.
        base: Logarithmic base to use when computing KL-divergence.

    Returns:
        Jensen-Shannon divergence of `P` and `Q`.
    """
    p = P / np.sum(P)
    q = Q / np.sum(Q)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=base) + entropy(q, m, base=base))


def standardise (name):
    """Method to standardise a given filename, and remove special characters."""
    return name.lower() \
           .replace('#', '') \
           .replace(',', '__') \
           .replace('.', 'p') \
           .replace('(', '__') \
           .replace(')', '') \
           .replace('%', '') \
           .replace('=', '_')
