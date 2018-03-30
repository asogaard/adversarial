#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from scipy.stats import entropy

# Project imports
from adversarial.utils import mkdir


# Define global variable(s)
MASSBINS = np.linspace(40, 300, (300 - 40) // 10 + 1, endpoint=True)


def showsave (f):
    """
    Method decorrator for all study method, to (optionally) show and save the
    canvas to file.
    """
    def wrapper (*args, **kwargs):
        # Run study
        c, args, path = f(*args, **kwargs)

        # Save
        if args.save:
            dir = '/'.join(path.split('/')[:-1])
            mkdir(dir)
            c.save(path)
            pass

        # Show
        if args.show:
            c.show()
            pass
        return

    return wrapper


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
