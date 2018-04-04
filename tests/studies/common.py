#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_curve


# ROOT import(s)
import ROOT

# Project imports
from adversarial.utils import mkdir

# Custom import(s)
import rootplotting as rp


# Global variable definition(s)
MASSBINS = np.linspace(50, 300, (300 - 50) // 5 + 1, endpoint=True)

HISTSTYLE = {  # key = signal / passing
    True: {
        'fillcolor': rp.colours[5],
        'linecolor': rp.colours[5],
        'fillstyle': 3454,
        },
    False: {
        'fillcolor': rp.colours[1],
        'linecolor': rp.colours[1],
        'fillstyle': 3445,
        }
}

# Global ROOT TStyle settings
ROOT.gStyle.SetHatchesLineWidth(3)
ROOT.gStyle.SetTitleOffset(1.6, 'y')


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


def signal_low (feat):
    """Method to determine whether the signal distribution is towards higher values."""
    return ('Tau21' in feat or 'D2' in feat or 'N2' in feat)


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


def metrics (data, feat, target_tpr=0.5, masscut=False, verbose=True):
    """
    Compute the standard metrics (bkg. rejection and JSD) from a DataFrame.
    Assuming that any necessary selection has already been imposed.

    Arguments:
        data: Pandas dataframe, assumed to hold all necessary columns (signal,
            weight, m, and `feat`) and to have been subjected to the nessecary
            selection up-stream (training/test split, phase space restriction.)
        feat: Name of feateure for which to compute the standard metrics.
        target_tpr: Signal efficiency at which to compute the standard metrics.
        masscut: Whether to impose additional 60 GeV < m < 100 GeV cut.
        verbose: Whether to print information to stdout.

    Returns:
        Tuple of (background rejection at `target_tpr`, JSD for background mass
        distributions at `target_tpr`).
    """

    # Background rejection at `target_tpr` signal efficiency
    # ------------------------------------------------------

    # (Opt.) mass cut mask
    msk  = (data['m'] > 60.) & (data['m'] < 100.) if masscut else np.ones_like(data['signal']).astype(bool)

    # scikit-learn assumes signal towards 1, background towards 0
    pred = data[feat].values.copy()
    if signal_low(feat):
        if verbose:
            print "metrics: Reversing cut direction for {}".format(feat)
            pass
        pred *= -1.
        pass

    # Compute ROC curve efficiencies
    fpr, tpr, thresholds = roc_curve(data.loc[msk, 'signal'], pred[msk], sample_weight=data.loc[msk, 'weight'], pos_label=1)

    if masscut:
        tpr_mass = np.mean(msk[data['signal'] == 1])
        fpr_mass = np.mean(msk[data['signal'] == 0])

        tpr *= tpr_mass
        fpr *= fpr_mass
        pass

    # Get background rejection factor @ eff_sig. = 50%
    idx = np.argmin(np.abs(tpr - target_tpr))
    rej = 1. / fpr[idx]
    cut = thresholds[idx]


    # JSD at `target_tpr` signal efficiency
    # -------------------------------------

    # Get JSD(pass || fail) @ eff_sig. = 50%
    msk_pass = pred > cut
    msk_bkg = data['signal'] == 0

    p, _ = np.histogram(data.loc[ msk_pass & msk_bkg, 'm'].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk_bkg, 'weight'].values, density=True)
    f, _ = np.histogram(data.loc[~msk_pass & msk_bkg, 'm'].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk_bkg, 'weight'].values, density=True)

    jsd = JSD(p, f)

    # Return metrics
    return rej, jsd


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
