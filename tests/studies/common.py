#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_curve

# ROOT import(s)
import ROOT

# Project imports
from adversarial.utils import mkdir, garbage_collect

# Custom import(s)
import rootplotting as rp
rp.colours.pop(3)  # To remove unused colour from list

# Global variable definition(s)
MASSBINS = np.linspace(50, 300, (300 - 50) // 5 + 1, endpoint=True)

HISTSTYLE = {  # key = signal / passing
    True: {
        'fillcolor': rp.colours[4],
        'linecolor': rp.colours[4],
        'fillstyle': 3354,
        },
    False: {
        'fillcolor': rp.colours[1],
        'linecolor': rp.colours[1],
        'fillstyle': 1001,
        'alpha': 0.5,
        }
    }

TEXT = ["#sqrt{s} = 13 TeV",
        #"Baseline selection",
        ]

# Global ROOT TStyle settings
ROOT.gStyle.SetHatchesLineWidth(3)
ROOT.gStyle.SetTitleOffset(1.6, 'y')


class TemporaryStyle ():
    """
    Context manager to temporarily modify global ROOT style settings.
    """

    def __init__ (self):
        self.rstyle = ROOT.gROOT.GetStyle(map(lambda ts: ts.GetName(), ROOT.gROOT.GetListOfStyles())[-1])
        return

    def __enter__ (self):
        style = ROOT.TStyle(self.rstyle)
        style.cd()
        return style

    def __exit__ (self, *args):
        self.rstyle.cd()
        return
    pass


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


@garbage_collect
def metrics (data, feat, target_tpr=0.5, masscut=False, verbose=False):
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
    if masscut:
        print "metrics: Applying mass cut."
        pass
    msk = (data['m'] > 60.) & (data['m'] < 100.) if masscut else np.ones_like(data['signal']).astype(bool)

    # scikit-learn assumes signal towards 1, background towards 0
    pred = data[feat].values.copy()
    if signal_low(feat):
        if verbose:
            print "metrics: Reversing cut direction for {}".format(feat)
            pass
        pred *= -1.
        pass

    # Compute ROC curve efficiencies
    fpr, tpr, thresholds = roc_curve(data.loc[msk, 'signal'], pred[msk], sample_weight=data.loc[msk, 'weight_test'])

    if masscut:
        tpr_mass = np.mean(msk[data['signal'] == 1])
        fpr_mass = np.mean(msk[data['signal'] == 0])

        tpr *= tpr_mass
        fpr *= fpr_mass
        pass

    # Get background rejection factor
    idx = np.argmin(np.abs(tpr - target_tpr))
    rej = 1. / fpr[idx]
    cut = thresholds[idx]


    # JSD at `target_tpr` signal efficiency
    # -------------------------------------

    # Get JSD(1/Npass dNpass/dm || 1/Nfail dNfail/dm)
    msk_pass = pred > cut
    msk_bkg  = data['signal'] == 0

    p, _ = np.histogram(data.loc[ msk_pass & msk_bkg, 'm'].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk_bkg, 'weight_test'].values, density=True)
    f, _ = np.histogram(data.loc[~msk_pass & msk_bkg, 'm'].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk_bkg, 'weight_test'].values, density=True)

    jsd = JSD(p, f)

    # Return metrics
    return rej, 1./jsd


@garbage_collect
def bootstrap_metrics (data, feat, num_bootstrap=10, **kwargs):
    """
    ...
    """
    # Compute metrics using bootstrapping
    bootstrap_rej, bootstrap_jsd = list(), list()
    for _ in range(num_bootstrap):
        #data_bootstrap = data.sample(frac=1.0, replace=True)
        #rej, jsd = metrics(data_bootstrap, feat, **kwargs)
        idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
        rej, jsd = metrics(data.iloc[idx], feat, **kwargs)
        bootstrap_rej.append(rej)
        bootstrap_jsd.append(jsd)
        pass

    return (np.mean(bootstrap_rej), np.std(bootstrap_rej)), (np.mean(bootstrap_jsd), np.std(bootstrap_jsd))


def standardise (name):
    """Method to standardise a given filename, and remove special characters."""
    return name.lower() \
           .replace('#', '') \
           .replace('minus', '') \
           .replace(',', '__') \
           .replace('.', 'p') \
           .replace('(', '__') \
           .replace(')', '') \
           .replace('%', '') \
           .replace('=', '_')
