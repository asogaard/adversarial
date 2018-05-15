 # -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_curve

# Project imports
from adversarial.utils import garbage_collect

# Global variable definition(s)
MASSBINS = np.linspace(50, 300, (300 - 50) // 5 + 1, endpoint=True)


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
def metrics (data, feat, target_tpr=0.5, cut=None, masscut=False, verbose=False):
    """
    Compute the standard metrics (bkg. rejection and JSD) from a DataFrame.
    Assuming that any necessary selection has already been imposed.

    Arguments:
        data: Pandas dataframe, assumed to hold all necessary columns (signal,
            weight_test, m, and `feat`) and to have been subjected to the
            nessecary selection up-stream (training/test split, phase space
            restriction.)
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
    if cut is None:
        idx = np.argmin(np.abs(tpr - target_tpr))
        cut = thresholds[idx]    
    else:
        print "metrics: Using manual cut of {:.2f} for {}".format(cut, feat)
        idx = np.argmin(np.abs(thresholds - cut))
        print "metrics:   effsig = {:.1f}%, effbkg = {:.1f}, threshold = {:.2f}".format(tpr[idx] * 100.,
                                                                                        fpr[idx] * 100.,
                                                                                        thresholds[idx])
        pass

    eff = tpr[idx]
    rej = 1. / fpr[idx]


    # JSD at `target_tpr` signal efficiency
    # -------------------------------------

    # Get JSD(1/Npass dNpass/dm || 1/Nfail dNfail/dm)
    msk_pass = pred > cut
    msk_bkg  = data['signal'] == 0

    p, _ = np.histogram(data.loc[ msk_pass & msk_bkg, 'm'].values, bins=MASSBINS, weights=data.loc[ msk_pass & msk_bkg, 'weight_test'].values, density=1.)
    f, _ = np.histogram(data.loc[~msk_pass & msk_bkg, 'm'].values, bins=MASSBINS, weights=data.loc[~msk_pass & msk_bkg, 'weight_test'].values, density=1.)

    jsd = JSD(p, f)

    # Return metrics
    return eff, rej, 1./jsd


@garbage_collect
def bootstrap_metrics (data, feat, num_bootstrap=10, **kwargs):
    """
    ...
    """
    # Compute metrics using bootstrapping
    bootstrap_eff, bootstrap_rej, bootstrap_jsd = list(), list(), list()
    for _ in range(num_bootstrap):
        idx = np.random.choice(data.shape[0], data.shape[0], replace=True)
        eff, rej, jsd = metrics(data.iloc[idx], feat, **kwargs)
        bootstrap_rej.append(rej)
        bootstrap_jsd.append(jsd)
        pass

    return (np.mean(bootstrap_eff), np.std(bootstrap_eff)), \
           (np.mean(bootstrap_rej), np.std(bootstrap_rej)), \
           (np.mean(bootstrap_jsd), np.std(bootstrap_jsd))
