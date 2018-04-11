# -*- coding: utf-8 -*-

"""Common methods for training and testing BDT classifiers."""

# Scientific import(s)
import numpy as np
import pandas as pd
from sklearn.externals.joblib.parallel import cpu_count, Parallel, delayed

# Project import(s)
from adversarial.utils import loadclf
from adversarial.profile import profile


def _predict(estimator, X, method, start, stop):
    """
    Indirect method caller, for sklearn classifier predict-type calls.
    """
    return getattr(estimator, method)(X[start:stop])


def parallel_predict(estimator, X, n_jobs=1, method='predict', batches_per_job=3):
    """
    Run sklearn classifier prediction in parallel.
    """
    n_jobs = max(cpu_count() + 1 + n_jobs, 1)  # XXX: this should really be done by joblib
    n_batches = batches_per_job * n_jobs
    n_samples = len(X)
    batch_size = int(np.ceil(n_samples / n_batches))
    parallel = Parallel(n_jobs=n_jobs, backend="threading")
    results = parallel(delayed(_predict, check_pickle=False)(estimator, X, method, i, i + batch_size) for i in range(0, n_samples, batch_size))
    return np.concatenate(results)


@profile
def add_bdt (data, var=None, path=None):
    """
    Add BDT-based classifier `feat` to `data`. Modifies `data` in-place.

    Arguments:
        data: Pandas DataFrame to which to add the DDT-transformed variable.
        var: Name of output feature.
        path: Path to trained BDT classifier.
    """

    # Check(s)
    assert var  is not None, "add_bdt: Please specify an output variable name."
    assert path is not None, "add_bdt: Please specify a model path."

    # Load model
    clf = loadclf(path)

    # Use parallelisation to speed-up prediction
    result = parallel_predict(clf, data, n_jobs=16, method='predict_proba')

    # Add new classifier to data array
    data[var] = pd.Series(result[:,1].flatten(), index=data.index)
    return
