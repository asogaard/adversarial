#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utilities for training and evaluating adversarial neural networks for de-correlated jet tagging."""

# Basic import(s)
import os
import re
import logging as log
import subprocess
import collections

# Scientific import(s)
import numpy as np

# Project import(s)
from .profile import Profile, profile

# Global variable(s)
_gpu_utilisation = None

def gpu_utilisation ():
    """Return dictionary of available GPUs and current utilisation.

    Assumes NVIDIA GPUs."""

    global _gpu_utilisation

    # Only query if not previously set
    if _gpu_utilisation is None:

        # Initialise output variable
        result = dict()

        try:
            # Try calling `nvidia-smi`
            ret = subprocess.check_output(["nvidia-smi"]).split('\n')

            # Get GPU names
            indices = np.where(['+' in l for l in ret])[0][3:-2] - 2  # Assuming regular structure of `nvidia-smi` output
            names = map(lambda s: int(s.split("|")[1].split()[0]), [ret[i] for i in indices])

            # Get GPU utilisations as integer in [0,100]
            utilisations = map(lambda s: int(s.split('|')[-2].split()[0][:-1]), filter(lambda s: '%' in s, ret))

            # Create dictionary
            result = dict(zip(names, utilisations))

        except OSError:
            # `nvidia-smi` command, and thus GPUs, not available
            pass

        # Set global value
        _gpu_utilisation = result
        pass

    return _gpu_utilisation


def rename_key (d, old, new):
    """Rename key in dict, if it exists."""
    if old in d:
        d[new] = d.pop(old)
        pass
    return d

'''
def snake_case (string):
    """ ... """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def latex (_name, ROOT=True):
    """..."""

    name = _name.lower()

    name = re.sub('^d2', 'D_{2}', name)
    name = re.sub('^pt$', 'p_{T}', name)
    #name = re.sub('rho', '\\rho', name)
    name = name.replace('rho', '\\rho')
    name = name.replace('tau21', '\\tau_{21}')
    name = name.replace('ddt', '^{DDT}')
    name = re.sub('\_([0-9]+)$', '^{(\\1)}', name)
    name = re.sub('-knn(.*)$', '^{kNN\\1}', name)

    # ML taggers
    if 'boost' in name or re.search('nn$', name) or re.search('^nn', name) or 'ann' in name:
        name = '\\textit{z}_{%s}' % _name
        pass

    name = re.sub('(\(.*\))([}]*)$', '\\2^{\\1}', name)

    # Remove duplicate superscripts
    name = re.sub("(\^.*)}\^{", "\\1", name)

    if name == _name.lower():
        name = _name
        pass

    if ROOT:
        return name.replace('\\', '#').replace('textit', 'it')
        pass
    return r"${}$".format(name)
'''

'''
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
'''

def flatten (container):
    """Unravel nested lists and tuples.

    From [https://stackoverflow.com/a/10824420]
    """
    if isinstance(container, (list,tuple)):
        for i in container:
            if isinstance(i, (list,tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i
            pass
    else:
        yield container


def apply_patch (d, u):
    """Update nested dictionary without overwriting previous levels.

    From [https://stackoverflow.com/a/3233356]
    """
    for k, v in u.iteritems():
        if isinstance(v, collections.Mapping):
            r = apply_patch(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
            pass
        pass
    return d


'''
# @TODO: Discontinue in favour of sklearn
def roc_efficiencies (sig, bkg, sig_weight=None, bkg_weight=None):
    """Compute the signal and background efficiencies for successive cuts.

    Adapted from [https://github.com/asogaard/AdversarialSubstructure/blob/master/utils.py]
    """

    # Check(s)
    if sig_weight is None:
        sig_weight = np.ones_like(sig)
        pass

    if bkg_weight is None:
        bkg_weight = np.ones_like(bkg)
        pass

    # Store and sort 2D array
    sig2 = np.vstack((sig.ravel(), sig_weight.ravel(), np.zeros_like(sig.ravel()))).T
    bkg2 = np.vstack((bkg.ravel(), np.zeros_like(bkg.ravel()), bkg_weight.ravel())).T
    sig_bkg      = np.vstack((sig2, bkg2))
    sig_bkg_sort = sig_bkg[sig_bkg[:,0].argsort()]

    # Accumulated (weighted) counts
    eff_sig = np.cumsum(sig_bkg_sort[:,1]) / np.sum(sig_weight)
    eff_bkg = np.cumsum(sig_bkg_sort[:,2]) / np.sum(bkg_weight)

    # Make sure that cut direction is correct
    if np.sum(eff_sig < eff_bkg) > len(eff_sig) / 2:
        eff_sig = 1. - eff_sig
        eff_bkg = 1. - eff_bkg
        pass

    return eff_sig, eff_bkg
'''

'''
# @TODO: Discontinue in favour of sklearn
def roc_auc (eff_sig, eff_bkg):
    """Compute the ROC area-under-the-curve for provided signal and background efficiencies."""

    # Check(s)
    assert len(eff_sig) == len(eff_bkg), "Number of signal ({}) and background ({}) efficiencies do not agree".format(len(eff_sig), len(eff_bkg))

    # Ensure efficiencies are increasing
    if eff_sig[0] > eff_sig[-1]:
        eff_sig = eff_sig[::-1]
        eff_bkg = eff_bkg[::-1]
        pass

    # Compute AUC as the average signal efficiency times the difference in
    # background efficiencies, summed of all ROC segments.
    auc = np.sum((eff_sig[:-1] + 0.5 * np.diff(eff_sig)) * np.diff(eff_bkg))

    return auc
'''

'''@profile
def initialise_backend (args):
    """Initialise the Keras backend.

    Args:
        args: Namespace containing command-line arguments from argparse. These
            settings specify which back-end should be configured, and how.
    """

    # Check(s)
    if args.gpu and args.theano and args.devices > 1:
        raise NotImplementedError("Distributed training on GPUs is current not enabled.")

    # Specify Keras backend and import module
    os.environ['KERAS_BACKEND'] = "theano" if args.theano else "tensorflow"

    # Get number of cores on CPU(s), name of CPU devices, and number of physical
    # cores in each device.
    try:
        cat_output = subprocess.check_output(["cat", "/proc/cpuinfo"]).split('\n')
        num_cpus  = len(filter(lambda line: line.startswith('cpu cores'),  cat_output))
        name_cpu  =     filter(lambda line: line.startswith('model name'), cat_output)[0] \
                        .split(':')[-1].strip()
        num_cores = int(filter(lambda line: line.startswith('cpu cores'),  cat_output)[0] \
                        .split(':')[-1].strip())
        log.info("Found {} {} devices with {} cores each.".format(num_cpus, name_cpu, num_cores))
    except subprocess.CalledProcessError:
        # @TODO: Implement CPU information for macOS
        num_cores = 1
        log.warning("Could not retrieve CPU information -- probably running on macOS. Therefore, multi-core running is disabled.")
        pass

    # Configure backends
    if args.theano:


        if args.devices > 1:
            log.warning("Currently it is not possible to specify more than one devices for Theano backend.")
            pass

        if not args.gpu:
            # Set number of OpenMP threads to use; even if 1, set to force
            # use of OpenMP which doesn't happen otherwise, for some
            # reason. Gives speed-up of factor of ca. 6-7. (60 sec./epoch ->
            # 9 sec./epoch)
            os.environ['OMP_NUM_THREADS'] = str(num_cores * 2)
            pass

        # Switch: CPU/GPU
        cuda_version = '8.0.61'
        standard_flags = [
            'device={}'.format('cuda' if args.gpu else 'cpu'),
            'openmp=True',
            ]
        dnn_flags = [
            'dnn.enabled=True',
            'dnn.include_path=/exports/applications/apps/SL7/cuda/{}/include/'.format(cuda_version),
            'dnn.library_path=/exports/applications/apps/SL7/cuda/{}/lib64/'  .format(cuda_version),
            ]
        os.environ["THEANO_FLAGS"] = ','.join(standard_flags + (dnn_flags if args.gpu else []))

    else:

        # Set print level to avoid unecessary warnings, e.g.
        #  $ The TensorFlow library wasn't compiled to use <SSE4.1, ...>
        #  $ instructions, but these are available on your machine and could
        #  $ speed up CPU computations.
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        # Switch: CPU/GPU
        """ Cf. [https://www.wiki.ed.ac.uk/display/ResearchServices/TensorFlow#TensorFlow-CUDA_VISIBLE_DEVICES]
        if args.gpu:
            # Set this environment variable to "0,1,...", to make Tensorflow
            # use the first N available GPUs
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, dict(filter(lambda t: t[1] < 80 and t[0] < args.devices, gpu_utilisation().iteritems())).keys())) #','.join(map(str,range(args.devices)))

        else:
            # Setting this enviorment variable to "" makes all GPUs invisible to
            # tensorflow, thus forcing it to run on CPU (on as many cores as
            # possible), cf. [https://stackoverflow.com/a/42750563]
            os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
            os.environ['CUDA_VISIBLE_DEVICES'] = ""
            pass
        #"""

        # Load the tensorflow module here to make sure only the correct
        # GPU devices are set up
        import tensorflow as tf

        # @TODO:
        # - Some way to avoid starving GPU of data?
        # - Remove bloat by using `multi_gpu_model`?

        # Manually configure Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1,
                                    allow_growth=True)

        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores * 2,
                                inter_op_parallelism_threads=num_cores * 2,
                                allow_soft_placement=True,
                                device_count={'GPU': args.devices if args.gpu else 0},
                                gpu_options=gpu_options if args.gpu else None,
                                )
                                #run_metadata=tf.RunMetadata())  # @TEMP

        session = tf.Session(config=config)
        pass

    # Import Keras backend
    import keras.backend as K
    K.set_floatx('float32')

    if not args.theano:
        # Set global Tensorflow session
        K.set_session(session)
        pass

    return'''
