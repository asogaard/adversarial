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


def split_indices (num_samples, num_splits, shuffle=True, seed=None):
    """Method to (shuffle and) split indices into a fixes number of batches.

    Ensures that there is the same number of indices in each split, which will
    discard `N % splits` indices at random.

    Args:
        num_samples: Either the total number of samples, or some container
            (list or numpy array), from which to deduce the number of sampes.
        num_splits: Number of splits/batches for which to get indices
        shuffle: Whether to shuffle the indices before splitting.
        seed: The seed to use for the random number generation.

    Returns:
        A list containing the indices to be used for each split/batch.
    """

    # Check(s)
    if   type(num_samples) in [list, tuple]:
        return split_indices(num_samples[0], num_splits, shuffle, seed)
    elif type(num_samples) is np.ndarray:
        num_samples = num_samples.shape[0]
    else:
        num_samples = int(num_samples)
        pass

    assert type(num_splits) is int, "`num_splits` of type " + str(type(num_splits)) + " is not accepted"

    # Create array if indices
    indices = np.arange(num_samples)

    # Shuffle
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
        pass

    # Perform splits
    num_samples_per_split = num_samples // num_splits
    result = [split[:num_samples_per_split] for split in np.array_split(indices, num_splits)]

    return result


def apply_slice (x, idx):
    """Apply slicing to numpy arrays in nested container."""

    if isinstance(x, dict):
        return {key: apply_slice(val, idx) for (key,val) in x.iteritems()}
    elif isinstance(x, (list, tuple)):
        return [apply_slice(el,idx) for el in x]
    elif isinstance(x, np.ndarray):
        return x[idx]
    else:
        raise Exception("apply_slice: Input type '{}' not recognised".format(type(x)))

    # Should never reach here
    return None


def validate_training_input (data_train, data_validation):
    """Make sure that the format of the data dicts makes sense.

    Args:
        data_train: Dictonary containing the training data, with fields
            'input', 'target', and (optionally) 'weights'.
        data_validation: Dictonary containing the validation data, with
            fields 'input', 'target', and (optionally) 'weights'.

    Raises:
        AssertionError: If the format of the data dicts are not valid.

    Returns:
        None
    """

    # Check that only supported fields were provided
    supported_fields = ['input', 'target', 'weights', 'mask']
    unsupported_fields = list(set(data_train.keys()) - set(supported_fields))
    assert len(unsupported_fields) == 0, "Training data dict has fields which are not supported: {}".format(', '.join(unsupported_fields))
    unsupported_fields = list(set(data_validation.keys()) - set(supported_fields))
    assert len(unsupported_fields) == 0, "Validation data dict has fields which are not supported: {}".format(', '.join(unsupported_fields))

    # Check that necessary fields were provided
    assert 'input'  in data_train, "Training data dict should have 'input' field"
    assert 'target' in data_train, "Training data dict should have 'target' field"
    if data_validation:
        assert 'input'  in data_validation, "Validation data dict should have 'input' field"
        assert 'target' in data_validation, "Validation data dict should have 'target' field"
        pass

    # Check that shapes of provided arrays are compatible
    #shapes = [data_train[key].shape[0] for key in list(set(flatten(data_train.keys())) - set(['mask']))]
    #assert all(shape == shapes[0] for shape in shapes), "Training sample counts are incompatible: [{}]".format(', '.join(map(str, shapes)))
    #shapes = [data_validation[key].shape[0] for key in list(set(flatten(data_validation.keys())) - set(['mask'])) ]
    #assert all(shape == shapes[0] for shape in shapes), "Validation sample counts are incompatible: [{}]".format(', '.join(map(str, shapes)))

    # (Opt.) Apply mask
    mask = data_train.pop('mask', None)
    if mask is not None:
        data_train = apply_slice(data_train, mask)
        pass
    mask = data_validation.pop('mask', None)
    if mask is not None:
        data_validation = apply_slice(data_validation, mask)
        pass

    return data_train, data_validation


@profile
def train_in_sequence (model, data_train, data_validation={}, config={}, callbacks=[]):
    """Train a model in standard Keras fashion, using a syntax similar to train_in_parallel.

    Description...

    Args:
        model: Keras model to be trained.
        ...

    Returns:
        Dict containing the trained model and the training history.
    """

    # Check(s)
    data_train, data_validation = validate_training_input(data_train, data_validation)

    # Define variables
    use_validation = bool(data_validation)

    # Compile sequential model
    model.compile(**config['compile'])

    # Format inputs
    X = data_train['input']
    Y = data_train['target']
    W = data_train['weights'] if 'weights' in data_train else None

    validation_data = (
        data_validation['input'],
        data_validation['target'],
        data_validation['weights'] if 'weights' in data_validation else None
        ) if use_validation else None

    # Compatibility for Keras version 1, where `epochs` argument is named
    # `nb_epoch`.
    import keras
    KERAS_VERSION = int(keras.__version__.split('.')[0])
    if KERAS_VERSION == 1:
        config['fit'] = rename_key(config['fit'], 'epochs', 'nb_epoch')
        pass

    # Perform fit
    hist = model.fit(X, Y, sample_weight=W, validation_data=validation_data, callbacks=callbacks, **config['fit'])

    return {'model': model, 'history': hist.history}


@profile
def train_in_parallel (model, data_train, data_validation={}, config={}, callbacks=[], mode=None, num_devices=1, seed=None):
    """Method to use data parallelism to train a model across multiple DEVICEs.

    Description...

    Args:
        model: Keras model to be trained in parallel. Assumes single input and output layers
        ...

    Returns:
        Dict containing the trained model and the training history.

    Raises:
        Exception: If the Keras backend is not Tensorflow.
    """

    # Check(s)
    data_train, data_validation = validate_training_input(data_train, data_validation)
    assert mode in ['gpu', 'cpu'], "Requested mode '{}' not recognised".format(mode)

    # Check backend for compatibility
    import keras.backend as K
    if K.backend() != 'tensorflow':
        log.warning("train_in_parallel only works for Tensorflow. Falling back to train_in_sequence")
        train_in_sequence(model, data_train, data_validation, config=config)
        return

    # Silently fall back to `train_in_sequence` if only one device is requested.
    if num_devices == 1:
        return train_in_sequence(model, data_train, data_validation, config=config)
    #else:
    #    from keras.utils import multi_gpu_model
    #    return train_in_sequence(multi_gpu_model(model, num_devices), data_train, data_validation, config=config)
    #    pass

    # Local imports (make sure Keras backend is set before elsewhere)
    import tensorflow as tf
    import keras
    from keras.models import Model
    from keras.layers import Input
    KERAS_VERSION = int(keras.__version__.split('.')[0])

    # Define variables
    use_validation = bool(data_validation)

    # Get indices of batches of data to be used on each DEVICE.
    device_splits_train      = split_indices(data_train     ['input'], num_devices)
    device_splits_validation = split_indices(data_validation['input'], num_devices) \
                               if use_validation else None

    # Get batched data
    device_data_train      = [apply_slice(data_train,      idx) for idx in device_splits_train]
    device_data_validation = [apply_slice(data_validation, idx) for idx in device_splits_validation] if use_validation else None

    # Create parallelised model
    # -- Get number of CPUs
    try:
        cat_output = subprocess.check_output(["cat", "/proc/cpuinfo"]).split('\n')
        num_cpus  = len(filter(lambda line: line.startswith('cpu cores'),  cat_output))
    except subprocess.CalledProcessError:
        # @TODO: Implement CPU information for macOS
        num_cpus = 1
        pass

    # -- Put inputs on main CPU (PS)
    #cpu_index = int(np.random.rand() * (num_cpus + 1))
    #log.info("Putting inputs on /cpu:{}".format(cpu_index))
    inputs = list()
    for device in range(num_devices):
        with tf.device('/cpu:{}'.format(device)):
            # Loop inputs (possibly one or zero)
            device_inputs = list()
            for matrix in flatten([device_data_train[device]['input']]):
                device_inputs.append(Input(matrix.shape[1:]))
                pass
            inputs.append(device_inputs)
            pass
        pass

    # -- Create replicate classifiers on devices
    outputs = list()
    for device in range(num_devices):
        with tf.device('/{}:{}'.format(mode, device)):
            outputs.append(model(inputs[device]))
            pass
        pass

    # -- Create parallelised model
    opts = {
        'inputs'  if KERAS_VERSION >= 2 else 'input':  list(flatten(inputs)),
        'outputs' if KERAS_VERSION >= 2 else 'output': list(flatten(outputs)),
        'name': model.name + '_parallelised',
        }
    parallelised = Model(**opts)

    # Replicate fields which are specific to each output node
    for field in ['loss', 'loss_weights']:
        if field in config['compile'] and isinstance(config['compile'][field], (list, tuple)):
            config['compile'][field] = config['compile'][field] * num_devices
            pass
        pass

    # Compile parallelised model
    parallelised.compile(**config['compile'])

    # Format data
    X = list(flatten([data['input']   for data in device_data_train]))
    Y = list(flatten([data['target']  for data in device_data_train]))
    W = list(flatten([data['weights'] for data in device_data_train])) if 'weights' in data_train else None

    validation_data = (
        list(flatten([data['input']   for data in device_data_validation])),
        list(flatten([data['target']  for data in device_data_validation])),
        list(flatten([data['weights'] for data in device_data_validation])) if 'weights' in data_validation else None
        ) if use_validation else None

    # Compatibility for Keras version 1, where `epochs` argument is named
    # `nb_epoch`.
    if KERAS_VERSION == 1:
        config['fit'] = rename_key(config['fit'], 'epochs', 'nb_epoch')
        pass

    # Perform fit
    hist = parallelised.fit(X, Y, sample_weight=W, validation_data=validation_data, callbacks=callbacks, **config['fit'])

    # Divide losses by number of devices, to take average
    history = hist.history
    for name in ['loss', 'val_loss']:
        if name in history:
            history[name] = [l / float(num_devices) for l in history[name]]
            pass
        pass

    return {'model': model, 'history': history}


@profile
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

    return
