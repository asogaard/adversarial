#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common, management-related utilities."""

# Basic import(s)
import os
import gc
import json
import gzip
import pickle
import subprocess

def garbage_collect (f):
    """
    Function decorator to manually perform garbage collection after the call,
    so as to avoid unecessarily large memory consumption.
    """
    def wrapper(*args, **kwargs):
        ret = f(*args, **kwargs)
        gc.collect()
        return ret
    return wrapper


def loadclf (path, zip=True):
    """
    Load pickled classifier from file.
    """

    # Check file suffix
    if path.endswith('.gz'):
        zip = True
        pass

    # Determine operation
    op = gzip.open if zip else open

    # Load model
    with op(path, 'r') as f:
        clf = pickle.load(f)
        pass

    return clf


def saveclf (clf, path, zip=True):
    """
    Save pickled classifier to file.
    """

    # Ensure model directory exists
    basedir = '/'.join(path.split('/')[:-1])
    if basedir:
        mkdir(basedir)
        pass

    # Ensure correct suffix
    if zip and not path.endswith('.gz'):
        path += '.gz'
    elif not zip and path.endswith('.gz'):
        zip = True
        pass

    # Determine operation
    op = gzip.open if zip else open

    # Save classifier
    with op(path, 'w') as f:
        pickle.dump(clf, f)
        pass

    return


def mkdir (path):
    """Script to ensure that the directory at `path` exists.

    Arguments:
        path: String specifying path to directory to be created.
    """

    # Check mether  output directory exists
    if not os.path.exists(path):
        print "mkdir: Creating output directory:\n  {}".format(path)
        try:
            os.makedirs(path)
        except OSError:
            # Apparently, `path` already exists.
            pass
        pass
    return


def kill (pid, name=None):
    """Query user to kill system process `pid`.

    Arguments:
        pid: ID of process to be killed assumed to be owned by user.
        name: Name of process.
    """

    # Get username
    username = subprocess.check_output(['whoami']).strip()

    # Get processes belonging to user
    lines = subprocess.check_output(["ps", "-u", username]).split('\n')[1:-1]

    # Get PIDs of processes belonging to user
    pids = [int(line.strip().split()[0]) for line in lines]

    # Check PID is suitable for deletion
    if int(pid) not in pids:
        print "[WARN]  No process with PID {} belonging to user {} was found running. Exiting.".format(pid, username)
        return

    # Query user
    print "[INFO]  {1} process ({0}) is running in background. Enter `q` to close it. Enter anything else to quit the program while leaving {1} running.".format(pid, name)
    response = raw_input(">> ")
    if response.strip() == 'q':
        subprocess.call(["kill", str(pid)])
    else:
        print "[INFO]  {} process is left running. To manually kill it later, do:".format(name)
        print "[INFO]    $ kill {}".format(pid)
        pass
    return


def save (basedir, name, model, history=None):
    """Standardised method to save Keras models to file.

    Arguments:
        basedir: Directory in which models should be saved. Is created if it
            doesn't already exist.
        name: Name of model to be saved, used in filenames.
        model: Keras model to be saved.
        history: Container with logged training history.
    """

    # Make sure output directory exists
    mkdir(basedir)

    # Save full model and model weights
    model.save        (basedir + '{}.h5'        .format(name))
    model.save_weights(basedir + '{}_weights.h5'.format(name))

    # Save training history
    if history is not None:
        with open(basedir + 'history__{}.json'.format(name), 'wb') as f:
            json.dump(history, f)
            pass
        pass
    return


def load (basedir, name, model=None):
    """Standardised method to load Keras models from file.
    If a pre-existing model is specified only weights are loaded into the model.

    Arguments:
        basedir: Directory from which models should be loaded.
        name: Name of model to be loaded, used in filenames.
        model: Pre-existing model.

    Returns:
        model: Keras model to be saved.
        history: Container with logged training history.

    Raises:
        IOError: If any of the attempted files do not exist.
    """

    # Import(s)
    from keras.models import load_model

    # Load full pre-trained model or model weights
    if model is None:
        model = load_model(basedir + '{}.h5'.format(name))
    else:
        model.load_weights(basedir + '{}_weights.h5'.format(name))
        pass

    # Load associated training histories
    try:
        history_file = basedir + 'history__{}.json'.format(name)
        with open(history_file, 'r') as f:
            history = json.load(f)
            pass
    except:
        print "[WARN] Could not find history file {}."
        history = None
        pass

    return model, history


def lwtnn_save(model, name, basedir='models/adversarial/lwtnn/'):
    """Method for saving classifier in lwtnn-friendly format.
    See [https://github.com/lwtnn/lwtnn/wiki/Keras-Converter]
    """
    # Check(s).
    if not basedir.endswith('/'):
        basedir += '/'
        pass

    # Make sure output directory exists
    mkdir(basedir)

    # Get the architecture as a json string
    arch = model.to_json()

    # Save the architecture string to a file
    with open(basedir + name + '_architecture.json', 'w') as arch_file:
        arch_file.write(arch)
        pass

    # Now save the weights as an HDF5 file
    model.save_weights(basedir + name + '_weights.h5')

    # Save full model to HDF5 file
    model.save(basedir + name + '.h5')
    return
