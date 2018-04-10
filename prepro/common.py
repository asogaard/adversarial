# -*- coding: utf-8 -*-

# Basic import(s)
import h5py
import numpy as np
import datetime
import argparse

# Project import(s)
from adversarial.utils import mkdir
from adversarial.utils import garbage_collect


def run_batched (process, args, max_processes, queue=None):
    """
    Generic method to run `process` in parallel on batches of `args`.

    Arguments:
        process: The process to batch, parallelise. Assumed to inherit from
            `multiprocessing.Process`
        args: List of arguments to be batch, each item passed to `process`.
        max_processes: Maximal number of concurrent processes to run.
    """

    # Check(s)
    assert isinstance(args, (list, tuple))
    assert len(args)

    # Batch the function `args` as to never occupy more than `max_processes`.
    batches = map(list, np.array_split(args, np.ceil(len(args) / float(max_processes))))

    # Loop batches of args
    results = list()
    for ibatch, batch in enumerate(batches):
        print "   Batch {}/{} | Contains {} arguments".format(ibatch + 1, len(batches), len(batch))

        # Convert files using multiprocessing
        processes = map(process, batch)

        # Add queue (possibly `None`)
        for p in processes:
            p.queue = queue
            pass

        # Start processes
        for p in processes: p.start()

        # (Opt.) Get results
        if queue is not None:
            results += [queue.get() for _ in processes]
            pass

        # Wait for conversion processes to finish
        for p in processes: p.join()
        pass

    return results


def get_parser (**kwargs):
    """
    General method to get argument parser for preprocessing scripts.

    Arguments:
        kwargs: Flags indicating which arguments should be used.
    """

    # Argument defaults
    default_input = '/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysis2016/FlatNtuplesR21/'
    default_dir   = '/eos/atlas/user/a/asogaard/adversarial/data/{:s}/'.format(str(datetime.date.today()))

    # List of possible arguments
    arguments = {'input': \
                     dict(action='store', type=str, default=default_input,
                          help='Input directory, from which to read input ROOT files.'),
                 'output': \
                     dict(action='store', type=str, default=default_dir,
                          help='Output directory, to which to write output files.'),
                 'dir': \
                     dict(action='store', type=str, default=default_dir,
                          help='Directory in which to read and write HDF5 files.'),
                 'max-processes': \
                     dict(action='store', type=int, default=5,
                          help='Maximum number of concurrent processes to use.'),
                 'size': \
                     dict(action='store', type=int, required=True,
                          help='Size of datasets in millions of events.')}

    # Validate
    kwargs = {k.replace('_','-'): v for (k,v) in kwargs.iteritems()}
    for k in set(kwargs) - set(arguments):
        raise IOError("get_parser: [ERROR] Keyword {} is not supported.".format(k))

    # Check(s)
    assert kwargs.get('input', False) == kwargs.get('output', False)
    assert kwargs.get('input', False) != kwargs.get('dir',    False)

    # Construct parser
    parser = argparse.ArgumentParser(description="[Generic] Perform preprocessing of ntuples to HDF5 files.")

    for k in filter(lambda k: kwargs[k], kwargs):
        parser.add_argument('--' + k, **arguments[k])
        pass

    return parser


@garbage_collect
def save_hdf5 (data, path, name='dataset', gzip=True):
    """
    Save numpy recarray to HDF5 file.

    Arguments:
        data: Numpy recarray to be saved to file.
        path: Path to HDF5 save file.
        name: Name of dataset in which to store the data.
        gzip: Whether to apply gzip compression to HDF5 file.
    """

    # Ensure directory exists
    basedir = '/'.join(path.split('/')[:-1])
    if basedir: mkdir(basedir)

    # Save array to HDF5 file
    with h5py.File(path, 'w') as hf:
        hf.create_dataset(name,  data=data, compression="gzip" if gzip else None)
        pass

    return


@garbage_collect
def load_hdf5 (path, name='dataset'):
    """
    Load numpy recarray from HDF5 file.

    Arguments:
        path: Path to HDF5 from which to read array.
        name: Name of dataset in which data is stored.
    """

    # Load array from HDF5 file
    with h5py.File(path, 'r') as hf:
        data = hf[name][:]
        pass

    return data
