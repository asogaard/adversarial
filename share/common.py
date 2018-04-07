# -*- coding: utf-8 -*-

# Basic import(s)
import h5py

# Project import(s)
from adversarial.utils import mkdir
from adversarial.utils import garbage_collect


@garbage_collect
def save_hdf5 (data, path, name='dataset', zip=True):
    """
    ...
    """

    # Ensure directory exists
    basedir = '/'.join(path.split('/')[:-1])
    if basedir: mkdir(basedir)

    # Save array to HDF5 file
    with h5py.File(path, 'w') as hf:
        hf.create_dataset(name,  data=data, compression="gzip" if zip else None)
        pass

    return


@garbage_collect
def load_hdf5 (path, name='dataset'):
    """
    ...
    """

    # Load array from HDF5 file
    with h5py.File(path, 'r') as hf:
        data = hf[name][:]
        pass

    return data
