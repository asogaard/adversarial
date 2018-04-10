# -*- coding: utf-8 -*-

# Basic import(s)
import h5py
import numpy as np

# Project import(s)
from adversarial.utils import mkdir
from adversarial.utils import garbage_collect

# Custom import(s)
import rootplotting.rootplotting as rp


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


def test_plot (data, filename, text=[]):
    """
    ...
    """

    # Plot
    xvar = 'pt'  # truth_pt
    c = rp.canvas(batch=True)

    bins = np.linspace(0, 2500, (2500 - 0) // 50 + 1, endpoint=True)
    c.hist(data[xvar].astype(np.float16), bins=bins, weights=data['weight_test'].astype(np.float16), fillcolor=rp.colours[1], alpha=0.5)
    c.logy()
    #c.xlabel("Large-#it{R} jet p_{T}^{truth} [GeV]")
    c.xlabel("Large-#it{R} jet p_{T}^{reco} [GeV]")
    c.ylabel("Number of jets")
    c.text(text,
           qualifier="Simulation Internal")
    c.save(filename)
    return
