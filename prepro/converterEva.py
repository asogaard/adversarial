#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Basic import(s)
import h5py
import itertools
from glob import glob

# Scientific import(s)
import numpy as np
import numpy.lib.recfunctions as rfn
import ROOT
import root_numpy

# Project import(s)
from adversarial.profile import profile

# Global variable definition(s)
COLLECTION = 'AntiKt10LCTopoTrimmedPtFrac5SmallR20JetsCalibSelect'


# Utility function(s)
glob_sort_list = lambda paths: sorted(list(itertools.chain.from_iterable(map(glob, paths))))

def rename (name):
    name = name.replace(COLLECTION, 'fjet')
    return name


def unravel (data):
    """
    ...
    """
    
    if not data.dtype.hasobject:
        return data

    # Identify variable-length (i.e. per-jet) and scalar (i.e. per-event)
    # fields
    jet_fields = list()
    for field, (kind, _) in data.dtype.fields.iteritems():
        if kind.hasobject:
            jet_fields.append(field)
            pass
        pass
    jet_fields   = sorted(jet_fields)
    event_fields = sorted([field for field in data.dtype.names if field not in jet_fields])

    # Loop events, take up to `nleading` jets from each (@TODO: pT-ordered?)
    jets = list()
    data_events = data[event_fields]
    data_jets   = data[jet_fields]
    
    rows = list()
    for jets, event in zip(data_jets, data_events):
        for jet in np.array(jets.tolist()).T:
            row = event.copy()
            row = rfn.append_fields(row, jet_fields, jet.tolist(), usemask=False)
            rows.append(row)
            pass
        pass
        
    return np.concatenate(rows)


# Main function definition.
@profile
def main (sig, bkg, treename='jetTree/nominal', shuffle=True, sample=None, seed=21, replace=True, nleading=2, frac_train=0.8):
    """
    ...
    """

    # For reproducibility
    rng = np.random.RandomState(seed=seed)

    # Check(s)
    if isinstance(sig, str):
        sig = [sig]
        pass
    if isinstance(bkg, str):
        bkg = [bkg]
        pass

    # Get glob'ed list of files for each category
    sig = glob_sort_list(sig)
    bkg = glob_sort_list(bkg)

    # Read in data
    branches = None
    selection = 'n{} > 0'.format(COLLECTION)

    kwargs = dict(treename=treename, branches=branches, selection=selection)

    data_sig = root_numpy.root2array(sig, **kwargs)
    data_bkg = root_numpy.root2array(bkg, **kwargs)

    # (Opt.) Unravel non-flat data
    data_sig = unravel(data_sig)
    data_bkg = unravel(data_bkg)

    # Append signal fields
    data_sig = rfn.append_fields(data_sig, "signal", np.ones ((data_sig.shape[0],)), usemask=False)
    data_bkg = rfn.append_fields(data_bkg, "signal", np.zeros((data_bkg.shape[0],)), usemask=False)

    # Concatenate arrays
    data = np.concatenate((data_sig, data_bkg))

    # Rename columns
    data.dtype.names = map(rename, data.dtype.names)

    # Object selection
    msk = (data['fjet_pt'] > 10.) & (data['fjet_JetConstitScaleMomentum_m'] > 10.)
    data = data[msk]

    # Append rhoDDT field
    data = rfn.append_fields(data, "rhoDDT", np.log(np.square(data['fjet_JetConstitScaleMomentum_m']) / data['fjet_pt']), usemask=False)

    # Append train field
    data = rfn.append_fields(data, "train", rng.rand(data.shape[0]) < frac_train, usemask=False)

    # (Opt.) Shuffle
    if shuffle:
        rng.shuffle(data)
        pass

    # (Opt.) Subsample
    if sample:
        data = rng.choice(data, sample, replace=replace)
        pass

    return data


# Main function call.
if __name__ == '__main__':
    basepath = '/afs/cern.ch/work/e/ehansen/public/DDT-studies-FourJets/'
    sig = basepath + 'zprime/*zprime*.root'
    bkg = basepath + 'jetjet/*.root'
    data = main(sig, bkg)
    with h5py.File('mytest-susy-fourjet.h5', 'w') as hf:
        hf.create_dataset('dataset',  data=data, compression='gzip')
        pass

    pass
