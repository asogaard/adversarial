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
JET_FIELDS = [
    '{}_E',
    '{}_pt',
    '{}_phi',
    '{}_eta',
    '{}_JetConstitScaleMomentum_eta',
    '{}_JetConstitScaleMomentum_phi',
    '{}_JetConstitScaleMomentum_m',
    '{}_JetConstitScaleMomentum_pt',
    '{}_JetEMScaleMomentum_eta',
    '{}_JetEMScaleMomentum_phi',
    '{}_JetEMScaleMomentum_m',
    '{}_JetEMScaleMomentum_pt',
    '{}_GhostArea',
    '{}_ActiveArea',
    '{}_VoronoiArea',
    '{}_ActiveArea4vec_pt',
    '{}_ActiveArea4vec_eta',
    '{}_ActiveArea4vec_phi',
    '{}_ActiveArea4vec_m',
    '{}_Split12',
    '{}_Split23',
    '{}_Split34',
    '{}_tau1_wta',
    '{}_tau2_wta',
    '{}_tau3_wta',
    '{}_tau21_wta',
    '{}_tau32_wta',
    '{}_ECF1',
    '{}_ECF2',
    '{}_ECF3',
    '{}_C2',
    '{}_D2',
    '{}_NTrimSubjets',
    '{}_Nclusters',
    '{}_nTracks',
]
JET_FIELDS = [field.format(COLLECTION) for field in JET_FIELDS]


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
def main (sig, bkg, treename='jetTree/nominal', shuffle=True, sample=None, seed=21, replace=True, nleading=2):
    """
    ...
    """

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
    object_selection = {'{0}_JetConstitScaleMomentum_m > 10.0 && {0}_pt > 10.'.format(COLLECTION): JET_FIELDS}

    kwargs = dict(treename=treename, 
                  branches=branches, 
                  selection=selection, 
                  object_selection=object_selection)

    data_sig = root_numpy.root2array(sig, **kwargs)
    data_bkg = root_numpy.root2array(bkg, **kwargs)

    # (Opt.) Unravel non-flat data
    data_sig = unravel(data_sig)
    data_bkg = unravel(data_bkg)

    # Append target fields
    data_sig = rfn.append_fields(data_sig, "target", np.ones ((data_sig.shape[0],)), usemask=False)
    data_bkg = rfn.append_fields(data_bkg, "target", np.zeros((data_bkg.shape[0],)), usemask=False)

    # Concatenate arrays
    data = np.concatenate((data_sig, data_bkg))

    # Rename columns
    data.dtype.names = map(rename, data.dtype.names)

    # Append rhoDDT fields
    data = rfn.append_fields(data, "rhoDDT", np.log(np.square(data['fjet_JetConstitScaleMomentum_m']) / data['fjet_pt']), usemask=False)

    # (Opt.) Shuffle
    rng = np.random.RandomState(seed=seed)
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
