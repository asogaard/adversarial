#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# ROOT import(s)
import ROOT
import root_numpy

# Global definition(s)
SELECTION = {
    'common': ("(fjet_truthJet_pt >  200E+03 && "
               " fjet_truthJet_pt < 2000E+03 && "
               " fjet_truthJet_eta > -2.0 && "
               " fjet_truthJet_eta <  2.0 && "
               " fjet_m >  40E+03 && "
               " fjet_m < 300E+03 &&"
               " fjet_numConstituents > 2)"),

    'sig': ("(W && "
            " fjet_dRmatched_WZChild1_dR < 0.75 && "
            " fjet_dRmatched_WZChild2_dR < 0.75 && "
            " fjet_truth_dRmatched_particle_dR < 0.75)"),

    'bkg': ("(!W && "
            " !top)")
    }

# -- Add `common` selection to all other keys
common_selection = SELECTION.pop('common')
for key in SELECTION.keys():
    SELECTION[key] += " && " + common_selection
    pass

# -- List of branches to read
branches = ['fjet_testing_weight_pt', 'fjet_pt', 'fjet_m', 'fjet_D2', 'fjet_Tau21_wta', 'fjet_truthJet_pt', 'W']

def rename (name):
    """
    Rename branches.
    """

    if name == 'W':
        return 'signal'
    elif name == 'fjet_testing_weight_pt':
        return 'weight'
    elif name == 'fjet_Tau21_wta':
        return 'Tau21'
    elif name == 'fjet_truthJet_pt':
        return 'truth_pt'
    elif 'fjet' in name:
        return name.replace('fjet_', '')
    raise Exception("Unkown branch {}".format(name))


# Main function definition
def main ():

    # Define variable(s)
    basedir = '/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysis2016/FlatNTuples/20170530/MLTrainingTesting/Testing/'

    paths = {
        'sig': basedir + 'Wtagging/testing_Wprime.root',
        'bkg': basedir + 'background/testing_dijet.root'
        }

    # Read data for each class
    data = dict()
    for cls in ['sig', 'bkg']:
  
        # Get ROOT TTree
        f = ROOT.TFile(paths[cls], 'READ')
        t = f.Get("FlatSubstructureJetTree")
 
        #### print "\nBranches found for class {:s}:".format(cls)
        #### for name in map(lambda br: br.GetName(), t.GetListOfBranches()):
        ####     print "  {:s}".format(name)
       
        # Read data from TTree
        data[cls] = root_numpy.tree2array(t, 
                                          branches=branches, 
                                          selection=SELECTION[cls], 
                                          #stop=2*int(3E+05 if cls == 'sig' else
                                          #1E+06))
                                          )

        # Take random samples with same statistics as reported in note
        got  = data[cls].size
        take = int(3E+05 if cls == 'sig' else 1E+06)
        print "Got {}, take {}".format(got, take)
        idx = np.random.choice(got, min(take, got), replace=False)
        data[cls] = data[cls][idx]

        # Rename fields
        data[cls].dtype.names = map(rename, data[cls].dtype.names)
        
        # Print status
        print "{:<3s}: {:8d}".format(cls, data[cls].size)
        pass

    #### Counts without selection:
    ####   sig:   311083
    ####   bkg: 14730759
    ####
    #### Counts with selection:
    ####   sig:   274890
    ####   bkg:  7569198

    # Concatenate datasets
    data = np.concatenate(list(data.itervalues()))

    # Re-scale fields
    data['m']        *= 1.0E-03
    data['pt']       *= 1.0E-03
    data['truth_pt'] *= 1.0E-03

    # Define selections
    preselection = (data['truth_pt'] > 500.) & (data['truth_pt'] < 1000.)
    mass_window  = (data['m']  >  60.) & (data['m']  <  100.)        

    background = (data['signal'] == 0)
    signal     = ~background

    def get_rejection (data, feat, target_tpr=0.5, preselection=None, selection=None, verbose=False):
        """
        Method to get background rejection at `target_tpr` signal efficiency.

        Arguments:
            data: Numpy recarray holding the fields 'signal', 'weight', and 
                `feat`.
            feat: Name of feature in `data` for which to compute background 
                rejection.
            target_tpr: Signal efficiency at which to compute background 
                rejection.
            preselection: Boolean numpy array indicating any preselection to be 
                performed before computing efficiencies and rejection factors.
            selection: Boolean numpy array indicating any selection to be 
                performed after the pre-selection, in conjunction with a cut on
                `feat` (e.g. a jet mass window selection).
            verbose: Whether to print information during call.

        Returns:
            The background rejection factor, after pre-selection and accounting 
            for any additional selection, computed at the specified signal 
            efficiency.
        """

        # Check(s)
        if preselection is None:
            preselection = np.ones((data.shape[0],)).astype(bool)
            pass

        if selection is None:
            selection = np.ones_like(preselection).astype(bool)
            pass

        # (Opt.) Print information
        if verbose:
            print "=" * 40
            print "get_rejection: Running with {}".format(feat)
            print "\nPre-selection efficiency:"
            print "  Signal:     {:4.1f}%".format(preselection[data['signal'] == 1].mean() * 100.)
            print "  Background: {:4.1f}%".format(preselection[data['signal'] == 0].mean() * 100.)

            print "\nSelection efficiency:"
            print "  Signal:     {:4.1f}%".format(selection[data['signal'] == 1].mean() * 100.)
            print "  Background: {:4.1f}%".format(selection[data['signal'] == 0].mean() * 100.)

            print "\nSelection efficiency, given preselection:"
            print "  Signal:     {:4.1f}%".format(selection[(data['signal'] == 1) & preselection].mean() * 100.)
            print "  Background: {:4.1f}%".format(selection[(data['signal'] == 0) & preselection].mean() * 100.)
            print "=" * 40
            pass

        # Perform pre-selection -- not accounted for in efficiency
        selection = selection[preselection].copy()
        data      = data     [preselection].copy()

        # Compute efficiencies for sequential cuts on `feat`
        eff_sig, eff_bkg, _ = roc_curve(data['signal'][selection], data[feat][selection],
                                        sample_weight=data['weight'][selection])

        # Compute selection efficiencies _after_ preselection
        sel_eff_sig = selection[data['signal'] == 1].mean()
        sel_eff_bkg = selection[data['signal'] == 0].mean()

        # Account for selection efficiency for each class
        eff_sig *= sel_eff_sig
        eff_bkg *= sel_eff_bkg

        # Compute background rejection at `target_tpr` signal efficiency
        idx = np.argmin(np.abs(eff_sig - target_tpr))
        rej = 1. / eff_bkg[idx]

        return rej


    for var in ['Tau21', 'D2']:
        
        rej1 = get_rejection(data, var)
        rej2 = get_rejection(data, var, preselection=preselection)
        rej3 = get_rejection(data, var, selection=mass_window)
        rej4 = get_rejection(data, var, preselection=preselection, selection=mass_window, verbose=True)
        print "Background rejection for {}:".format(var)
        print "  Inclusive sample:               {:4.1f}".format(rej1)
        print "  pT pre-selection:               {:4.1f}".format(rej2)
        print "  Inclusive sample + mass window: {:4.1f}".format(rej3)
        print "  pT pre-selection + mass window: {:4.1f}".format(rej4)

        exit()

        # Compute efficiencies
        msk = mass_window & preselection
        eff_sig, eff_bkg, thresholds = roc_curve(data['signal'][msk],
                                                 data[var][msk],
                                                 sample_weight=data['weight'][msk])

        eff_sig_incl, eff_bkg_incl, _ = roc_curve(data['signal'][preselection],
                                                  data[var][preselection],
                                                  sample_weight=data['weight'][preselection])
        
        # Scale efficiencies
        eff_sig_scaled = eff_sig * mass_window[signal]    .mean()
        eff_bkg_scaled = eff_bkg * mass_window[background].mean()

        # Get background rejection
        target_tpr = 0.5
        idx        = np.argmin(np.abs(eff_sig        - target_tpr))
        idx_incl   = np.argmin(np.abs(eff_sig_incl   - target_tpr))
        idx_scaled = np.argmin(np.abs(eff_sig_scaled - target_tpr))
        rej        = 1. / eff_bkg       [idx]
        rej_incl   = 1. / eff_bkg_incl  [idx_incl]
        rej_scaled = 1. / eff_bkg_scaled[idx_scaled]
        
        print "    Background rejections for {:<6s} {:4.1f} (scaled: {:4.1f} | incl: {:4.1f})".format(var + ':', rej, rej_scaled, rej_incl)
        
        valid        = eff_bkg        > 0.
        valid_incl   = eff_bkg_incl   > 0.
        valid_scaled = eff_bkg_scaled > 0.

        xlim = 0.45, 0.90
        ylim = 0, 90

        fig, ax = plt.subplots()
        plt.plot(eff_sig       [valid],        1. / eff_bkg       [valid],        color='blue')
        plt.plot(eff_sig_incl  [valid_incl],   1. / eff_bkg_incl  [valid_incl],   color='black')
        plt.plot(eff_sig_scaled[valid_scaled], 1. / eff_bkg_scaled[valid_scaled], color='red')

        plt.plot([target_tpr, target_tpr], [ylim[0], rej], color='blue', linestyle=':')
        plt.plot([xlim[0],    target_tpr], [rej,     rej], color='blue', linestyle=':')

        plt.plot([target_tpr, target_tpr], [ylim[0],  rej_incl], color='black', linestyle=':')
        plt.plot([xlim[0],    target_tpr], [rej_incl, rej_incl], color='black', linestyle=':')

        plt.plot([target_tpr, target_tpr], [ylim[0],    rej_scaled], color='red', linestyle=':')
        plt.plot([xlim[0],    target_tpr], [rej_scaled, rej_scaled], color='red', linestyle=':')

        plt.xlabel('Signal efficiency')
        plt.ylabel('Background rejection ({})'.format(var))
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.show()
        pass
    
    # ...

    return 0


# Main function call
if __name__ == '__main__':
    main()
    pass
 
