#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for conveniently formatting and storing the W/top-tagging ntuples."""

# Basic import(s)
import gc
import re
import glob
import itertools

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy

# Scientific import(s)
import numpy as np
from numpy.lib import recfunctions
import matplotlib.pyplot as plt

# Project import(s)
from adversarial.utils import mkdir
from adversarial.profile import Profile, profile

# Local import(s)
from .common import *

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Prepare data for training and evaluation of adversarial neural networks for de-correlated jet tagging.")

parser.add_argument('--input', action='store', type=str,
                    default='/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysis2016/FlatNtuplesR21/',
                    help='Input directory, from which to read input ROOT files.')

# Global definition(s)
SELECTION = {
    'common': ["TMath::Abs(fjet_truthJet_eta) < 2.0",
               "fjet_JetpTCorrByCombinedMass >  200E+03",
               "fjet_JetpTCorrByCombinedMass < 2000E+03",
               "fjet_CaloTACombinedMassUncorrelated >  50E+03",
               "fjet_CaloTACombinedMassUncorrelated < 300E+03",
               "fjet_numConstituents > 2",
               ],

    'sig': ["TMath::Abs(fjet_truth_dRmatched_particle_flavor) == 24",  # W
            "TMath::Abs(fjet_dRmatched_WZChild1_dR) < 0.75",
            "TMath::Abs(fjet_dRmatched_WZChild2_dR) < 0.75",
            "TMath::Abs(fjet_truth_dRmatched_particle_dR) < 0.75",
            ],

    'bkg': ["TMath::Abs(fjet_truth_dRmatched_particle_flavor) != 24",  # W
            "TMath::Abs(fjet_truth_dRmatched_particle_flavor) != 23",  # Z
            "TMath::Abs(fjet_truth_dRmatched_particle_flavor) !=  6",  # top
            ],

    }

# -- Add `common` selection to all other keys
common_selection = SELECTION.pop('common')
for key in SELECTION.keys():
    SELECTION[key] = "({})".format(" && ".join(SELECTION[key] + common_selection))
    pass

# -- 'Wtop' indicates use for W-tagging DNN
BRANCHES = [

    # Truth variables
    'fjet_truthJet_pt',

    # Weights
    'fjet_testing_weight_pt',
    ]


def rename (name):
    """Rename ntuple branches to for consistency with naming in project code"""
    if name == 'W':
        return 'signal'
    name = name.replace('fjet_', '')
    name = name.replace('truthJet', 'truth')
    name = name.replace('testing_weight_pt', 'weight_test')
    return name


def main4 ():

    # Example for JZxW background samples
    data = None
    paths = sorted(glob.glob("/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysis2016/FlatNtuplesR21/submitDir-JZ*W/data-tree/*.root"))

    # Loop all Rel. 21 ntuple paths
    for path in paths:

    	# Get data tree
        f = ROOT.TFile(path, 'READ')
        t = f.Get('FlatSubstructureJetTree')

        # Read in data as a numpy recarray
        branches = ['fjet_testing_weight_pt', 'fjet_truthJet_pt']
        part = root_numpy.tree2array(t, branches=branches)

    	# Concatenate
        data = part if data is None else np.concatenate((data, part))
    	pass

    # Plot
    bins = np.linspace(0, 2500, 2500 // 50 + 1, endpoint=True)
    c = rp.canvas(batch=True)
    c.hist(data['fjet_truthJet_pt'] / 1000., bins=bins, weights=data['fjet_testing_weight_pt'], fillcolor=rp.colours[1], alpha=0.5)
    c.logy()
    c.xlabel("Large-#it{R} jet p_{T}^{truth} [GeV]")
    c.ylabel("Number of jets")
    c.text(["#sqrt{s} = 13 TeV, %s" % ("QCD multijets" if key == 'bkg' else "#it{W} jets"),
            "Rel. 21 #it{W} tagging ntuples",
            "No selection"],
           qualifier="Simulation Internal")
    c.save('temp_pt_bkg.pdf')
    return


def main3 ():


    # Get path
    path = '/eos/atlas/user/a/asogaard/adversarial/data/2018-04-07/data.h5'

    # Load data
    data = load_hdf5(path)

    # Create test plot
    for sig, key in zip([0, 1], ['bkg', 'sig']):
        test_plot(data[data['signal'] == sig], 'temp_pt_{}_hdf5_reweight.pdf'.format(key),
                ["#sqrt{s} = 13 TeV, %s" % ("QCD multijets" if key == 'bkg' else "#it{W} jets"),
                 "Rel. 21 #it{W} tagging ntuples",
                 "Baseline selection",
                 "From HDF5 (reweight)"])

        # Define common variables
        msk = data['signal'] == sig
        bins = np.linspace(0, 2500, (2500 - 0) // 50 + 1, endpoint=True)
        opts = dict(bins=bins, alpha=0.4)#, normed=True)

        # Compare un-, train-, and test-weighted
        fig, ax = plt.subplots()
        plt.suptitle('Signal' if sig else 'Background')
        plt.hist(data['pt'][msk].astype(np.float64), weights=data['weight_train'][msk].astype(np.float64), label='Train (flat)', **opts)
        plt.hist(data['pt'][msk].astype(np.float64), weights=data['weight_test'] [msk].astype(np.float64), label='Test (cross-sec.)',  **opts)
        plt.hist(data['pt'][msk].astype(np.float64), label='Unweighted', **opts)
        ax.set_yscale('log')
        plt.xlabel("Jet p_{T} [GeV]")
        plt.ylabel("Fraction of jets")
        plt.legend()
        plt.savefig('temp_pt_{}_hdf5_reweight_pyplot.pdf'.format(key))
        plt.show()
        pass

    return


def main2 ():


    for grp, key in itertools.product(['slim', 'full'], ['sig', 'bkg']):

        # Manual garbage collection
        gc.collect()

        # Get paths
        paths = sorted(glob.glob('/eos/atlas/user/a/asogaard/adversarial/data/2018-04-07/{}/*_{}.h5'.format(key, grp)))

        # Load data
        data = None
        for path in paths:
            part = load_hdf5(path)
            data = part if data is None else np.concatenate((data, part))
            pass

        # Create test plot
        test_plot(data, 'temp_pt_{}_hdf5_{}.pdf'.format(key, grp),
                ["#sqrt{s} = 13 TeV, %s" % ("QCD multijets" if key == 'bkg' else "#it{W} jets"),
                 "Rel. 21 #it{W} tagging ntuples",
                 "Baseline selection",
                 "From HDF5 ({})".format(grp)])
        pass

    return


def main1 ():

    # Parse command-line argument
    args = parser.parse_args()

    # Modify input/output directory names to conform to convention
    if not args.input .endswith('/'): args.input  += '/'

    # Reading input ROOT files
    with Profile("Reading input ROOT files"):

        # Find datasets
        print "Reading input files from:\n  {}".format(args.input)

        path_pattern = args.input + 'submitDir-{}*/data-tree/*.root'

        # Loop classes
        for key in SELECTION.keys():
            data = None

            print "\n== {}".format(key)

            # Get ROOT ntuple paths for current class
            paths = glob.glob(path_pattern.format('wprime' if key == 'sig' else 'JZ'))
            print "   Found {} input data files.".format(len(paths))

            # Sort paths according to JZxW slice index
            if key == 'bkg':
                JZxW = [int(re.search('JZ(\d+)W', path).groups(1)[0]) for path in paths]
                paths = zip(*sorted(zip(JZxW, paths), key=lambda t: t[0]))[1]
            else:
                paths = sorted(paths)
                pass

            # Loop all ROOT ntuple paths
            for ipath, path in enumerate(paths):
                print "   [{}/{}] {}".format(ipath + 1, len(paths), path.split('/')[-1])

                # Manual garbage collection
                gc.collect()

                # Get data tree
                f = ROOT.TFile(path, 'READ')
                t = f.Get('FlatSubstructureJetTree')

                # Read in data as a numpy recarray
                branches = ['fjet_testing_weight_pt', 'fjet_truthJet_pt']
                a = root_numpy.tree2array(t, branches=branches)

                # Rename columns
                a.dtype.names = map(rename, a.dtype.names)

                # Rescale MeV -> GeV
                a['truth_pt'] /= 1000.

                # Concatenate
                data = a if data is None else np.concatenate((data,a))
                pass

            test_plot(data, 'temp_pt_{}.pdf'.format(key),
                    ["#sqrt{s} = 13 TeV, %s" % ("QCD multijets" if key == 'bkg' else "#it{W} jets"),
                     "Rel. 21 #it{W} tagging ntuples",
                     "No selection"])

            pass
        pass

    return


# Main function call
if __name__ == '__main__':
    #main1()
    #main2()
    #main3()
    main4()
    pass
