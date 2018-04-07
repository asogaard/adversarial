#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for conveniently formatting and storing the W/top-tagging ntuples."""

# Basic import(s)
import gc
import glob

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy

# Scientific import(s)
import numpy as np
from numpy.lib import recfunctions

# Project import(s)
from adversarial.utils import mkdir
from adversarial.profile import Profile, profile
from .common import load_hdf5, save_hdf5

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Prepare data for training and evaluation of adversarial neural networks for de-correlated jet tagging.")

parser.add_argument('--input', action='store', type=str,
                    default='/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysis2016/FlatNtuplesR21/',
                    help='Input directory, from which to read input ROOT files.')
parser.add_argument('--output', action='store', type=str,
                    default='/eos/atlas/user/a/asogaard/adversarial/data/2018-04-07/',
                    help='Output directory, to which to write output files.')

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

    # Kinematics
    'fjet_CaloTACombinedMassUncorrelated', # Combined jet mass | Wtop
    'fjet_JetpTCorrByCombinedMass',        # Corrected jet pT  | Wtop
    'fjet_phi', # Kept for control
    'fjet_eta', # Kept for control

    # Event-level info
    'EventInfo_NPV',
    'EventInfo_eventNumber',  # Checking for duplicates
    'EventInfo_runNumber',    # Checking for duplicates
    'fjet_nthLeading',

    # Truth variables
    'fjet_truthJet_pt',

    # Substructure variables.
    'fjet_Tau1_wta',
    'fjet_Tau2_wta',
    'fjet_Tau3_wta',
    'fjet_Tau21_wta', # Wtop
    'fjet_Tau32_wta',

    'fjet_ECF1',
    'fjet_ECF2',
    'fjet_ECF3',
    'fjet_e2',
    'fjet_e3',
    'fjet_C2', # Wtop
    'fjet_D2', # Wtop

    'fjet_Angularity',   # Wtop
    'fjet_Aplanarity',   # Wtop
    'fjet_Dip12',
    'fjet_FoxWolfram20', # Wtop
    'fjet_KtDR',         # Wtop
    'fjet_N2beta1',
    'fjet_PlanarFlow',   # Wtop
    'fjet_Sphericity',
    'fjet_Split12',      # Wtop
    'fjet_Split23',
    'fjet_ThrustMaj',
    'fjet_ThrustMin',
    'fjet_ZCut12',       # Wtop
    'fjet_Qw',
    'fjet_Mu12',

    'fjet_SDt_Dcut1',
    'fjet_SDt_Dcut2',

    # Weights
    'fjet_testing_weight_pt',
    ]


def rename (name):
    """Rename ntuple branches to for consistency with naming in project code"""
    if name == 'W':
        return 'signal'
    name = name.replace('fjet_', '')
    name = name.replace('_wta', '')
    name = name.replace('beta1', '')
    name = name.replace('truthJet', 'truth')
    name = name.replace('CaloTACombinedMassUncorrelated', 'm')
    name = name.replace('JetpTCorrByCombinedMass', 'pt')
    name = name.replace('testing_weight_pt', 'weight_test')
    name = name.replace('EventInfo_NPV', 'npv')
    name = name.replace('EventInfo_', '')
    return name


# Main function definition
@profile
def main ():

    # Parse command-line argument
    args = parser.parse_args()
    
    # Modify input/output directory names to conform to convention
    if not args.input .endswith('/'): args.input  += '/'
    if not args.output.endswith('/'): args.output += '/'

    # Reading input ROOT files
    with Profile("Reading input ROOT files"):

        # Find datasets
        print "Reading input files from:\n  {}".format(args.input)
        print "Writing output files to: \n  {}".format(args.output)

        path_pattern = args.input + 'submitDir-{}*/data-tree/*.root'

        # Loop classes
        data = dict(sig=None, bkg=None)
        for key in SELECTION.keys():
            print "\n== {}".format(key)

            # Get ROOT ntuple paths for current class
            paths = sorted(glob.glob(path_pattern.format('wprime' if key == 'sig' else 'JZ')))
            print "   Found {} input data files.".format(len(paths))

            # Loop all ROOT ntuple paths
            for ipath, path in enumerate(paths):
                print "   [{}/{}] {}".format(ipath + 1, len(paths), path.split('/')[-1])

                # Manual garbage collection
                gc.collect()

                # Get data tree
                f = ROOT.TFile(path, 'READ')
                t = f.Get('FlatSubstructureJetTree')

                # Read in data as a numpy recarray
                a = root_numpy.tree2array(t, branches=BRANCHES, selection=SELECTION[key])
                print "     Got {} samples".format(a.size)

                # Rename columns
                a.dtype.names = map(rename, a.dtype.names)

                # Rescale MeV -> GeV
                a['m']        /= 1000.
                a['pt']       /= 1000.
                a['truth_pt'] /= 1000.

                # Add `signal` column
                signal = (np.ones((a.shape[0],)) * (1 if key == 'sig' else 0)).astype(int)
                a = recfunctions.append_fields(a, 'signal', signal)
                
                # Writing output HDF5 file
                save_hdf5(data, path.replace('.root', '') + '_full.h5')
                pass
            pass
        pass

    return


# Main function call
if __name__ == '__main__':
    print ""
    main()
    pass
