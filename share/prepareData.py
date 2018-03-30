#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for conveniently formatting and storing the W/top-tagging ntuples."""

# Basic import(s)
import os
import h5py

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
import numpy as np
import root_numpy

# Project import(s)
from adversarial.utils import mkdir
from adversarial.profile import Profile, profile

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description="Prepare data for training and evaluation of adversarial neural networks for de-correlated jet tagging.")

parser.add_argument('--input', action='store', type=str,
                    default='/eos/atlas/atlascerngroupdisk/perf-jets/JSS/TopBosonTagAnalysis2016/FlatNtuplesR21/',
                    help='Input directory, from which to read input ROOT files.')
parser.add_argument('--output', action='store', type=str,
                    default='/eos/atlas/user/a/asogaard/adversarial/data/2018-03-30/',
                    help='Output directory, to which to write output files.')

# Global definition(s)
SELECTION = {
    'common': ("(fjet_truthJet_pt >  200E+03 && "
               " fjet_truthJet_pt < 2000E+03 && "
               " fjet_truthJet_eta > -2.0 && "
               " fjet_truthJet_eta <  2.0 && "
               " fjet_m >  50E+03 && "
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

# -- 'Wtop' indicates use for W-tagging DNN
BRANCHES = [
    # Kinematics
    'fjet_CaloTACombinedMassUncorrelated', # Combined jet mass | Wtop
    'fjet_JetpTCorrByCombinedMass',        # Corrected jet pT  | Wtop
    'fjet_phi', # Kept for control
    'fjet_eta', # Kept for control

    # Event-level info
    'EventInfo_NPV',   # Kept for control
    'fjet_nthLeading', # @TODO: Check. Only use == 1?

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
    'fjet_training_weight_pt_W',

    # Signal flag
    'W',
    ]


def rename (name):
    """Rename ntuple branches to for consistency with naming in project code"""
    if name == 'W':
        return 'signal'
    name = name.replace('fjet_', '')
    name = name.replace('_wta', '')
    name = name.replace('beta1', '')
    name = name.replace('CaloTACombinedMassUncorrelated', 'm')
    name = name.replace('JetpTCorrByCombinedMass', 'pt')
    name = name.replace('training_weight_pt_W', 'weight_train')
    name = name.replace('testing_weight_pt', 'weight_test')
    return name


# Main function definition
@profile
def main ():

    # Reading in NTuples
    # --------------------------------------------------------------------------
    with Profile("reading"):

        # Parse command-line argument
        args = parser.parse_args()

        #  Modify input/output directory names to conform to convention
        if not args.input .endswith('/'): args.input  += '/'
        if not args.output.endswith('/'): args.output += '/'

        # Relative paths to training- and test sets
        train_file = 'Training/Wtagging_fullycontained/training_W_tagging_fully_contained.root'
        test_file1 = 'Testing/Wtagging/testing_Wprime.root'
        test_file2 = 'Testing/background/testing_dijet.root'
        files = [train_file, test_file1, test_file2]

        print "Reading input files from:\n  {}".format(args.input)
        print "Writing output files to: \n  {}".format(args.output)
        print "Assuming the files:{}\n exist in the input directory.".format('\n  '.join([''] + files))

        paths = [args.input + f for f in files]

        data = None
        for key in SELECTION.keys():
            for path in paths:
                f = ROOT.TFile(path, 'r')
                t = f.Get('FlatSubstructureJetTree')
                a = root_numpy.tree2array(t, branches=BRANCHES, selection=SELECTION[key])
                a.dtype.names = map(rename, a.dtype.names)
                data = a if (data is None) else np.concatenate((data, a))
                pass
            pass

        # Rescale MeV -> GeV
        data['m']  /= 1000.
        data['pt'] /= 1000.

        print data.dtype.names

        pass


    # Writing out HDF5 file(s)
    # --------------------------------------------------------------------------
    with Profile("writing"):

        # Save as HDF5
        mkdir(args.output)
        with h5py.File(args.output + 'data.h5', 'w') as hf:
            hf.create_dataset('dataset',  data=data)
            pass

        pass

    # Testing reading in HDF5 file(s)
    # --------------------------------------------------------------------------
    with Profile("reading test"):
        with h5py.File(args.output + 'data.h5', 'r') as hf:
            new_data = hf['dataset'][:]
            pass
        print "Read {} samples.".format(new_data.shape[0])
        pass

    return


# Main function call
if __name__ == '__main__':
    print ""
    main()
    pass
