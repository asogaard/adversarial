#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for converting W/top tagging ntuples to HDF5 files.

> 1. Convert
  2. Slim
  3. Reweight
"""

# Basic import(s)
import re
import glob
import multiprocessing

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
import root_numpy

# Scientific import(s)
import numpy as np
from numpy.lib import recfunctions

# Project import(s)
from adversarial.utils import garbage_collect
from adversarial.profile import Profile, profile
from .common import load_hdf5, save_hdf5, get_parser, run_batched

# Command-line arguments parser
parser = get_parser(input=True, output=True, max_processes=True)
parser.description = "Convert ntuples to HDF5."

# Global definition(s)
SELECTION = {
    'common': ["TMath::Abs(fjet_truthJet_eta) < 2.0",
               "fjet_truthJet_pt > 200E+03",
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
    'fjet_CaloTACombinedMassUncorrelated',  # Combined jet mass | Wtop
    'fjet_JetpTCorrByCombinedMass',         # Corrected jet pT  | Wtop
    'fjet_phi',  # Kept for control
    'fjet_eta',  # Kept for control

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
    'fjet_Tau21_wta',  # Wtop
    'fjet_Tau32_wta',

    'fjet_ECF1',
    'fjet_ECF2',
    'fjet_ECF3',
    'fjet_e2',
    'fjet_e3',
    'fjet_C2',  # Wtop
    'fjet_D2',  # Wtop

    'fjet_Angularity',    # Wtop
    'fjet_Aplanarity',    # Wtop
    'fjet_Dip12',
    'fjet_FoxWolfram20',  # Wtop
    'fjet_KtDR',          # Wtop
    'fjet_N2beta1',
    'fjet_PlanarFlow',    # Wtop
    'fjet_Sphericity',
    'fjet_Split12',       # Wtop
    'fjet_Split23',
    'fjet_ThrustMaj',
    'fjet_ThrustMin',
    'fjet_ZCut12',        # Wtop
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

    # Find datasets
    print "Reading input files from:\n  {}".format(args.input)
    print "Writing output files to: \n  {}".format(args.output)

    path_pattern = args.input + 'submitDir-{}*/data-tree/*.root'

    # Loop classes
    for key in SELECTION.keys():
        print "\n== {}".format(key)

        # Get ROOT ntuple paths for current class
        paths = sorted(glob.glob(path_pattern.format('wprime' if key == 'sig' else 'JZ')))
        print "   Found {} input data files.".format(len(paths))

        if len(paths) == 0:
            continue

        # Run batched conversion in parallel
        run_batched(FileConverter, [(path, key, args) for path in paths], max_processes=args.max_processes)
        pass

    return


class FileConverter (multiprocessing.Process):

    def __init__ (self, vargs):
        """
        Process for converting standard-format W/top tagging ROOT file to HDF5.

        Arguments:
            path: Path to the ROOT file to be converted.
            key: Class to which the file pointer to by `path` belongs
            args: Namespace containing command-line arguments, to configure the
                reading and writing of files.
        """

        # Unpack input arguments
        path, key, args = vargs

        # Base class constructor
        super(FileConverter, self).__init__()

        # Member variable(s)
        self.__path = path
        self.__key  = key
        self.__args = args
        return


    @garbage_collect
    def run (self):

        # Get unique file identifier
        identifier = re.search("(WZqqqq_m[\d]+|JZ[\d]+W)\.", self.__path).groups(1)[0]

        # Get data tree
        f = ROOT.TFile(self.__path, 'READ')
        t = f.Get('FlatSubstructureJetTree')

        # Read in data as a numpy recarray
        data = root_numpy.tree2array(t, branches=BRANCHES, selection=SELECTION[self.__key])
        print "     Got {:8d}/{:8d} samples ({})".format(data.size, t.GetEntries(), identifier)

        # Rename columns
        data.dtype.names = map(rename, data.dtype.names)

        # Rescale energy-type variables (MeV -> GeV)
        data['m']        /= 1000.
        data['pt']       /= 1000.
        data['truth_pt'] /= 1000.

        # Add `signal` column
        signal = (np.ones((data.shape[0],)) * int(1 if self.__key == 'sig' else 0)).astype(int)
        data = recfunctions.append_fields(data, 'signal', signal)

        # Writing output HDF5 file
        filename = self.__path.split('/')[-1].replace('.root', '') + '_full.h5'
        save_hdf5(data, self.__args.output + self.__key + '/' + filename)
        return
    pass


# Main function call
if __name__ == '__main__':
    main()
    pass
