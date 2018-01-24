#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing optimisation study."""

# Basic import(s)
import os
import glob

# Scientific import(s)
import ROOT
import numpy as np
import root_numpy

# Project import(s)
from adversarial.profile import profile, Profile
from adversarial.new_utils import parse_args, initialise, mkdir
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Common definitions
    # --------------------------------------------------------------------------
    experiment  = 'classifier'
    num_devices = 6

    paths = sorted(glob.glob('optimisation/experiments/{}/output/*.out'.format(experiment)), key=os.path.getmtime)


    # Perform study
    # --------------------------------------------------------------------------
    with Profile("Study: Optimisation"):

        # Loop all run outputs
        results = list()
        for path in paths:

            # Run-log
            with open(path, 'r') as f:

                # Get lines with results
                lines = filter(lambda l: 'result' in l, f)
                if lines:
                    # Get optimisation metric result
                    result = float(lines[0].split()[-1]) / float(num_devices)
                    results.append(result)
                    pass
                pass
            pass

        best_results = [np.min(results[:i+1]) for i in range(len(results))]
        
        # Plot results
        c = rp.canvas(batch=True)
        ymax = 0.06
        oobx = map(lambda t: t[0], filter(lambda t: t[1] > ymax, enumerate(results)))
        ooby = np.ones_like(oobx) * 0.96 * ymax
        
        # Plots
        bins = np.arange(len(results)) + 1
        c.graph(results,      bins=bins, markercolor=rp.colours[1],                 option='AP', label='Evaluations')
        c.graph(ooby,         bins=oobx, markercolor=rp.colours[1], markerstyle=22, option='P')
        c.graph(best_results, bins=bins, linecolor=rp.colours[5],   linewidth=2,    option='L',  label='Best result')

        # Decorations
        c.xlabel("Bayesian optimisation evaluation")
        c.ylabel("Optimisation metric (avg. 5-fold CV L_{clf.}^{val.})")
        c.xlim(0, max(bins) + 1)
        c.ylim(0, ymax)
        c.legend()
        c.text(["#sqrt{s} = 13 TeV",
                "Baseline selection",
                "Standalone classifier (NN) optimisation"
                ],
               qualifier=QUALIFIER)
        # Save
        mkdir('figures/')
        c.save('figures/optimisation_{}.pdf'.format(experiment))
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
