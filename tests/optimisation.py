#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing optimisation study."""

# Basic import(s)
import os
import re
import glob

# Scientific import(s)
from ROOT import TGraphErrors
import numpy as np

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
        means, stds, results = list(), list(), list()
        for path in paths:

            # Run-log
            with open(path, 'r') as f:
                lines = [l for l in f]
                # Number of training epochs, to identify the last one
                num_epochs = max(map(int, map(lambda l: l.split('/')[-1], filter(lambda l: re.search('^Epoch [\d]+/[\d]+ *$', l), lines))))

                # Indices of line holding the results for the last training epoch in each CV fold
                try:
                    indices = np.array(zip(*filter(lambda t: 'Epoch {e}/{e}'.format(e=num_epochs) in t[1], enumerate(lines)))[0]) + 1
                except IndexError:
                    # ...
                    continue

                # Validation losses for each CV fold
                val_losses = list()
                for idx in indices:
                    fields = lines[idx].split()
                    num_devices = len(filter(lambda f: 'val_classifier_loss_' in f, fields))
                    jdx = fields.index('val_loss:') + 1
                    val_loss = float(fields[jdx]) / num_devices
                    val_losses.append(val_loss)
                    pass

                # Append results for current evaluation
                means.append(np.mean(val_losses))
                stds .append(np.std (val_losses))

                """
                # Get lines with results
                lines = filter(lambda l: 'result' in l, lines)
                if lines:
                    # Get optimisation metric result
                    result = float(lines[0].split()[-1]) / float(num_devices)
                    results.append(result)
                    pass
                """
                pass
            pass

        # Compute running, best mean
        means = np.array(means)
        stds  = np.array(stds)
        bins = np.arange(len(means), dtype=np.float) + 1.
        best_mean = np.array([np.min(means[:i+1]) for i in range(len(means))])
        idx_improvements = [0] + list(np.where(np.abs(np.diff(best_mean)) > 0)[0] + 1)

        # Create graph
        graph = TGraphErrors(len(bins), bins, means, bins * 0, stds)

        # Plot results
        c = rp.canvas(batch=True)
        ymax = 0.06 * 6
        oobx = map(lambda t: t[0], filter(lambda t: t[1] > ymax, enumerate(means)))
        ooby = np.ones_like(oobx) * 0.96 * ymax

        # Plots
        c.graph(graph,     markercolor=rp.colours[1], linecolor=rp.colours[1], option='AP', label='Evaluations (CV avg. #pm std.)')
        #c.graph(results,   bins=bins, markercolor=rp.colours[5], option='AP', label='"result"')
        c.graph(ooby,      bins=oobx, markercolor=rp.colours[1], markerstyle=22, option='P')
        c.graph(best_mean, bins=bins, linecolor=rp.colours[5],   linewidth=2,    option='L',  label='Best result')
        c.graph(best_mean[idx_improvements], bins=bins[idx_improvements],  markercolor=rp.colours[5], markersize=0.5, option='P')

        # Decorations
        c.xlabel("Bayesian optimisation evaluation")
        c.ylabel("Optimisation metric (5-fold CV L_{clf.}^{val.})")
        c.xlim(0, max(bins) + 1)
        c.ylim(0, ymax)
        c.legend(xmin=0.475, width=0.22)
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
