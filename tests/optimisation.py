#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing optimisation study."""

# Basic import(s)
import re
import glob

# Scientific import(s)
from ROOT import TGraphErrors
import numpy as np

# Project import(s)
from adversarial.utils import parse_args, initialise, mkdir
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Common definitions
    experiment  = 'classifier'
    paths = sorted(glob.glob('optimisation/output/{}/output/*.out'.format(experiment)))
    
    num_steps = 100

    # Loop all run outputs
    means, stds, results = list(), list(), list()
    for path in paths[:num_steps]:

        # Run-log
        with open(path, 'r') as f:
            lines = [l for l in f]

            # Number of training epochs, to identify the last one
            num_epochs = max(map(int, map(lambda l: l.split('/')[-1], filter(lambda l: re.search('^Epoch [\d]+/[\d]+ *$', l), lines))))

            # Indices of line holding the results for the last training epoch in each CV fold
            try:
                indices = np.array(zip(*filter(lambda t: 'Epoch {e}/{e}'.format(e=num_epochs) in t[1], enumerate(lines)))[0]) + 1
            except IndexError: continue

            # Validation losses for each CV fold
            val_losses = list()
            for idx in indices:
                fields = lines[idx].split()
                jdx = fields.index('val_loss:') + 1
                val_loss = float(fields[jdx])
                val_losses.append(val_loss)
                pass

            # Append results for current evaluation
            means.append(np.mean(val_losses))
            stds .append(np.std (val_losses))
            pass
        pass

    # Check losses
    print "Optimisation metrics, sorted by mean + 1 sigma, for robustness:"
    for i,m,s in sorted(zip(range(len(means)), means, stds), key=lambda t: t[1] + t[2]):
        print "  [{:3d}] {:7.4f} ± {:6.4f}".format(i + 1, m,s)
        pass

    # Compute running, best mean
    means = np.array(means)
    stds  = np.array(stds)
    bins = np.arange(len(means), dtype=np.float) + 1
    best_mean = np.array([np.min(means[:i+1]) for i in range(len(means))])
    idx_improvements = [0] + list(np.where(np.abs(np.diff(best_mean)) > 0)[0] + 1)

    # Create graph
    graph = TGraphErrors(len(bins), bins, means, bins * 0, stds)

    # Plot
    plot(experiment, means, graph, idx_improvements, best_mean, bins)

    return 0


def plot (*argv):
    """
    Method for delegating plotting.
    """

    # Unpack arguments
    experiment, means, graph, idx_improvements, best_mean, bins = argv

    # Plot results
    c = rp.canvas(batch=True)
    ymax = 1.0 # 1.5
    ymin = 0.3
    oobx = map(lambda t: t[0], filter(lambda t: t[1] > ymax, enumerate(means)))
    ooby = np.ones_like(oobx) * 0.96 * (ymax - ymin) + ymin

    # Plots
    markersize = 0.8
    c.graph(graph, markercolor=rp.colours[1], linecolor=rp.colours[1], markerstyle=20, markersize=markersize, option='AP', label='Evaluations', legend_option='PE')
    c.graph(ooby, bins=oobx, markercolor=rp.colours[1], markerstyle=22, option='P')
    c.graph(best_mean, bins=bins, linecolor=rp.colours[5], linewidth=2, option='L')
    c.graph(best_mean[idx_improvements], bins=bins[idx_improvements], markercolor=rp.colours[5], markerstyle=24, markersize=markersize, option='P')

    # Decorations
    c.pad()._yaxis().SetNdivisions(505)
    c.xlabel("Bayesian optimisation step")
    c.ylabel("Cross-validation optimisation metric, L_{clf}^{val}")
    c.xlim(0, len(bins))
    #c.ylim(0, ymax)
    c.ylim(0.3, 1.0)
    c.legend(width=0.22, ymax=0.816, categories=[
            ('Best result', dict(linecolor=rp.colours[5], linewidth=2, markercolor=rp.colours[5], markerstyle=24, option='LP')),
        ])
    c.text(["#sqrt{s} = 13 TeV",
            "Neural network (NN) classifier"
            ],
           qualifier=QUALIFIER)
    # Save
    mkdir('figures/optimisation/')
    c.save('figures/optimisation/optimisation_{}.pdf'.format(experiment))

    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
