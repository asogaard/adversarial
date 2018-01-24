#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing loss study."""

# Basic import(s)
import glob
import json

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
    num_folds   = 5
    num_devices = 6

    paths = sorted(glob.glob('trained/history__crossval_{}__*of{}.json'.format(experiment, num_folds)))
    


    # Perform study
    # --------------------------------------------------------------------------
    with Profile("Study: Loss"):

        losses = {'train': list(), 'val': list()}
        for path in paths:
            with open(path, 'r') as f:
                d = json.load(f)
                pass
            
            loss = np.array(d['val_loss']) / float(num_devices)
            losses['val'].append(loss)
            loss = np.array(d['loss']) / float(num_devices)
            losses['train'].append(loss)
            pass

        # Plot results
        c = rp.canvas(batch=True)
        bins     = np.arange(len(loss)) + 1
        histbins = np.arange(len(loss) + 1) + 0.5
        
        # Plots
        categories = list()
        # -- Validation
        loss_mean = np.mean(losses['val'], axis=0)
        loss_std  = np.std (losses['val'], axis=0)
        hist = ROOT.TH1F('val_loss', "", len(histbins) - 1, histbins)
        for idx in range(len(loss_mean)):            
            hist.SetBinContent(idx + 1, loss_mean[idx])
            hist.SetBinError  (idx + 1, loss_std [idx])
            pass

        ROOT.gStyle.SetTitleOffset(2.2, 'y')
        c.hist([0], bins=[0,50], linewidth=0, linestyle=0)  # Force correct x-axis
        c.hist(hist, fillcolor=rp.colours[5], alpha=0.3, option='LE3')
        c.hist(hist, linecolor=rp.colours[5], linewidth=3, option='HISTL')

        categories += [('Validation (CV avg. #pm std.)',
                        {'linestyle': 1, 'linewidth': 3,
                         'linecolor': rp.colours[5], 'fillcolor': rp.colours[5],
                         'alpha': 0.3,  'option': 'FL'})]

        # -- Training
        loss_mean = np.mean(losses['train'], axis=0)
        loss_std  = np.std (losses['train'], axis=0)
        hist = ROOT.TH1F('loss', "", len(histbins) - 1, histbins)
        for idx in range(len(loss_mean)):            
            hist.SetBinContent(idx + 1, loss_mean[idx])
            hist.SetBinError  (idx + 1, loss_std [idx])
            pass

        c.hist(hist, fillcolor=rp.colours[1], alpha=0.3, option='LE3')
        c.hist(hist, linecolor=rp.colours[1], linewidth=3, linestyle=2, option='HISTL')

        categories += [('Training    (CV avg. #pm std.)',
                        {'linestyle': 2, 'linewidth': 3,
                         'linecolor': rp.colours[1], 'fillcolor': rp.colours[1],
                         'alpha': 0.3,  'option': 'FL'})]
        
        # Decorations
        c.xlabel("Cross-validation (CV) training epoch")
        c.ylabel("Optimisation metric (L_{clf.}^{val.})")
        c.xlim(0, max(bins))
        c.ylim(0.012, 0.018)
        c.legend(xmin=0.475, categories=categories)
        c.text(["#sqrt{s} = 13 TeV",
                "Baseline selection",
                "Standalone classifier (NN) optimisation"
                ],
               qualifier=QUALIFIER)
        # Save
        mkdir('figures/')
        c.save('figures/loss_{}.pdf'.format(experiment))
        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
