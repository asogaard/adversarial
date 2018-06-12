#!/usr/bin/env python

import numpy as np
rng = np.random.RandomState(21)  # For reproducibility
import pandas as pd

import matplotlib.pyplot as plt
import rootplotting as rp

from ROOT import TGraphErrors

import gzip
import pickle



def OpenFile (fi,opt):

    with open(fi, opt) as f:
        x = pickle.load(f)
        pass

    return x

# Main function definition
def main ():

    c = rp.canvas(batch=True)	

    #inputs needed
    x = OpenFile('inputs_Fig5/x_axis_uboost_ur_0p10_te_92_rel21_fixed_FIG5.pkl.gz', 'rb')
    ll_train = OpenFile('inputs_Fig5/train_uboost_ur_0p00_te_92_rel21_fixed_FIG5.pkl.gz', 'rb')
    ll_test = OpenFile('inputs_Fig5/test_uboost_ur_0p00_te_92_rel21_fixed_FIG5.pkl.gz', 'rb')
    ll_train_uboost = OpenFile('inputs_Fig5/train_uboost_ur_0p10_te_92_rel21_fixed_FIG5.pkl.gz', 'rb')
    ll_test_uboost = OpenFile('inputs_Fig5/test_uboost_ur_0p10_te_92_rel21_fixed_FIG5.pkl.gz', 'rb')


    bins = np.arange(len(x), dtype=np.float) + 1
    means_train = np.array(ll_train)
    means_test = np.array(ll_test)
    means_train_ub = np.array(ll_train_uboost)
    means_test_ub = np.array(ll_test_uboost)

    graph1 = TGraphErrors(len(bins), bins, means_train, bins * 0)
    graph2 = TGraphErrors(len(bins), bins, means_test, bins * 0)
    graph3 = TGraphErrors(len(bins), bins, means_train_ub, bins * 0)
    graph4 = TGraphErrors(len(bins), bins, means_test_ub, bins * 0)

    c.graph(graph1, markercolor=rp.colours[4], linecolor=rp.colours[5],linewidth=2, markersize=0.5, linestyle =1,option='AL', label='Train Adaboost ', legend_option='l')
    c.graph(graph2, markercolor=rp.colours[4], linecolor=rp.colours[5],linewidth=2, markersize=0.5, linestyle =2, option='L', label='Test Adaboost', legend_option='l')
    c.graph(graph3, markercolor=rp.colours[1], markerstyle=22, markersize=0.5, linecolor=rp.colours[1],linewidth=2,linestyle =1, option='L', label='Train uBoost (#alpha=0.1)', legend_option='l')
    c.graph(graph4, markercolor=rp.colours[1], markerstyle=22, markersize=0.5, linecolor=rp.colours[1], linewidth=2,linestyle =2, option='L', label='Test uBoost (#alpha=0.1)', legend_option='l')

    ymax = 0.8
    ymin = 0.3
    # Decorations
    c.pad()._yaxis().SetNdivisions(505)
    c.xlabel("Boosting step / no. estimator")
    c.ylabel("Log-loss")
    c.xlim(-10, len(bins))
    c.ylim(ymin, ymax)
    c.legend(xmin=.6,xmax=.7,ymin=0.6,ymax=0.8, width=0.22)
    c.text(["#sqrt{s} = 13 TeV", "#it{W} jet tagging"], qualifier="Simulation Internal")
    # Save
    c.save('./uboost_vs_adaboost_logloss_fig5.pdf')


    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    #args = parse_args()

    # Call main function
    main()
    pass

