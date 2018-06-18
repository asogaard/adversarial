#!/usr/bin/env python

# Basic import(s)
import pickle
import numpy as np
from glob import glob
rng = np.random.RandomState(21)  # For reproducibility

# Project import(s)
import rootplotting as rp


# Utility method(s)
def OpenFile (fi, opt='rb'):
    """
    Return pickled file contents.
    """
    with open(fi, opt) as f:
        x = pickle.load(f)
        pass

    return x


# Main function definition
def main ():

    # Global defintions
    alpha     = 0.3
    nb_epochs = 500

    # Load epochs and log-losses from files
    #pattern =
    #'tests/inputs_Fig5/{v:s}_uboost_ur_{a:4.2f}_te_92_rel21_fixed_FIG5'
    pattern = 'tests/inputs_Fig5/{v:s}_uboost_ur_{a:4.2f}_te_92_rel21_fixed_*'
    x           = OpenFile(glob(pattern.format(v='x_axis', a=alpha).replace('.', 'p') + '.pkl.gz')[0])[:nb_epochs]
    ll_ab_train = OpenFile(glob(pattern.format(v='train',  a=0)    .replace('.', 'p') + '.pkl.gz')[0])[:nb_epochs]
    ll_ab_test  = OpenFile(glob(pattern.format(v='test',   a=0)    .replace('.', 'p') + '.pkl.gz')[0])[:nb_epochs]
    ll_ub_train = OpenFile(glob(pattern.format(v='train',  a=alpha).replace('.', 'p') + '.pkl.gz')[0])[:nb_epochs]
    ll_ub_test  = OpenFile(glob(pattern.format(v='test',   a=alpha).replace('.', 'p') + '.pkl.gz')[0])[:nb_epochs]

    # Plot log-loss curves
    c = rp.canvas(batch=True)	
    
    # -- Common plotting options
    opts = dict(linewidth=2, legend_option='L')
    c.graph(ll_ab_train, bins=x, linecolor=rp.colours[5], linestyle=1, option='AL', label='AdaBoost', **opts)
    c.graph(ll_ab_test,  bins=x, linecolor=rp.colours[5], linestyle=2, option='L',  **opts)
    c.graph(ll_ub_train, bins=x, linecolor=rp.colours[1], linestyle=1, option='L',  label='uBoost', **opts)
    c.graph(ll_ub_test,  bins=x, linecolor=rp.colours[1], linestyle=2, option='L',  **opts)

    # -- Decorations
    c.pad()._yaxis().SetNdivisions(505)
    c.xlabel("Training epoch")
    c.ylabel("BDT classifier loss")
    c.xlim(0, len(x))
    c.ylim(0.3, 1.4) 
    c.legend(width=0.28)
    c.legend(header='Dataset:',
             categories=[('Training', {'linestyle': 1}),
                         ('Testing',  {'linestyle': 2})],
             width=0.28, ymax=0.69)

    for leg in c.pad()._legends:
        leg.SetFillStyle(0)
        pass
    
    c.text(["#sqrt{s} = 13 TeV",
            "#it{W} jet tagging",
            "Uniforming rate #alpha = {:3.1f}".format(alpha)],
           qualifier="Simulation Internal")
    
    # -- Save
    c.save('figures/test_uboost_vs_adaboost_logloss_fig5_alpha{:4.2f}'.format(alpha).replace('.', 'p') + '.pdf')
    return


# Main function call
if __name__ == '__main__':

    # Call main function
    main()
    pass

