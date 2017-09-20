#!/usr/bin/env python

"""..."""

# Basic import(s)
# ...

# Scientific import(s)
import numpy as np
from root_numpy import fill_profile
import ROOT

import matplotlib.pyplot as plt
plt.switch_backend('pdf')
plt.style.use('edinburgh')

# Custom import(s)
#from rootplotting import ap

# Project import(s)
from adversarial.profile import *

# Global variables
linestyles = ['-', '--', '-.', ':']
colours    = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))


def wpercentile (data, percents, weights=None):
    ''' percents in units of 1%
    weights specifies the frequency (count) of data.
    From [https://stackoverflow.com/a/31539746]
    '''
    if weights is None:
        return np.percentile(data, percents)
    ind = np.argsort(data)
    d = data[ind]
    w = weights[ind]
    p = 100. * w.cumsum() / w.sum()
    y = np.interp(percents, p, d)
    return y


@profile
def plot_profiles (data, args, var, name='tagger_profile', title=''):
    """Plot percentile profiles of tagger varaibles versus jet mass.

    Args:
        data: Dict containing all relevant data: `X`, `Y`, `P`, `W`, `sig`, and
            `bkg`.
        args: Namespace containing command-line arguments.
        var: Name of tagger variable, to be found in `data['bkg']`, _or_ a Keras
            model assumed to be the classifier to be profiled and to take
            `data['X']` as input.
        name: Name of file to which to save the figure (.pdf suffix appended
            automatically).
        title: Figure (sub-)title.
    """

    # Create figure
    fig, ax = plt.subplots()

    # Plotting variables
    edges = np.linspace(0, 300, 30 + 1, endpoint=True)
    bins  = edges[:-1] + 0.5 * np.diff(edges)
    step = 10.
    percentiles = np.linspace(step, 100 - step, int(100 / step) - 1, endpoint=True)
    profiles = [[] for _ in percentiles]
    num_bootstrap = 10
    classifier = None
    
    # Get tagger variable array
    if isinstance(var, str):
        # Assume `var` is a valid key in `data['bkg']`
        tagger = data['bkg'][var]
    else:
        # Assume `var` is a Keras model describing a classifier taking
        # `data['X']` as input
        classifier, var = var, var.name
        msk_bkg = (data['Y'] == 0.)
        tagger = classifier.predict(data['X'][msk_bkg], batch_size=2048).flatten()
        pass

    masses = data['bkg']['m']
    weight = data['bkg']['weight']
    
    # Loop mass bins
    for (mass_down, mass_up) in zip(edges[:-1], edges[1:]):
        
        # Get array of `var` within the current jet-mass band
        msk = (data['bkg']['m'] >= mass_down) & (data['bkg']['m'] < mass_up)
        arr_tagger = tagger[msk]
        arr_weight = data['bkg']['weight'][msk]
        
        # Perform bootstrapping of the tagger variable array to estimate error
        # bands on percentile contours.
        # Cf. e.g. [https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf]
        bootstrap_percentiles = [[] for _ in percentiles]
        num_samples = len(arr_tagger)
        for bts in range(num_bootstrap):
            
            if num_samples == 0:
                indices = []
            else:
                indices = np.random.choice(num_samples, num_samples, replace=True)
                pass
            
            bts_tagger = arr_tagger[indices] 
            bts_weight = arr_weight[indices]
            
            # Loop percentiles for current mass bins
            for idx, perc in enumerate(percentiles):
                if len(bts_tagger) == 0:
                    bootstrap_percentiles[idx].append(np.nan)
                else:
                    bootstrap_percentiles[idx].append(wpercentile(bts_tagger, perc, bts_weight))
                    pass
                pass
            pass

        errorbands = np.std(bootstrap_percentiles, axis=1)
        
        # Loop percentiles
        for idx, perc in enumerate(percentiles):
            if len(arr_tagger) == 0:
                profiles[idx].append(np.nan)
            else:
                profiles[idx].append(wpercentile(arr_tagger, perc, arr_weight))
                pass
            pass # end: loop percentiles
        pass # end: loop mass bins
    
    # Plot profile
    for profile, error, perc in zip(profiles, errorbands, percentiles):
        plt.plot(bins, profile, color=colours[0], linewidth=2 if perc == 50 else 1, label='Median' if perc == 50 else None)
        plt.fill_between(bins, profile - error, profile + error, color=colours[0], alpha=0.1 if perc == 50 else 0.05, label='Bootstr. RMS' if perc == 50 else None)
        pass

    # Plot mean profile with error bars
    profile = ROOT.TProfile('profile', "", len(bins), edges)
    fill_profile(profile, np.vstack((masses, tagger)).T, weight)

    means, rmses = list(), list()
    for i in range(1, 1 + len(bins)):
        mean = profile.GetBinContent(i)
        rms  = profile.GetBinError  (i)
        if mean == 0 and rms == 0:
            mean = np.nan
            rms  = np.nan
            pass
        means.append(mean)
        rmses.append(rms)
        pass

    binwidths = np.diff(edges) / 2.
    plt.errorbar(bins, means, xerr=binwidths, yerr=rmses, fmt='k.', color='black', label=r'Mean $\pm$ RMS')
    
    # Text
    mid = len(percentiles) // 2
    
    arr_profiles = np.array(profiles).flatten()
    arr_profiles = arr_profiles[~np.isnan(arr_profiles)]
    diff = np.max(arr_profiles) - np.min(arr_profiles)
    
    opts = dict(horizontalalignment='center', verticalalignment='bottom', fontsize='x-small')
    text_string = r"$\varepsilon_{bkg.}$ = %d%%"
    
    plt.text(edges[-1], profiles[-1] [-1] + 0.02 * diff, text_string % percentiles[-1],  **opts) # 90%
    
    opts = dict(horizontalalignment='left', verticalalignment='center', fontsize='x-small')
    text_string = "%d%%"
    
    plt.text(edges[-1], profiles[mid][-1], text_string % percentiles[mid], **opts) # 50%
    plt.text(edges[-1], profiles[0]  [-1], text_string % percentiles[0],   **opts) # 10%

    # Decorations
    plt.xlabel("Jet mass [GeV]",  horizontalalignment='right', x=1.0)
    plt.ylabel("{}".format(var),  horizontalalignment='right', y=1.0)
    plt.title('Percentile profiles for {}{}'.format(var, (': ' + title) if title else ''), fontweight='medium')
    plt.legend()
    if classifier is not None:
        plt.ylim(-0.05, 1.05)
        pass

    # Save figure
    plt.savefig(args.output + '{}__{}.pdf'.format(name, var))
    
    # Close figure
    plt.close()

    return


@profile
def plot_posterior (data, args, adversary, name='posterior', title=''):
    """..."""

    # @TODO:
    # - Documentation
    # - Proper treatment of > 1 de-correlation variables
    
    # Create figure
    fig, ax = plt.subplots()
    
    # Variable definitions
    edges  = np.linspace(-0.2, 1.2, 2 * 70 + 1, endpoint=True)
    
    z_slices  = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    #P2_slices = [0, 1]
    P1        = np.linspace(0, 1, 1000 + 1, endpoint=True)
        
    # Plot prior
    msk = (data['Y'] == 0)
    plt.hist(data['P'][:,0][msk], bins=edges, weights=data['U'][msk], normed=True, color='gray', alpha=0.5, label='Prior')
    
    # Plot adversary posteriors
    # for i, P2_slice in enumerate(P2_slices):
    # P2 = np.ones_like(P1) * P2_slice
    # P_test = np.vstack((P1, P2)).T
    P_test = P1
    for j, z_slice in enumerate(z_slices):
        z_test = np.ones_like(P1) * z_slice
        posterior = adversary.predict([z_test, P_test])
        plt.plot(P1, posterior, color=colours[j], label='clf. = %.1f' % z_slices[j])
        pass
    
    # Decorations
    plt.xlabel("Normalised jet log(m)", horizontalalignment='right', x=1.0)
    plt.ylabel("Jets",                  horizontalalignment='right', y=1.0)
    plt.title("De-correlation p.d.f.'s{}".format((': ' + title) if title else ''), fontweight='medium')
    plt.ylim(0, 5.)
    plt.legend()
    
    # Save figure
    plt.savefig(args.output + name + '.pdf')

    # Close figure
    plt.close(fig)

    return


"""
    # Plotting: Re-weighting
    # --------------------------------------------------------------------------
    with Profile("Plotting: Re-weighting"):
        

        fig, ax = plt.subplots(2, 4, figsize=(12,6))

        w_bkg  = bkg['weight']
        rw_bkg = bkg['reweight']
        w_tar  = np.ones((N_tar,)) * np.sum(bkg['weight']) / float(N_tar)

        for row, var in enumerate(['m', 'pt']):
            edges = np.linspace(0, np.max(bkg[var]), 60 + 1, endpoint=True)
            nbins  = len(edges) - 1

            v_bkg  = bkg[var]     # Background  mass/pt values for the background
            rv_bkg = P_bkg[:,row] # Transformed mass/pt values for the background
            rv_tar = P_tar[:,row] # Transformed mass/pt values for the targer

            ax[row,0].hist(v_bkg,  bins=edges, weights=w_bkg,  alpha=0.5, label='Background')
            ax[row,1].hist(v_bkg,  bins=edges, weights=rw_bkg, alpha=0.5, label='Background')
            ax[row,2].hist(rv_bkg, bins=nbins, weights=w_bkg,  alpha=0.5, label='Background') # =rw_bkg
            ax[row,2].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')
            ax[row,3].hist(rv_bkg, bins=nbins, weights=rw_bkg, alpha=0.5, label='Background')
            ax[row,3].hist(rv_tar, bins=nbins, weights=w_tar,  alpha=0.5, label='Target')

            for col in range(4):
                if col < 4: # 3
                    ax[row,col].set_yscale('log')
                    ax[row,col].set_ylim(1E+01, 1E+06)
                    if row == 1:
                        ax[row,col].set_ylim(1E-01, 1E+05)
                        pass
                    pass
                ax[row,col].set_xlabel("Jet %s%s%s" % (var, " (transformed)" if col > 1 else '', " (re-weighted)" if (col + 1) % 2 == 0 else ''))
                if col == 0:
                    ax[row,col].set_ylabel("Jets / {:.1f} GeV".format(np.diff(edges)[0]))
                    pass
                pass
            pass

        plt.legend()
        plt.savefig(args.output + 'priors_1d.pdf')

        # Plot 2D prior before and after re-weighting
        log.debug("Plotting 2D prior before and after re-weighting")
        fig, ax = plt.subplots(1,2,figsize=(11,5), sharex=True, sharey=True)
        h = ax[0].hist2d(P_bkg[:,0], P_bkg[:,1], bins=40, weights=bkg['weight'],   vmin=0, vmax=5, normed=True)
        h = ax[1].hist2d(P_bkg[:,0], P_bkg[:,1], bins=40, weights=bkg['reweight'], vmin=0, vmax=5, normed=True)
        ax[0].set_xlabel("Scaled log(m)")
        ax[1].set_xlabel("Scaled log(m)")
        ax[0].set_ylabel("Scaled log(pt)")
        
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.925, 0.15, 0.025, 0.7])
        fig.colorbar(h[3], cax=cbar_ax)
        plt.savefig(args.output + 'priors_2d.pdf')
        pass


    # Plotting: Cost log for classifier-only fit
    # --------------------------------------------------------------------------
    # Optimal number of training epochs
    opt_epochs = None
    
    with Profile("Plotting: Cost log, cross-val."):        

        fig, ax = plt.subplots()
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
        
        # @NOTE: Assuming no early stopping
        epochs = 1 + np.arange(len(histories[0]['loss']))
        
        for fold, hist in enumerate(histories): 
            plt.plot(epochs, hist['val_loss'], color=colours[1], linewidth=0.6, alpha=0.3,
                     label='Validation (fold)' if fold == 0 else None)
            pass
        
        val_avg = np.mean([hist['val_loss'] for hist in histories], axis=0)
        plt.plot(epochs, val_avg,   color=colours[1], label='Validation (avg.)')

        # Store the optimal number of training epochs        
        opt_epochs = epochs[np.argmin(val_avg)]
        log.info("Using optimal number of {:d} training epochs".format(opt_epochs))
        
        for fold, hist in enumerate(histories):
            plt.plot(epochs, hist['loss'],     color=colours[0], linewidth=1.0, alpha=0.3,
                     label='Training (fold)'   if fold == 0 else None)
            pass
        
        train_avg = np.mean([hist['loss'] for hist in histories], axis=0)
        plt.plot(epochs, train_avg, color=colours[0], label='Train (avg.)')
        
        plt.title('Classifier-only, stratified {}-fold training'.format(args.folds), fontweight='medium')
        plt.xlabel("Training epochs",    horizontalalignment='right', x=1.0)
        plt.ylabel("Objective function", horizontalalignment='right', y=1.0)
        
        epochs = [0] + list(epochs)
        step = max(int(np.floor(len(epochs) / 10.)), 1)
        
        plt.xticks(filter(lambda x: x % step == 0, epochs))
        plt.legend()
        plt.savefig(args.output + 'costlog.pdf')
        pass
    

    # Plotting: Cost log for adversarial fit
    # --------------------------------------------------------------------------
    with Profile("Plotting: Cost log, adversarial, full"):        

        fig, ax = plt.subplots()
        colours = map(lambda d: d['color'], list(plt.rcParams["axes.prop_cycle"]))
        history = result['history']
        print "history keys:", history.keys()
        epochs = 1 + np.arange(len(history['loss']))
        lambda_reg = cfg['adversary']['model']['lambda_reg']
        lr_ratio   = cfg['adversary']['model']['lr_ratio']
        
                   
        classifier_loss = np.mean([loss for key,loss in history.iteritems() if key.startswith('adversary') and int(key.split('_')[-1]) % 2 == 1 ], axis=0)
        adversary_loss  = np.mean([loss for key,loss in history.iteritems() if key.startswith('adversary') and int(key.split('_')[-1]) % 2 == 0 ], axis=0) * lambda_reg
        #combined_loss   = np.array(history['loss']) * lambda_reg /
        #float(lr_ratio)
        combined_loss   = classifierloss + adversary_loss
        
        plt.plot(epochs, classifier_loss, color=colours[0],  linewidth=1.4,  label='Classifier')
        plt.plot(epochs, adversary_loss,  color=colours[1],  linewidth=1.4,  label=r'Adversary (\lambda = {})'.format(lambda_reg))
        plt.plot(epochs, combined_loss,   color=colours[-1], linestyle='--', label='Combined')

        plt.title('Adversarial training', fontweight='medium')
        plt.xlabel("Training epochs", horizontalalignment='right', x=1.0)
        plt.ylabel("Objective function",   horizontalalignment='right', y=1.0)
        ax.set_yscale('log')
        
        epochs = [0] + list(epochs)
        step = max(int(np.floor(len(epochs) / 10.)), 1)
        
        plt.xticks(filter(lambda x: x % step == 0, epochs))
        plt.legend()
        plt.savefig(args.output + 'adversary_costlog.pdf')
        pass

    
    # Plotting: Distributions/ROC
    # --------------------------------------------------------------------------
    with Profile("Plotting: Distributions/ROC"):

        # Tagger variables
        variables = ['tau21', 'D2', 'NN', 'ANN']

        # Plotted 1D tagger variable distributions
        fig, ax = plt.subplots(1, len(variables), figsize=(len(variables) * 4, 4))


        for ivar, var in enumerate(variables):

            # Get axis limits
            if var == 'D2':
                edges = np.linspace(0, 5, 50 + 1, endpoint=True)
            else:
                edges = np.linspace(0, 1, 50 + 1, endpoint=True)
                pass

            # Get value- and weight arrays
            v_sig = np.array(sig[var])
            v_bkg = np.array(bkg[var])

            w_sig = np.array(sig['weight'])
            w_bkg = np.array(bkg['weight'])

            # Mask out NaN's
            msk = ~np.isnan(sig[var])
            v_sig = v_sig[msk]
            w_sig = w_sig[msk]
            
            msk = ~np.isnan(bkg[var])
            v_bkg = v_bkg[msk]
            w_bkg = w_bkg[msk]

            # Plot distributions
            ax[ivar].hist(v_bkg, bins=edges, weights=w_bkg, alpha=0.5, normed=True, label='Background')
            ax[ivar].hist(v_sig, bins=edges, weights=w_sig, alpha=0.5, normed=True, label='Signal')

            ax[ivar].set_xlabel("Jet {}".format(var),
                                horizontalalignment='right', x=1.0)

            ax[ivar].set_ylabel("Jets / {:.3f}".format(np.diff(edges)[0]),
                                horizontalalignment='right', y=1.0)
            pass

        plt.legend()
        plt.savefig(args.output + 'tagger_distributions.pdf')

        # Plotted ROCs
        fig, ax = plt.subplots(figsize=(5,5))

        ax.plot([0,1],[0,1], 'k--', linewidth=1.0, alpha=0.2)
        for ivar, var in enumerate(reversed(variables)):
            eff_sig, eff_bkg = roc_efficiencies(sig[var], bkg[var], sig['weight'], bkg['weight'])
            try:
                auc = roc_auc(eff_sig, eff_bkg)
            except: # Efficiencies not monotonically increasing
                auc = 0.
                pass
            ax.plot(eff_bkg, eff_sig, label='{} (AUC: {:.3f})'.format(var, auc))
            pass

        plt.xlabel("Background efficiency", horizontalalignment='right', x=1.0)
        plt.ylabel("Signal efficiency",     horizontalalignment='right', y=1.0)
        plt.legend()
        plt.savefig(args.output + 'tagger_ROCs.pdf')
        pass

"""
