#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Standard import(s)
import ROOT
import root_numpy

# Project import(s)
from adversarial.utils import parse_args, load_data, latex
from adversarial.utils import MASSBINS, wpercentile
from .studies.common import *

# Custom import(s)
import rootplotting as rp


# Main function definition
def main (args):

    # ...

    # Load data
    data_, features, _ = load_data(args.input + 'data.h5', train=True)

    for pt_bin in [(200., 500.), (500., 1000.)]:

        # Impose pT-cut
        data = data_[(data_['pt'] >= pt_bin[0]) & (data_['pt'] < pt_bin[1])]

        var = 'Tau21'
        msk_sig = (data['signal'] == 1)
        x = data[var].values
        m = data['m'].values
        w = data['weight_test'].values


        # Get cut value
        cut = wpercentile(x[msk_sig], 50., weights=w)
        print "Cut value: {:.2f}".format(cut)

        # Discard signal
        x = x[~msk_sig]
        m = m[~msk_sig]
        w = w[~msk_sig]

        # Get pass mask
        msk_pass = x < cut
        print "Background efficiency: {:.1f}%".format(100. * w[msk_pass].sum() / w.sum())

        # Canvas
        offset = 0.06
        margin = 0.3
        # @NOTE
        #   A = Height of pad 0
        #   B = Height of pads 1,2
        #   C = Height of pad 3
        # -->
        #   A = 0.5
        #
        #   (1. - 2 * offset) * B = (1. - 2*offset - margin) * C
        #   ==>
        #   B = C * (1. - 2*offset - margin) / (1. - 2 * offset)
        #   ==>
        #   B = C * (1 - margin / (1. - 2 * offset))
        #
        #   A + 2 * B + C = 1
        #   ==>
        #   A + 2 * C * (1 - margin / (1. - 2 * offset)) + C = 1
        #   ==>
        #   C = (1 - A) / (1 + 2 * (1 - margin / (1. - 2 * offset)))

        A = 0.5
        C = (1 - A) / (1 + 2 * (1 - margin / (1. - 2 * offset)))
        B = C * (1 - margin / (1. - 2 * offset))

        c = rp.canvas(batch=True, num_pads=4, fraction=(A, B, B, C), size=(600, 700))

        # Set pad margins
        c.pad(0)._bare().SetBottomMargin(offset)
        c.pad(1)._bare().SetTopMargin   (offset)
        c.pad(1)._bare().SetBottomMargin(offset)
        c.pad(2)._bare().SetTopMargin   (offset)
        c.pad(2)._bare().SetBottomMargin(offset)
        c.pad(3)._bare().SetTopMargin   (offset)
        c.pad(3)._bare().SetBottomMargin(offset + margin)

        # Styling
        HISTSTYLE[True] ['label'] = 'Passing cut, #it{{P}}'.format(latex(var, ROOT=True))
        HISTSTYLE[False]['label'] = 'Failing cut, #it{{F}}'.format(latex(var, ROOT=True))

        # Histograms
        F = c.hist(m[~msk_pass], bins=MASSBINS, weights=w[~msk_pass], normalise=True, **HISTSTYLE[False])
        P = c.hist(m[ msk_pass], bins=MASSBINS, weights=w[ msk_pass], normalise=True, **HISTSTYLE[True])

        P, F = map(root_numpy.hist2array, [P,F])
        M = (P + F) / 2
        c.hist(M, bins=MASSBINS, normalise=True, linewidth=3, linecolor=ROOT.kViolet, linestyle=2, label='Average, #it{M}')


        # Compute divergences
        KL_PM = - P * np.log2(M / P)
        KL_FM = - F * np.log2(M / F)
        JSD    = (KL_PM + KL_FM) / 2.
        JSDsum = np.cumsum(JSD)

        opts  = dict(bins=MASSBINS, fillcolor=ROOT.kGray, alpha=0.5)

        # Draw divergences
        c.pad(1).hist(KL_PM, **opts)
        c.pad(1).ylim(-0.12, 0.05)
        c.pad(1).yline(0.)

        c.pad(2).hist(KL_FM, **opts)
        c.pad(2).ylim(-0.05, 0.12)
        c.pad(2).yline(0.)

        c.pad(3).hist(JSD, **opts)
        c.pad(3).ylim(0., 0.03)
        c.pad(3).yline(0.)

        o = rp.overlay(c.pad(3), color=ROOT.kViolet, ndiv=502)
        o.hist(JSDsum, bins=MASSBINS, linecolor=ROOT.kViolet)
        o.label("#sum_{i #leq n} JSD(P #parallel F)")
        o.lim(0, 0.2)
        #o._update_overlay()

        # Styling axes
        c.pad(0)._xaxis().SetTitleOffset(999.)
        c.pad(1)._xaxis().SetTitleOffset(999.)
        c.pad(2)._xaxis().SetTitleOffset(999.)
        c.pad(3)._xaxis().SetTitleOffset(5.)
        c.pad(0)._xaxis().SetLabelOffset(999.)
        c.pad(1)._xaxis().SetLabelOffset(999.)
        c.pad(2)._xaxis().SetLabelOffset(999.)

        c.pad(0)._yaxis().SetNdivisions(505)
        c.pad(1)._yaxis().SetNdivisions(502)
        c.pad(2)._yaxis().SetNdivisions(502)
        c.pad(3)._yaxis().SetNdivisions(502)

        c.pad(0).ylim(0, 0.20)
        c.pad(0).cd()
        c.pad(0)._get_first_primitive().Draw('SAME AXIS')

        # Decorations
        c.text(TEXT + [
            "Multijets, training dataset",
            "Cut on {:s} at #varepsilon_{{sig}}^{{rel}} = 50%".format(latex(var, ROOT=True)),
            "p_{{T}} #in  [{:.0f}, {:.0f}] GeV".format(*pt_bin)
        ], qualifier='Simulation Internal')
        c.legend(width=0.25)
        c.xlabel("Large-#it{R} jet mass [GeV]")
        c.ylabel("Fraction of jets")
        c.pad(1).ylabel('KL(P #parallel M)')
        c.pad(2).ylabel('KL(F #parallel M)')
        c.pad(3).ylabel('JSD(P #parallel F)')

        # Save
        c.save('figures/massdecorrelationmetric_{:s}__pT{:.0f}_{:.0f}GeV.pdf'.format(var, *pt_bin))
        pass
    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args(backend=True, plots=True)

    # Call main function
    main(args)
    pass
