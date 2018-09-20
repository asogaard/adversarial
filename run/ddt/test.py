#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for testing DDT transform."""

# Basic import(s)
import math
from array import array

# Scientific import(s)
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Project import(s)
from adversarial.utils import parse_args, initialise, load_data, mkdir, saveclf, latex, garbage_collect
from adversarial.profile import profile, Profile
from adversarial.constants import *

# Local import(s)
from .common import *
from tests.studies.common import TemporaryStyle

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialise
    args, cfg = initialise(args)

    # Load data
    data, _, _ = load_data(args.input + 'data.h5', test=True)

    # Add Tau21DDT variable
    add_ddt(data, VAR_TAU21)

    # Load transform
    ddt = loadclf('models/ddt/ddt.pkl.gz')

    # --------------------------------------------------------------------------
    # 1D plot

    # Define variable(s)
    msk = data['signal'] == 0

    # Fill profiles
    profiles = dict()
    for var in [VAR_TAU21, VAR_TAU21 + 'DDT']:
        profiles[var] = fill_profile(data[msk], var)
        pass

    # Convert to graphs
    graphs = dict()
    for key, profile in profiles.iteritems():
        # Create arrays from profile
        arr_x, arr_y, arr_ex, arr_ey = array('d'), array('d'), array('d'), array('d')
        for ibin in range(1, profile.GetXaxis().GetNbins() + 1):
            if profile.GetBinContent(ibin) != 0. or profile.GetBinError(ibin) != 0.:
                arr_x .append(profile.GetBinCenter (ibin))
                arr_y .append(profile.GetBinContent(ibin))
                arr_ex.append(profile.GetBinWidth  (ibin) / 2.)
                arr_ey.append(profile.GetBinError  (ibin))
                pass
            pass

        # Create graph
        graphs[key] = ROOT.TGraphErrors(len(arr_x), arr_x, arr_y, arr_ex, arr_ey)
        pass

    # Plot 1D transform
    plot1D(graphs, ddt, arr_x)


    # --------------------------------------------------------------------------
    # 2D plot

    # Create contours
    binsx = np.linspace(1.5, 5.0, 40 + 1, endpoint=True)
    binsy = np.linspace(0.0, 1.4, 40 + 1, endpoint=True)

    contours = dict()
    for sig in [0,1]:

        # Get signal/background mask
        msk = data['signal'] == sig

        # Normalise jet weights
        w  = data.loc[msk, VAR_WEIGHT].values
        w /= math.fsum(w)

        # Prepare inputs
        X = data.loc[msk, [VAR_RHODDT, VAR_TAU21]].values

        # Fill, store contour
        contour = ROOT.TH2F('2d_{}'.format(sig), "", len(binsx) - 1, binsx, len(binsy) - 1, binsy)
        root_numpy.fill_hist(contour, X, weights=w)
        contours[sig] = contour
        pass

    # Linear discriminant analysis (LDA)
    lda = LinearDiscriminantAnalysis()
    X = data[[VAR_RHODDT, VAR_TAU21]].values
    y = data['signal'].values
    w = data[VAR_WEIGHT].values
    p = w / math.fsum(w)
    indices = np.random.choice(y.shape[0], size=int(1E+06), p=p, replace=True)
    lda.fit(X[indices], y[indices])  # Fit weighted sample

    # -- Linear fit to decision boundary
    xx, yy = np.meshgrid(binsx, binsy)
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    yboundary = binsy[np.argmin(np.abs(Z - 0.5), axis=0)]
    xboundary = binsx
    lda = LinearRegression()
    lda.fit(xboundary.reshape(-1,1), yboundary)

    # Plot 2D scatter
    plot2D(data, ddt, lda, contours, binsx, binsy)
    return


def plot1D (*argv):
    """
    Method for delegating 1D plotting.
    """

    # Unpack arguments
    graphs, ddt, arr_x = argv

    # Style
    ROOT.gStyle.SetTitleOffset(1.4, 'x')

    # Canvas
    c = rp.canvas(batch=True)

    # Setup
    pad = c.pads()[0]._bare()
    pad.cd()
    pad.SetTopMargin(0.10)
    pad.SetTopMargin(0.10)

    # Profiles
    c.graph(graphs[VAR_TAU21],         label="Original, #tau_{21}",          linecolor=rp.colours[4], markercolor=rp.colours[4], markerstyle=24, legend_option='PE')
    c.graph(graphs[VAR_TAU21 + 'DDT'], label="Transformed, #tau_{21}^{DDT}", linecolor=rp.colours[1], markercolor=rp.colours[1], markerstyle=20, legend_option='PE')

    # Fit
    x1, x2 = min(arr_x), max(arr_x)
    intercept, coef = ddt.intercept_ + ddt.offset_, ddt.coef_
    y1 = intercept + x1 * coef
    y2 = intercept + x2 * coef
    c.plot([y1,y2], bins=[x1,x2], color=rp.colours[-1], label='Linear fit', linewidth=1, linestyle=1, option='L')

    # Decorations
    c.xlabel("Large-#it{R} jet #rho^{DDT} = log[m^{2} / (p_{T} #times 1 GeV)]")
    c.ylabel("#LT#tau_{21}#GT, #LT#tau_{21}^{DDT}#GT")

    c.text(["#sqrt{s} = 13 TeV,  Multijets"], qualifier=QUALIFIER)
    c.legend(width=0.25, xmin=0.57, ymax=None if "Internal" in QUALIFIER else 0.85)

    c.xlim(0, 6.0)
    c.ylim(0, 1.4)
    c.latex("Fit range", sum(FIT_RANGE) / 2., 0.08, textsize=13, textcolor=ROOT.kGray + 2)
    c.xline(FIT_RANGE[0], ymax=0.82, text_align='BR', linecolor=ROOT.kGray + 2)
    c.xline(FIT_RANGE[1], ymax=0.82, text_align='BL', linecolor=ROOT.kGray + 2)

    # Save
    mkdir('figures/ddt/')
    c.save('figures/ddt/ddt.pdf')
    return


def plot2D (*argv):
    """
    Method for delegating 2D plotting.
    """

    # Unpack arguments
    data, ddt, lda, contours, binsx, binsy = argv

    with TemporaryStyle() as style:

        # Style
        style.SetNumberContours(10)

        # Canvas
        c = rp.canvas(batch=True)

        # Axes
        c.hist([binsy[0]], bins=[binsx[0], binsx[-1]], linestyle=0, linewidth=0)

        # Plotting contours
        for sig in [0,1]:
            c.hist2d(contours[sig], linecolor=rp.colours[1 + 3 * sig], label="Signal" if sig else "Background", option='CONT3', legend_option='L')
            pass

        # Linear fit
        x1, x2 = 1.5, 5.0
        intercept, coef = ddt.intercept_ + ddt.offset_, ddt.coef_
        y1 = intercept + x1 * coef
        y2 = intercept + x2 * coef
        c.plot([y1,y2], bins=[x1,x2], color=rp.colours[-1], label='DDT transform fit', linewidth=1, linestyle=1, option='L')

        # LDA decision boundary
        y1 = lda.intercept_ + x1 * lda.coef_
        y2 = lda.intercept_ + x2 * lda.coef_
        c.plot([y1,y2], bins=[x1,x2],  label='LDA boundary', linewidth=1, linestyle=2, option='L')

        # Decorations
        c.text(["#sqrt{s} = 13 TeV"], qualifier=QUALIFIER)
        c.legend()
        c.ylim(binsy[0], binsy[-1])
        c.xlabel("Large-#it{R} jet " + latex('rhoDDT', ROOT=True))
        c.ylabel("Large-#it{R} jet " + latex('Tau21',  ROOT=True))

        # Save
        mkdir('figures/ddt')
        c.save('figures/ddt/ddt_2d.pdf')
        pass
    return


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
