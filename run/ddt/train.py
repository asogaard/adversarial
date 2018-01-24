#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script for performing DDT study."""

# Scientific import(s)
import ROOT
import numpy as np
import pandas as pd
import root_numpy
from array import array

# Project import(s)
from adversarial.profile import profile, Profile
from adversarial.new_utils import parse_args, initialise, load_data, mkdir
from adversarial.constants import *

# Custom import(s)
import rootplotting as rp


# Main function definition
@profile
def main (args):

    # Initialising
    # --------------------------------------------------------------------------
    args, cfg = initialise(args)


    # Loading data
    # --------------------------------------------------------------------------
    data, features, _ = load_data(args.input + 'data.h5')
    data = data[data['train'] == 1]


    # Common definition
    # --------------------------------------------------------------------------
    bins = np.linspace(-1, 6, 7 * 8 + 1, endpoint=True)  # Binning in rhoDDT
    fit_range = (1.5, 4.0)  # Range in rhoDDT to be fitted
    msk = (data['signal'] == 0)  # Background mask
    profiles, graphs = dict(), dict()


    # Perform study
    # --------------------------------------------------------------------------
    with Profile("Study: DDT"):

        # Adding rhoDDT variable
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Adding rhoDDT variable"):
            data['rhoDDT'] = pd.Series(np.log(np.square(data['m'])/(data['pt'] * 1.)), index=data.index)
            pass


        # Filling Tau21 profile
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Filling Tau21 profile"):
            profiles['Tau21'] = ROOT.TProfile('profile_Tau21', "", len(bins) - 1, bins)
            root_numpy.fill_profile(profiles['Tau21'], data.loc[msk, ['rhoDDT', 'Tau21']].as_matrix(), weights=data.loc[msk, 'weight'].as_matrix().flatten())
            pass


        # Fitting profile
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Fitting profile"):
            fit = ROOT.TF1('fit', 'pol1', *fit_range)
            profiles['Tau21'].Fit('fit', 'RQ0')
            intercept, slope = fit.GetParameter(0), fit.GetParameter(1)
            print "Fitted function:"
            print "  intercept: {:f}".format(intercept)
            print "  slope:     {:f}".format(slope)
            pass


        # Adding Tau21DDT variable
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Adding Tau21DDT variable"):
            data['Tau21DDT'] = pd.Series(data['Tau21'] - slope * (data['rhoDDT'] - fit_range[0]), index=data.index)
            pass


        # Filling Tau21DDT profile
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Filling Tau21DDT profile"):
            profiles['Tau21DDT'] = ROOT.TProfile('profile_Tau21DDT', "", len(bins) - 1, bins)
            root_numpy.fill_profile(profiles['Tau21DDT'], data.loc[msk, ['rhoDDT', 'Tau21DDT']].as_matrix(), weights=data.loc[msk, 'weight'].as_matrix().flatten())
            pass


        # Convert to graphs
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Convert to graphs"):

            # Loop profiles
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
            pass


        # Creating figure
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        with Profile("Creating figure"):

            # Canvas
            c = rp.canvas(batch=True)

            # Profiles
            c.graph(graphs['Tau21'],    label="Original, #tau_{21}",          linecolor=rp.colours[5], markercolor=rp.colours[5])
            c.graph(graphs['Tau21DDT'], label="Transformed, #tau_{21}^{DDT}", linecolor=rp.colours[1], markercolor=rp.colours[1], markerstyle=21)

            # Fit
            x1, x2 = min(arr_x), max(arr_x)
            #x1, x2 = fit_range
            y1 = intercept + x1 * slope
            y2 = intercept + x2 * slope
            c.plot([y1,y2], bins=[x1,x2], color=rp.colours[-1], label='Linear fit', linewidth=1, linestyle=1, option='L')

            # Decorations
            c.xlabel("Large-#it{R} jet #rho^{DDT} = log(m_{calo}^{2}/ p_{T} / 1 GeV)")
            c.ylabel("#LT#tau_{21}#GT, #LT#tau_{21}^{DDT}#GT")
            c.text(["#sqrt{s} = 13 TeV,  QCD jets",
                    "Training dataset",
                    "Baseline selection",
                    ],
                qualifier=QUALIFIER)
            c.legend()
            c.ylim(0, 1.4)
            c.xline(fit_range[0], text='Fit range', ymax=0.82, text_align='BR')
            c.xline(fit_range[1], text='Fit range', ymax=0.82, text_align='BL')

            # Save
            mkdir('figures/')
            c.save('figures/ddt.pdf')
            pass

        pass

    return 0


# Main function call
if __name__ == '__main__':

    # Parse command-line arguments
    args = parse_args()

    # Call main function
    main(args)
    pass
