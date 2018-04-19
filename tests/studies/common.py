#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Scientific import(s)
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import roc_curve

# ROOT import(s)
import ROOT

# Project imports
from adversarial.utils import mkdir, garbage_collect

# Custom import(s)
import rootplotting as rp
rp.colours.pop(3)  # To remove unused colour from list

# Global variable definition(s)
HISTSTYLE = {  # key = signal / passing
    True: {
        'fillcolor': rp.colours[4],
        'linecolor': rp.colours[4],
        'fillstyle': 3354,
        'alpha': 0.5,
        },
    False: {
        'fillcolor': rp.colours[1],
        'linecolor': rp.colours[1],
        'fillstyle': 1001,
        'alpha': 0.5,
        }
    }

TEXT = ["#sqrt{s} = 13 TeV",
        #"Baseline selection",
        ]

# Global ROOT TStyle settings
ROOT.gStyle.SetHatchesLineWidth(3)
ROOT.gStyle.SetTitleOffset(1.6, 'y')


class TemporaryStyle ():
    """
    Context manager to temporarily modify global ROOT style settings.
    """

    def __init__ (self):
        self.rstyle = ROOT.gROOT.GetStyle(map(lambda ts: ts.GetName(), ROOT.gROOT.GetListOfStyles())[-1])
        return

    def __enter__ (self):
        style = ROOT.TStyle(self.rstyle)
        style.cd()
        return style

    def __exit__ (self, *args):
        self.rstyle.cd()
        return
    pass


def showsave (f):
    """
    Method decorrator for all study method, to (optionally) show and save the
    canvas to file.
    """
    def wrapper (*args, **kwargs):
        # Run study
        c, args, path = f(*args, **kwargs)

        # Save
        if args.save:
            dir = '/'.join(path.split('/')[:-1])
            mkdir(dir)
            c.save(path)
            pass

        # Show
        if args.show:
            c.show()
            pass
        return

    return wrapper


def standardise (name):
    """Method to standardise a given filename, and remove special characters."""
    return name.lower() \
           .replace('#', '') \
           .replace('minus', '') \
           .replace(',', '__') \
           .replace('.', 'p') \
           .replace('(', '__') \
           .replace(')', '') \
           .replace('%', '') \
           .replace('=', '_')
