#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Common, text-related utilities."""

# Basic import(s)
import re


def belongs_to (x, module):
    """Return whether `x` belongs to `module` as determined by typename."""
    return module.__name__ in type(x).__module__


def snake_case (string):
    """ ... """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def latex (variable_name, ROOT=True):
    """LaTeX-format variable nameself.

    Arguments:
        variable_name: Input name of variable.
        ROOT: Whether to use ROOT-style syntax instead of standard LaTeX syntax.

    Returns:
        Latex-formatted variable name.
    """

    name = variable_name.lower()

    name = re.sub('^d2', 'D_{2}', name)
    name = re.sub('^n2', 'N_{2}', name)
    name = re.sub('^pt$', 'p_{T}', name)
    #name = re.sub('rho', '\\rho', name)
    name = name.replace('rho', '\\rho')
    name = name.replace('tau21', '\\tau_{21}')
    name = name.replace('ddt', '^{DDT}')
    name = name.replace('css', '^{CSS}')
    name = re.sub('\_([0-9]+)$', '^{(\\1)}', name)
    name = re.sub('-k(.*)nn(.*)$', '^{#it{k}\\1NN\\2}', name)

    # ML taggers
    if 'boost' in name or re.search('nn$', name) or re.search('^nn', name) or 'ann' in name:
        name = '\\textit{z}_{%s}' % variable_name
        pass

    name = re.sub('(\(.*\))([}]*)$', '\\2^{\\1}', name)

    # Remove duplicate superscripts
    name = re.sub("(\^.*)}\^{", "\\1", name)

    if name == variable_name.lower():
        name = variable_name
        pass

    if ROOT:
        return name.replace('\\', '#').replace('textit', 'it')
        pass
    return r"${}$".format(name)
