# -*- coding: utf-8 -*-
"""
Various utilities

Note: standard units are G, K, s

Use the unitDict to convert to standard units if necessary

Created on Sat Jan 18 10:50:22 2020

@author: Margaret
@author: Jovan
"""

import re

import logging
log = logging.getLogger("SuperJIVA." + __name__)

unitDict = {
    'p': 1e-12,
    'n': 1e-9,
    'u': 1e-6,
    'm': 1e-3,
    'k': 1e3,
    'M': 1e6,
    'G': 1e9,
    'T': 1e12
}

def get_val(istr):
    """ a helper wrapper to ignore values after ; or ,
    """
    out=istr.split(';', 1)
    o2=out[0].split(',', 1)
    return str2val(o2[0])[0]


# Something to consider adding: a explicit check that the units
# are a reasonable value.
def str2val(istr):
    """
    Convert a string with a unit into a (number, unit)
    Only standard units are given; prefixes are removed
    """
    # first extract the numerical part
    match = re.search(r'([0-9\.eE\-\+]+)', istr)
    if match is None:
        raise RuntimeError("Malformed string passed to str2val")
    numString = match.group(0)
    try:
        val = int(numString)
    except ValueError:
        val = float(numString)

    unitStr = istr[match.end():].strip()  # remove the actual number
    unit = unitStr
    if len(unitStr) > 0:
        # don't require the unit to be just text because can also have G/cm, for example,
        # or G cm^-1. Can potentially get fancier, but for now assume all good.
        
        # make sure not to translate m, G, T, if it is for meters, Gauss, Tesla
        if len(unitStr) > 1 and unitStr[1] not in (' ', '/'):
            if unitStr[0] in unitDict:
                val *= unitDict[unitStr[0]]
                unit = unitStr[1:]

    return (val, unit)

