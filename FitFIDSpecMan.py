#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit the FID and output SpecMan commands to update the tuning
"""

import numpy as np
from FitFID import fitFID

# gyrometric ratio of the electron
ABS_GAMMA = 17.60859644  # rad / (us G) 
ABS_GAMMA_2PI = ABS_GAMMA / (2 * np.pi)  # MHz / G

def main():
    """ This function is only called if this module is
    executed directly. It parses the command line
    and calls the fitFID function. It then outputs the
    results as SpecMan commands to tune the machine

    Usage: FitFIDSpecMan.py field onResFile --offRes=offResFile
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('field', type=float, help="Input field [G]")
    parser.add_argument('onRes', type=argparse.FileType('rb'),
                        help="Input tdms on-resonance file. Use '-' for stdin")
    parser.add_argument('--offRes', type=argparse.FileType('rb'),
                        help="Input tdms off-resonance file. Use '-' for stdin")

    args = parser.parse_args()

    x0, fwhm, phase = fitFID(args.onRes, args.offRes)  # units: MHz,MHz,degrees
    
    newField = args.field - x0 / ABS_GAMMA_2PI

    print(f".daemon.qs.phref='{phase:.2f}deg'")
    print(f".spec.FLD.Field={newField:.3f}")

if __name__ == "__main__":
    # execute only if run as a script
    main()
