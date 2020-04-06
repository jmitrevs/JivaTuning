#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Some functions to read a TDMS file containinng a FID
and perform a fit of the FFT.
"""

import numpy as np
import scipy.optimize
from nptdms import TdmsFile
from tdms_readraw import tdms_readraw

import logging
log = logging.getLogger("SuperJIVA." + __name__)

# the number of points to not look at in the beginning
NUM_CUT = 106

# when doing the fit only look at the range center-FIT_HWIDTH:center+FIT_HWIDTH
FIT_HWIDTH = 10  

# complex function, 
def epr_Lorentzian(x, x0, fwhm, scale):
    """ x, scale complex: x0, fwhm real, return complex
    """
    xc = 2*(x-x0)/fwhm
    return scale * 2/np.pi/fwhm *(1-1j*xc)/(1+xc*xc)

# expanded 
def epr_Lorentzian_exp(x, x0, fwhm, scaler, scalei):
    scale = scaler + 1j*scalei
    xc = 2*(x-x0)/fwhm
    return scale * 2/np.pi/fwhm *(1-1j*xc)/(1+xc*xc)

def func_wrap(f, x, y):
    def func(params):
        return f(x, *params) - y
    return func

def func_wrap_abs(f, x, y):
    def func(params):
        return np.abs(f(x, *params) - y)
    return func


def fitFID(onRes, offRes=None):
    """ 
    Given the onRes and offRes files (offRes is optional)

    If an off-resolution file is included, it is subtracted 
    from the on-resolution.

    Returns (x0, fwhm, phase) in (MHz, MHz, degrees) of the Lorentzian fit
    """

    onFile = TdmsFile(onRes)

    # parse the files
    ax_on, spec_on, desc_on = tdms_readraw(onFile)

    times = ax_on['x']
    on = spec_on[:,0,0,0]
    subtracted = on
    
    if offRes:
        offFile = TdmsFile(offRes)
        ax_off, spec_off, desc_off = tdms_readraw(offFile)
        off = spec_off[:,0,0,0]
        subtracted -= off
    
    sampleSpacing = times[1] - times[0]
    
    # t = times[NUM_CUT:] * 1e6  # make us  (useful for plotting)
    data = subtracted[NUM_CUT:]

    dataFFT = np.fft.fft(data, n=2048)
    freq = np.fft.fftfreq(dataFFT.size, sampleSpacing) * 1e-6 # make MHz

    # let's recenter at zero
    freq = np.fft.fftshift(freq)
    dataFFT = np.fft.fftshift(dataFFT)
    
    # determine the absolute max value
    amax = np.argmax(np.abs(dataFFT))

    fit, fitErr = scipy.optimize.leastsq(func_wrap_abs(epr_Lorentzian_exp,
                                           freq[amax-FIT_HWIDTH:amax+FIT_HWIDTH],
                                           dataFFT[amax-FIT_HWIDTH:amax+FIT_HWIDTH]),
                                     (1, 1, 1, 1))

    x0 = fit[0]
    fwhm = fit[1]
    scale = fit[2] + 1j * fit[3]
    return (x0, fwhm, np.degrees(np.angle(scale)))

def main():
    """ This function is only called if this module is
    executed directly. It parses the command line
    and calls the fitFID function.

    Usage: FitFID.py onResFile --offRes=offResFile

    Returns x0,fwhm,phase in MHz,MHz,degrees of the Lorentzian fit
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('onRes', type=argparse.FileType('rb'),
                        help="Input tdms on-resonance file. Use '-' for stdin")
    parser.add_argument('--offRes', type=argparse.FileType('rb'),
                        help="Input tdms off-resonance file. Use '-' for stdin")

    args = parser.parse_args()

    x0, fwhm, phase = fitFID(args.onRes, args.offRes)
    print(f"{x0:.4f},{fwhm:.4f},{phase:.2f}") 

if __name__ == "__main__":
    # execute only if run as a script
    main()
