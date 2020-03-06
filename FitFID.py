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


# The fitting functions
def Lorentzian(x, mean, scale, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return scale * gamma / np.pi / ((x-mean)**2 + gamma**2)

# complex
def epr_Lorentzian(x, x0, scale, fwhm):
    xc = 2*(x-x0)/fwhm
    return scale * 2/np.pi/fwhm *(1-1j*xc)/(1+xc*xc)

# expanded 
def epr_Lorentzian_exp(x, x0r, x0i, scaler, scalei, fwhmr):
    x0 = x0r + 1j*x0i
    scale = scaler + 1j*scalei
    fwhm = fwhmr
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

    Returns (mean, fwhm, phase) in (MHz, MHz, radians) of the Lorentzian fit
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

    # the phase at the maximum
    phase = np.angle(dataFFT[amax])

    # phase rotate so that the maximum is real
    dataFFTRot = dataFFT * np.exp(-1j * phase)

    
    
    # fit, fitErr = scipy.optimize.curve_fit(Lorentzian,
    #                                        freq[amax-FIT_HWIDTH:amax+FIT_HWIDTH],
    #                                        dataFFTRot[amax-FIT_HWIDTH:amax+FIT_HWIDTH].real)

    # fit, fitErr = scipy.optimize.leastsq(func_wrap_abs(Lorentzian,
    #                                        freq[amax-FIT_HWIDTH:amax+FIT_HWIDTH],
    #                                        dataFFTRot[amax-FIT_HWIDTH:amax+FIT_HWIDTH].real),
    #                                  (1, 1, 1))

    #print(fit, phase)
    #return (fit[0], fit[2], phase)
    
    fit, fitErr = scipy.optimize.leastsq(func_wrap_abs(epr_Lorentzian_exp,
                                           freq[amax-FIT_HWIDTH:amax+FIT_HWIDTH],
                                           dataFFT[amax-FIT_HWIDTH:amax+FIT_HWIDTH]),
                                     (1, 1, 1, 1, 1))

    #print(fit)
    x0 = fit[0] + 1j*fit[1]
    scale = fit[2] + 1j * fit[3]
    fwhm = np.abs(fit[4])
    return (np.abs(x0), fwhm, np.angle(scale))

def main():
    """ This function is only called if this module is
    executed directly. It parses the command line
    and calls the controlApp function.

    Usage: FitFID.py --onRes=file --offRes=file
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--echo", help="Echo responses to stdout", action="store_true")
    parser.add_argument('onRes', type=argparse.FileType('rb'),
                        help="Input tdms on-resonance file. Use '-' for stdin")
    parser.add_argument('--offRes', type=argparse.FileType('rb'),
                        help="Input tdms off-resonance file. Use '-' for stdin")

    args = parser.parse_args()

    mean, fwhm, phase = fitFID(args.onRes, args.offRes)
    print(f"mean = {mean:.2f} MHz, FWHM = {fwhm:.2f} MHz, phase = {phase:.2f} rad") 

if __name__ == "__main__":
    # execute only if run as a script
    main()
