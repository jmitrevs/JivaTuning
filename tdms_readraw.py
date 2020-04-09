#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Read raw data file and return ax, data&baseline, dsc

Note: standard units are G, K, s

Created on Sun Dec 15 09:22:16 2019

@author: Margaret
@author: Jovan

"""


from utils_misc import str2val, get_val
import numpy as np
import logging
log = logging.getLogger("SuperJIVA." + __name__)


def tdms_stream2array(indata, d1, d2, d3, d4):
    """
    convert a 1D array from the TDMS file and output a 4D
    numpy array with the right dimensions. Currently only support
    a TDMS file with only a single copy of the data--no averaging
    """

    # check to make sure it's not malformed
    max_dim = np.array([d1, d2, d3, d4]).astype(float)
    total_data = np.prod(max_dim)
    if total_data != indata.size:
        raise NotImplementedError("A TDMS file with only a single copy of the data is currently supported")

    # return the numpy array
    return np.reshape(indata, [d1, d2, d3, d4], order='F')


def SpecManTDMSpar(tdmsFile):
    """
    This function parses the metadata (properties) in a TdmsFile
    input: a TdmsFile
    output: (axes: dict, desc: dict)
       The axes dictionary provides more information on the sweep axes.
       The desc dictionary is a more direct copy of the properties from
          the input tdms, though the range variables are expanded, and
          a version with "label" is added to specify the units.
    """

    # the output dictionaries
    axes = {}
    desc = {}

    root_properties = tdmsFile.properties
    axes['title'] = root_properties.get('name', '?')

    # We should put the root properties in desc
    for name, value in root_properties.items():
        desc[name] = value

    # parse the sweep axes
    sweepax = []
    shots = 1

    # Let's parse the basic axis properties.

    # The properties should be ordered, but just in case, let's create the iterKeys
    # explicitly to enforce the order

    axes_properties = tdmsFile['axis'].properties
    params = tdmsFile['params'].properties
    aquisition = tdmsFile['aquisition'].properties

    # iterate over the sweep axes
    iterKeys = ["transient"]
    i = 0
    while "sweep" + str(i) in axes_properties:
        iterKeys.append("sweep" + str(i))
        i += 1

    for key in iterKeys:
        value = axes_properties[key].split(',')
        ax = {}
        ax['t'] = value[0].strip()[0]  # just take the first character
        ax['dim'] = int(value[1])
        if ax['t'] in ('S', 'I', 'A', 'R'):
            ax['size'] = 1
        else:
            ax['size'] = ax['dim']
        ax['reps'] = int(value[2])
        if ax['t'] == 'S':
            shots *= ax['dim'] * ax['reps']
        else:
            shots *= ax['reps']

        ax['var'] = value[3:]

        # let's parse the values.
        for var in ax['var']:
            tempparam = var.replace(' ', '_')
            if tempparam in params:
                desc["params_" + tempparam] = parse_range(params[tempparam], ax['dim'])

        sweepax.append(ax)

    stream_properties = tdmsFile['streams']['Re'].properties
    triggers = stream_properties.get('triggers', 1)

    sweepax[0]['size'] = sweepax[0]['size'] * triggers
    axes['sweepax'] = sweepax
    axes['shots'] = shots

    axislabel = 'xyzt'
    counter = 0
    for axis in sweepax:
        if axis['size'] > 1:
            log.debug(f"axis = {axis}")
            if axis['t'] == 'I' or axis['t'] == 'A':
                arr, unit = parse_range('1sl step 1sl;', axis['size'])
            elif axis['t'] == 'T':
                dwell_time = stream_properties.get('dwelltime', 1E-9)
                arr, unit = parse_range(f"0 s step {dwell_time} s", axis['size'])
            else:
                tempparam = axis['var'][0].replace(' ', '_')
                if tempparam in params:
                    arr, unit = desc["params_" + axis['var'][0]]
                elif tempparam in aquisition:
                    log.debug("This path of execution might need a bit more checking")
                    strr = aquisition[tempparam]
                    log.warning(f"aquisition variable {tempparam} has value {strr}, value not used")
                    arr = np.arange(axis['size'])
                    arr = arr.reshape([-1, 1], order='F')
                    unit = 's'
                else:
                    raise RuntimeError("found an unexpected paremeter in file")

            if len(arr) > axis['size']:
                raise RuntimeError("I don't think this should ever happen")
                arr = arr[0:axis['size']]

            axl = axislabel[counter]
            ax2 = axl+'label'
            axes[axl] = arr
            axes[ax2] = axis['var'][0] + ", " + unit
            counter = counter + 1

    axes['StartTime'] = root_properties['starttime']
    axes['FinishTime'] = root_properties['finishtime']
    axes['ExpTime'] = root_properties['totaltime']
    axes['RepTime'] = get_val(params.get('RepTime', '15 us'))

    # add a few more values to the desc
    for grp in ['exp_info', 'sample_info']:
        for name, value in tdmsFile[grp].properties.items():
            newkey = grp + '_' + name
            desc[newkey] = value

    return axes, desc


def parse_range(line, asize):
    """
    This takes a text description of a range and the array size,
    returning the pair (the numpy array, units of array)
    """
    # first remove the stuff after the ;
    line = line.split(';', 1)[0]
    if "step" in line:
        minval, stepval = [str2val(x) for x in line.split('step', 1)]
        arr = np.arange(float(asize))
        return (arr*stepval[0] + minval[0], minval[1])
    elif "logto" in line:
        minval, maxval = [str2val(x) for x in line.split('logto', 1)]
        return (np.logspace(np.log10(minval[0]), np.log10(maxval[0]), asize), minval[1])
    elif "to" in line:
        minval, stepval = [str2val(x)[0] for x in line.split('logto', 1)]
        return (np.linspace(minval[0], maxval[0], asize), minval[1])
    else:
        # string of the type 10ns, 20ns, 30ns;
        stringarray = line.split(',')
        return (np.array([str2val(x)[0] for x in stringarray]), str2val(stringarray[0])[1])


def tdms_readraw(fff):
    """
    input: a TdmsFile object from a raw, SpecMan tdms file
    ouput: (axis information: dictionary, data: np.array, description: dictionary) tuple
    """

    # get the information about the axes and other properties
    ax, desc = SpecManTDMSpar(fff)

    #  add real data for reference stream to exp
    stream_properties = fff['streams']['Re'].properties
    dim1 = stream_properties['dim1']
    dim2 = stream_properties['dim2']
    dim3 = stream_properties['dim3']
    dim4 = stream_properties['dim4']

    sreal = tdms_stream2array(fff['streams']['Re'].data, dim1, dim2, dim3, dim4)
    simag = tdms_stream2array(fff['streams']['Im'].data, dim1, dim2, dim3, dim4)

    spec = sreal + 1j * simag

    return ax, spec, desc
