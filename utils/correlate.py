"""
Functions to cross correlate template with data
"""

import numpy as np

from datetime import datetime, timedelta
from math import pi, sqrt

def simple(template, data):
    """
    Simple version but heavy computing time

    Input:
        type template = 1D Numpy array
        template = Waveform template
        type data = 1D Numpy array
        data = Time series (much) longer than the template
    Output:
        type cc = 1D Numpy array
        cc = Cross correlation value for all possible time lags
    """
    # Initialization
    M = len(template)
    N = len(data)
    cc = np.zeros(N - M + 1)
    # Normalize template
    mt = np.mean(template)
    st = sqrt(M * np.sum(np.power(template - mt, 2.0)))
    # Loop on samples
    for i in range(0, N - M + 1):
        subdata = data[i : (M + i)]
        # Normalize data subset
        md = np.mean(subdata)
        sd = sqrt(M * np.sum(np.power(subdata - md, 2.0)))
        # Cross correlation
        cc[i] = M * np.sum((template - mt) * (subdata - md)) / (st * sd)
    return cc

def optimized(template, data):
    """
    Faster version with optimized computing time

    Input:
        type template = 1D Numpy array
        template = Waveform template
        type data = 1D Numpy array
        data = Time series (much) longer than the template
    Output:
        type cc = 1D Numpy array
        cc = Cross correlation value for all possible time lags
    """
    # Initialization
    M = len(template)
    N = len(data)
    # Optimized cross-correlation
    mt = np.mean(template)
    st = sqrt(M * np.sum(np.power(template - mt, 2.0)))
    I = np.arange(0, N - M + 1)
    xs = np.insert(np.cumsum(data), 0, 0)
    xsum = xs[I + M] - xs[I]
    x2s = np.insert(np.cumsum(np.power(data, 2.0)), 0, 0)
    x2sum = x2s[I + M] - x2s[I]
    sd = np.sqrt(M * x2sum - xsum * xsum)
    cc = (M * np.correlate(data, template) - np.sum(template) * xsum) / (st * sd)
    return cc

if __name__ == '__main__':

    # Define template and data
    N = 1200
    t = np.arange(0, N)
    template = 0.5 * (1.0 - np.sin(2.0 * pi * t / N))
    data = np.tile(template, 60)

    # Check if we get the same result
    cc1 = simple(template, data)
    cc2 = optimized(template, data)
    diff = np.sum(np.abs(cc1 - cc2))
    print('Difference = {}'.format(diff))

    # Number of computations
    NC = 100

    # Run simple correlation
    t1 = datetime.now()
    for i in range(0, NC):
        cc = simple(template, data)
    t2 = datetime.now()
    dt = t2 - t1
    print('Time for simple correlation = {}'.format(dt.total_seconds()))

    # Run optimized correlation
    t1 = datetime.now()
    for i in range(0, NC):
        cc = optimized(template, data)
    t2 = datetime.now()
    dt = t2 - t1
    print('Time for optimized correlation = {}'.format(dt.total_seconds()))
