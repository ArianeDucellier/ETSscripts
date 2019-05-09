"""
Functions to correlate template with data
"""

import numpy as np

from datetime import datetime, timedelta
from math import pi, sqrt

def simple(template, data):
    """
    """
    # Initialization
    M = len(template)
    N = len(data)
    cc = np.zeros(N - M + 1)
    # Normalize template
    mt = np.mean(template)
    template_dm = template - mt
    st = sqrt(np.sum(np.power(template_dm, 2.0)))
    # Loop on samples
    for i in range(0, N - M + 1):
        subdata = data[i : (M + i)]
        # Normalize data subset
        md = np.mean(subdata)
        subdata_dm = subdata - md
        sd = sqrt(np.sum(np.power(subdata_dm, 2.0)))
        # Cross correlation
        cc[i] = np.sum(template_dm * subdata_dm) / (st * sd)
    return cc

def optimized(template, data):
    """
    """
    # Initialization
    M = len(template)
    N = len(data)
    cc = np.zeros(N - M + 1)
    # Normalize template
    mt = np.mean(template)
    template_dm = template - mt
    st = np.sum(np.power(template, 2.0))
    # First data subset
    subdata = data[0 : M]
    md = np.mean(subdata)
    # Loop on samples
    for i in range(0, N - M + 1):
        subdata = data[i : (M + i)]
        # Normalize data subset
        if (i > 0):
            md = md + (data[i + M - 1] - data[i - 1]) / M
        subdata_dm = subdata - md
        sd = np.sum(np.power(subdata_dm, 2.0))
        # Cross correlation
        cc[i] = np.sum(template_dm * subdata_dm) / sqrt(st * sd)
    return cc

if __name__ == '__main__':

    # Define template and data
    N = 1200
    t = np.arange(0, N)
    template = np.sin(2.0 * pi * t / N)
    data = np.tile(template, 60)

    # Check if we get the same result
    cc1 = simple(template, data)
    cc2 = optimized(template, data)
    diff = np.sum(np.abs(cc1 - cc2))
    print('Difference = {}'.format(diff))

    # Number of computations
    NC = 10

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
