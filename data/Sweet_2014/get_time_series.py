"""
This script looks at the catalog from Sweet (2014) and
transform the LFE detections for each template into a time series
"""

import numpy as np
import pickle

from datetime import datetime, timedelta
from scipy.io import loadmat

from date import matlab2ymdhms

def get_time_series(n, window, tbegin, tend):
    """
    Function to transform the LFE catalog into a time series

    Input:
        type n = int
        n = Index of the LFE template
        type window = float
        window = Duration of the time window where we count the number of LFEs
                 (in seconds)
        type tbegin = obspy UTCDateTime
        tbegin = Beginning time of the catalog 
        type tend = obspy UTCDateTime
        tend = End time of the catalog
    Output:
        type X = numpy array
        X = Time series with number of LFEs per time window
    """
    # Get the time of LFE detections
    data = loadmat('catalogs/LFE' + str(n) + 'catalog.mat')
    if (n <= 4):
        LFEtime = data['peakTimes'][0]
    else:
        LFEtime = data['peakTimes']
    dt = tend - tbegin
    duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
    nw = int(duration / window)
    X = np.zeros(nw, dtype=int)
    # Loop on LFEs
    for i in range(0, np.shape(LFEtime)[0]):
        if (n <= 4):
            (myYear, myMonth, myDay, myHour, myMinute, mySecond, \
                 myMicrosecond) = matlab2ymdhms(LFEtime[i], False)
        else:
            (myYear, myMonth, myDay, myHour, myMinute, mySecond, \
                 myMicrosecond) = matlab2ymdhms(LFEtime[i][0], False)
        t = datetime(myYear, myMonth, myDay, myHour, myMinute, mySecond, \
            myMicrosecond)
        # Add LFE to appropriate time window
        if ((tbegin <= t) and (t < tbegin + timedelta(seconds=nw * window))):
            dt = t - tbegin
            duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * \
                0.000001
            index = int(duration / window)
            X[index] = X[index] + 1
    return X

if __name__ == '__main__':

    # Number of LFE families
    nf = 9

    # Beginning and end of the period we are looking at
    tbegin = datetime(2006, 10, 1, 0, 0, 0)
    tend = datetime(2011, 10, 1, 0, 0, 0)

    # We construct the time series by counting the number of LFEs
    # per one-minute-long time window
    window = 60.0

    # Loop on templates
    for n in range(0, nf):
        X = get_time_series(n + 1, window, tbegin, tend)
        filename = 'LFE' + str(n + 1)
        output = 'timeseries/{}.pkl'.format(filename)
        pickle.dump([window, tbegin, tend, X], open(output, 'wb'))
