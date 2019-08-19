"""
This script looks at the catalog from Plourde et al. (2015) and transform
the LFE detections for each template into a time series
"""

import numpy as np
import pickle

from datetime import datetime, timedelta

def get_time_series(filename, window, tbegin, tend):
    """
    Function to transform the LFE catalog into a time series

    Input:
        type filename = string
        filename = Name of the LFE template
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
    LFEtime = np.loadtxt('detections/' + filename + '_detect5_cull.txt', \
        dtype={'names': ('unknown', 'day', 'hour', 'second', 'threshold'), \
             'formats': (np.float, '|S6', np.int, np.float, np.float)}, \
        skiprows=2)
    dt = tend - tbegin
    duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
    nw = int(duration / window)
    X = np.zeros(nw, dtype=int)
    # Loop on LFEs
    for i in range(0, np.shape(LFEtime)[0]):
        YMD = LFEtime[i][1].astype(str)
        myYear = 2000 + int(YMD[0 : 2])
        myMonth = int(YMD[2 : 4])
        myDay = int(YMD[4 : 6])
        myHour = LFEtime[i][2] - 1
        myMinute = int(LFEtime[i][3] / 60.0)
        mySecond = int(LFEtime[i][3] - 60.0 * myMinute)
        myMicrosecond = int(1000000.0 * (LFEtime[i][3] - 60.0 * myMinute \
            - mySecond))
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

    # Get the names of the template detection files
    templates = np.loadtxt('templates_list.txt', \
        dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
        'eZ', 'nb'), \
             'formats': ('S13', 'S3', np.float, np.float, np.float, \
        np.float, np.float, np.int)}, \
        skiprows=1)

    # Beginning and end of the period we are looking at
    tbegin = datetime(2008, 3, 1, 0, 0, 0)
    tend = datetime(2008, 5,  1, 0, 0, 0)

    # We construct the time series by counting the number of LFEs
    # per one-minute-long time window
    window = 60.0

    # Loop on templates
    for i in range(0, np.shape(templates)[0]):
        filename = templates[i][0].astype(str)
        X = get_time_series(filename, window, tbegin, tend)
        output = 'timeseries/{}.pkl'.format(filename)
        pickle.dump([window, tbegin, tend, X], open(output, 'wb'))
