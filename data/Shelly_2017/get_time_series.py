"""
This script looks at the catalog from Shelly (2017) and transform
the LFE detections for each family into a time series
"""

import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from math import floor

def get_time_series(LFEtime, filename, window, tbegin, tend):
    """
    Function to transform the LFE catalog into a time series

    Input:
        type LFEtime = panda DataFrame
        LFEtime = Times of LFE detections
        type filename = string
        filename = Name of the LFE family
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
    # Length of the time series
    dt = tend - tbegin
    duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
    nw = int(duration / window)
    # get the data for the family
    data = LFEtime.loc[LFEtime['ID'] == filename]
    X = np.zeros(nw, dtype=int)
    # Loop on LFEs
    for j in range(0, data.shape[0]):
        myYear = data['year'].iloc[j]
        myMonth = data['month'].iloc[j]
        myDay = data['day'].iloc[j]
        myHour = data['hr'].iloc[j]
        myMinute = data['min'].iloc[j]
        mySecond = int(floor(data['sec'].iloc[j]))
        myMicrosecond = int(1000000.0 * (data['sec'].iloc[j] - mySecond))
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

    # Read the LFE file
    LFEtime = pd.read_csv('jgrb52060-sup-0002-datas1.txt', \
        delim_whitespace=True, header=None, skiprows=2)
    LFEtime.columns = ['year', 'month', 'day', 's_of_day', 'hr', 'min', \
        'sec', 'ccsum', 'meancc', 'med_cc', 'seqday', 'ID', 'latitude', \
        'longitude', 'depth', 'n_chan']
    LFEtime['ID'] = LFEtime.ID.astype('category')
    families = LFEtime['ID'].cat.categories.tolist()

    # Beginning and end of the period we are looking at
    tbegin = datetime(2001, 4, 6, 0, 0, 0)
    tend = datetime(2016, 9, 20, 0, 0, 0)

    # We construct the time series by counting the number of LFEs
    # per one-day-long time window
    window = 60.0

    # Loop on LFE families
    for i in range(0, len(families)):
        X = get_time_series(LFEtime, families[i], window, tbegin, tend)
        output = 'timeseries/{}.pkl'.format(families[i])
        pickle.dump([window, tbegin, tend, X], open(output, 'wb'))
