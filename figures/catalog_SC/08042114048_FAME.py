import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from math import floor

def get_time_series(df, window, tbegin, tend):
    """
    Function to transform the LFE catalog into a time series

    Input:
        type df = panda DataFrame
        df = Times of LFE detections
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
    X = np.zeros(nw, dtype=int)
    # Loop on LFEs
    for j in range(0, len(df)):
        myYear = df['year'].iloc[j]
        myMonth = df['month'].iloc[j]
        myDay = df['day'].iloc[j]
        myHour = df['hour'].iloc[j]
        myMinute = df['minute'].iloc[j]
        mySecond = int(floor(df['second'].iloc[j]))
        myMicrosecond = int(1000000.0 * (df['second'].iloc[j] - mySecond))
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

# Plot figure
plt.figure(1, figsize=(10, 5))
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params)
window = 86400.0
path = '/Users/ariane/Documents/ResearchProject/ETSscripts/catalog/'

# Family 080421.14.048
tbegin = datetime(2007, 7, 23, 0, 0, 0)
tend = datetime(2009, 6, 13, 0, 0, 0)

# With FAME data
df = pickle.load(open(path + 'LFEs_unknown/080421.14.048/catalog_200707-200912.pkl', 'rb'))
X = get_time_series(df, window, tbegin, tend)
plt.stem(np.arange(0, len(X)), X, 'k-', markerfmt=' ', basefmt=' ')
plt.xlim([-0.5, len(X) - 0.5])
plt.ylabel('Number of LFEs', fontsize=24)
plt.title('Family 080421.14.048 ({:d} LFEs)'.format(np.sum(X)), fontsize=24)

plt.savefig('08042114048_FAME.eps', format='eps')
plt.close(1)
