"""
This script looks at the catalog from southern Cascadia (2007-2009) and plots
the hourly number of LFEs in function of time in function of latitude
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from math import floor

params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)

# List of LFE families
templates = np.loadtxt('../Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Beginning and end of the period we are looking at
tbegin = datetime(2007, 7, 1, 0, 0, 0)
tend = datetime(2009, 7, 1, 0, 0, 0)

# We construct the time series by counting the number of LFEs
# per one-day-long time window
window = 86400.0

# Length of the time series
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)

# Amplitude coefficients
amp = 0.01

# Start figure
plt.figure(1, figsize=(20, 10))

# Loop on templates
for i in range(0, np.shape(templates)[0]):

    # Get latitude
    latitude = templates[i][2]

    # Open LFE catalog
    namedir = 'catalogs/' + templates[i][0].astype(str)
    namefile = namedir + '/catalog_2007_2009.pkl'
    df = pickle.load(open(namefile, 'rb'))

    # Filter LFEs
    maxc = np.max(df['nchannel'])
    df = df.loc[df['cc'] * df['nchannel'] >= 0.1 * maxc]

    # Get time series
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

    days = np.arange(0, len(X))[X >= 1]
    Xpos = X[X >= 1]
    plt.stem(days, latitude + amp * Xpos, 'k-', markerfmt=' ', basefmt=' ', bottom=latitude)
    plt.axhline(latitude, linewidth=1, color='k')

plt.xlim([-0.5, len(X) - 0.5])
plt.xlabel('Time (days) since 2007/07/01', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.title('Daily LFEs for all families', fontsize=24)
plt.savefig('LFEdistribution/all_families.eps', format='eps')
plt.close(1)
    