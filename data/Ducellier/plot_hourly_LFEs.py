"""
This script looks at the catalog from southern Cascadia (2007-2009) and plots
for each family the daily number of LFEs in function of time
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from math import cos, floor, pi, sin, sqrt

params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)

# Earth's radius and ellipticity
a = 6378.136
e = 0.006694470

# Conversion lat-lon into kilometers
lat0 = 40.836
lon0 = -123.497
dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
    sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
    pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)

# List of LFE families
templates = np.loadtxt('../Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Beginning and end of the period we are looking at
tbegin = datetime(2008, 4, 15, 0, 0, 0)
tend = datetime(2008, 5, 15, 0, 0, 0)

# We construct the time series by counting the number of LFEs
# per one-day-long time window
window = 3600.0

# Length of the time series
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)

# Loop on templates
for i in range(0, np.shape(templates)[0]):

    # Distance to epicenter
    latitude = templates[i][2]
    longitude = templates[i][3]
    x = (longitude - lon0) * dx
    y = (latitude - lat0 ) * dy
    dist = sqrt(x**2.0 + y**2.0)

    # Open LFE catalog
    namedir = 'catalogs/' + templates[i][0].astype(str)
    namefile = namedir + '/catalog_2007_2009.pkl'
    df = pickle.load(open(namefile, 'rb'))

    # Filter LFEs
    #df = df.loc[df['cc'] >= 0.07]

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

    # April 30th 2008 earthquake
    t = datetime(2008, 4, 30, 3, 3, 6, 0)
    dt = t - tbegin
    duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
    index = int(duration / window)
            
    # Plot figure
    plt.figure(1, figsize=(20, 10))
    plt.axvline(index, linewidth=2, color='red')
    plt.stem(np.arange(0, len(X)), X, 'k-', markerfmt=' ', basefmt=' ')
    plt.xlim([-0.5, len(X) - 0.5])
    plt.xlabel('Time (hours) since 2008/05/15', fontsize=24)
    plt.ylabel('Number of LFEs', fontsize=24)
    plt.title('Family {} ({:4.2f} km from epicenter)'.format(templates[i][0].astype(str), dist), \
        fontsize=24)
    plt.savefig('zoom_eq/' + templates[i][0].astype(str) + '.eps', format='eps')
    plt.close(1)
