"""
This script looks at the catalog from Plourde et al. (2015) and plots for
each template the hourly number of LFEs in function of time
"""

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta

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
# per one-hour-long time window
window = 3600.0

# Length of the time series
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)

# Loop on templates
for i in range(0, np.shape(templates)[0]):
    # Get the time of LFE detections
    filename = templates[i][0].astype(str)
    LFEtime = np.loadtxt('detections/' + filename + '_detect5_cull.txt', \
        dtype={'names': ('unknown', 'day', 'hour', 'second', 'threshold'), \
             'formats': (np.float, '|S6', np.int, np.float, np.float)}, \
        skiprows=2)
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
    # Plot figure
    plt.figure(1, figsize=(20, 10))
    plt.stem(np.arange(0, len(X)) / 24.0, X, 'k-', markerfmt=' ', basefmt=' ')
    plt.xlim([-0.5, len(X) / 24.0 - 0.5])
    plt.xlabel('Time (days) since 2008/03/01', fontsize=24)
    plt.ylabel('Number of LFEs', fontsize=24)
    plt.title('Template {} ({:d} LFEs)'.format(filename, np.sum(X)), \
        fontsize=24)
    plt.savefig('LFEdistribution/' + filename + '.eps', format='eps')
    plt.close(1)
    