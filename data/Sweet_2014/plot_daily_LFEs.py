"""
This script looks at the catalog from Sweet (2014) and plots
for each family the daily number of LFEs in function of time
"""

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta
from scipy.io import loadmat

from date import matlab2ymdhms

# Number of LFE families
nf = 9

# Beginning and end of the period we are looking at
tbegin = datetime(2006, 10, 1, 0, 0, 0)
tend = datetime(2011, 10, 1, 0, 0, 0)

# We construct the time series by counting the number of LFEs
# per one-day-long time window
window = 86400.0

# Length of the time series
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)

# Loop on templates
for n in range(0, nf):
    data = loadmat('catalogs/LFE' + str(n + 1) + 'catalog.mat')
    if (n + 1 <= 4):
        LFEtime = data['peakTimes'][0]
    else:
        LFEtime = data['peakTimes']
    dt = tend - tbegin
    duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
    nw = int(duration / window)
    X = np.zeros(nw, dtype=int)
    # Loop on LFEs
    for i in range(0, np.shape(LFEtime)[0]):
        if (n + 1 <= 4):
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
    # Plot figure
    plt.figure(1, figsize=(20, 10))
    plt.stem(np.arange(0, len(X)), X, 'k-', markerfmt=' ', basefmt=' ')
    plt.xlim([-0.5, len(X) - 0.5])
    plt.xlabel('Time (days) since 2006/10/01', fontsize=24)
    plt.ylabel('Number of LFEs', fontsize=24)
    plt.title('Family {} ({:d} LFEs)'.format(n + 1, np.sum(X)), \
        fontsize=24)
    plt.savefig('LFEdistribution/LFE' + str(n + 1) + '.eps', format='eps')
    plt.close(1)
