"""
This module contains a function to get the timing of the LFEs from the catalog
of Plourde et al. (2015), transform it into a time series by counting the
number of events per time window, and compute the variance for different
length of the time window
"""

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime, timedelta
from sklearn import linear_model
from sklearn.metrics import r2_score

def compute_variance(filename, winlen, tbegin, tend):
    
    # Get the time of LFE detections
    LFEtime = np.loadtxt('../data/LFEcatalog/detections/' + filename + \
        '_detect5_cull.txt', \
        dtype={'names': ('unknown', 'day', 'hour', 'second', 'threshold'), \
             'formats': (np.float, '|S6', np.int, np.float, np.float)}, \
        skiprows=2)
    # Initialize variance
    v = np.zeros(len(winlen))
    for l in range(0, len(winlen)):
        dt = tend - tbegin
        duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
        nw = int(duration / winlen[l])
        nLFE = np.zeros(nw)
        # Loop on time windows
        for i in range(0, nw):
            t1 = tbegin + timedelta(seconds=np.float64(i * winlen[l]))
            t2 = tbegin + timedelta(seconds=np.float64((i + 1) * winlen[l]))
            # Loop on LFEs
            for j in range(0, np.shape(LFEtime)[0]):
                YMD = LFEtime[j][1]
                myYear = 2000 + int(YMD[0 : 2])
                myMonth = int(YMD[2 : 4])
                myDay = int(YMD[4 : 6])
                myHour = LFEtime[j][2] - 1
                myMinute = int(LFEtime[j][3] / 60.0)
                mySecond = int(LFEtime[j][3] - 60.0 * myMinute)
                myMicrosecond = int(1000000.0 * \
                    (LFEtime[j][3] - 60.0 * myMinute - mySecond))
                t = datetime(myYear, myMonth, myDay, myHour, myMinute, \
                    mySecond, myMicrosecond)
                if ((t >= t1) and (t < t2)):
                    nLFE[i] = nLFE[i] + 1
        v[l] = np.var(nLFE)
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(winlen), np.log10(v / winlen), 'ko')
    x = np.reshape(np.log10(winlen), (len(winlen), 1))
    y = np.reshape(np.log10(v / winlen), (len(winlen), 1))
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (window length)', fontsize=24)
    plt.ylabel('Log (variance / window length)', fontsize=24)
    plt.title('{:d} LFEs - alpha = {:4.2f} - R2 = {:4.2f}'.format( \
        np.shape(LFEtime)[0], - regr.coef_[0][0], R2), fontsize=24)
    plt.savefig('variance/' + filename + '.eps', format='eps')

if __name__ == '__main__':

    # Set the parameters
    filename = '080429.15.005'
    winlen = 6 * 60 * np.array([1, 2, 10, 20, 100, 200, 1000])
    tbegin = datetime(2008, 3, 1, 0, 0, 0)
    tend = datetime(2008, 5, 1, 0, 0, 0)

    compute_variance(filename, winlen, tbegin, tend)
