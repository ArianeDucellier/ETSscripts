"""
Script to make a figure explaining the method used
to compute the fractional parameter d
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

from datetime import datetime, timedelta
from scipy.io import loadmat
from sklearn import linear_model
from sklearn.metrics import r2_score

from date import matlab2ymdhms

# Draw time series
tbegin = datetime(2006, 10, 1, 0, 0, 0)
tend = datetime(2011, 10, 1, 0, 0, 0)
window = 86400.0
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)
data = loadmat('../../data/Sweet_2014/catalogs/LFE1catalog.mat')
LFEtime = data['peakTimes'][0]
cc = data['xcor'][0]
LFEtime = LFEtime[cc >= 0.275]
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)
X = np.zeros(nw, dtype=int)
for i in range(0, np.shape(LFEtime)[0]):
    (myYear, myMonth, myDay, myHour, myMinute, mySecond, \
        myMicrosecond) = matlab2ymdhms(LFEtime[i], False)
    t = datetime(myYear, myMonth, myDay, myHour, myMinute, mySecond, \
        myMicrosecond)
    if ((tbegin <= t) and (t < tbegin + timedelta(seconds=nw * window))):
        dt = t - tbegin
        duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * \
            0.000001
        index = int(duration / window)
        X[index] = X[index] + 1
plt.figure(1, figsize=(20, 10))
plt.stem(np.arange(0, len(X)), X, 'k-', markerfmt=' ', basefmt=' ')
plt.xlim([-0.5, len(X) - 0.5])
plt.xlabel('Time (days) since 2006/10/01', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.savefig('timeseries_LFE1_Sweet.eps', format='eps')
plt.close(1)

# Get times series (one-minute-long bins)
window = 60.0
nw = int(duration / window)
X = np.zeros(nw, dtype=int)
for i in range(0, np.shape(LFEtime)[0]):
    (myYear, myMonth, myDay, myHour, myMinute, mySecond, \
        myMicrosecond) = matlab2ymdhms(LFEtime[i], False)
    t = datetime(myYear, myMonth, myDay, myHour, myMinute, mySecond, \
        myMicrosecond)
    if ((tbegin <= t) and (t < tbegin + timedelta(seconds=nw * window))):
        dt = t - tbegin
        duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * \
            0.000001
        index = int(duration / window)
        X[index] = X[index] + 1

# Compute fractional parameter
def aggregate(X, m):
    N = len(X)
    N2 = int(N / m)
    X2 = X[0 : N2 * m]
    X2 = np.reshape(X2, (N2, int(m)))
    Xm = np.mean(X2, axis=1)
    return Xm

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980, \
    3827, 4914, 6310, 8103, 10404, 13359, 17154, 22026, 28282, 36315, \
    46630, 59874, 76879, 98715], dtype=int)

V = np.zeros(len(m))
for i in range(0, len(m)):
    Xm = aggregate(X, m[i])
    V[i] = np.var(Xm)

# Draw visualization of fractional parameter
nLFE = np.sum(X)
x = np.reshape(np.log10(m), (len(m), 1))
y = np.reshape(np.log10(V * m), (len(V), 1))
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(x, y)
y_pred = regr.predict(x)
R2 = r2_score(y, y_pred)
d = 0.5 * regr.coef_[0][0]
plt.figure(1, figsize=(10, 10))
plt.plot(np.log10(m), np.log10(V * m), 'ko')
plt.plot(x, y_pred, 'r-')
plt.xlabel('Log (Sample size)', fontsize=24)
plt.ylabel('Log (Sample size * Variance)', fontsize=24)
plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
    nLFE, d, R2), fontsize=24)
plt.savefig('variance_LFE1_Sweet.eps', format='eps')
plt.close(1)
