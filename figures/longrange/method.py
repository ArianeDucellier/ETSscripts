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

# Draw visualization of fractional parameter
data = pickle.load(open('../../longrange/variance_Sweet/LFE1.pkl', 'rb'))
m = data[0]
V = data[1]
nLFE = data[2]
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
