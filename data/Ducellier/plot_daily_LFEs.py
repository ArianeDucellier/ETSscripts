"""
This script looks at the catalog from Ducellier and plots for
each family the daily number of LFEs in function of time
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from math import floor

# Beginning and end of the period we are looking at
tbegin = datetime(2007, 7, 1, 0, 0, 0)
tend = datetime(2008, 1, 1, 0, 0, 0)

# We construct the time series by counting the number of LFEs
# per one-day-long time window
window = 86400.0

# Length of the time series
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)

# Family name
filename = '080421.14.048'

df = pickle.load(open('../../../AWS/ETSscripts/catalog/LFEs/' + \
    filename + '/catalog_2007_07-12.pkl', 'rb'))
df = df.loc[df['cc'] >= 0.06]
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
# Plot figure
plt.figure(1, figsize=(20, 10))
plt.stem(np.arange(0, len(X)), X, 'k-', markerfmt=' ', basefmt=' ')
plt.xlim([-0.5, len(X) - 0.5])
plt.xlabel('Time (days) since 2007/08/01', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.title('Family {} ({:d} LFEs)'.format(filename, np.sum(X)), \
    fontsize=24)
plt.savefig('LFEdistribution/' + filename + '.eps', format='eps')
plt.close(1)
