"""
This script looks at the catalog from Shelly (2017) and plots for
each family the daily number of LFEs in function of time
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datetime import datetime, timedelta
from math import floor

# Read the LFE file
LFEtime = pd.read_csv('jgrb52060-sup-0002-datas1.txt', \
    delim_whitespace=True, header=None, skiprows=2)
LFEtime.columns = ['year', 'month', 'day', 's_of_day', 'hr', 'min', 'sec', \
    'ccsum', 'meancc', 'med_cc', 'seqday', 'ID', 'latitude', 'longitude', \
    'depth', 'n_chan']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

# Beginning and end of the period we are looking at
tbegin = datetime(2001, 4, 6, 0, 0, 0)
tend = datetime(2016, 9, 20, 0, 0, 0)

# We construct the time series by counting the number of LFEs
# per one-day-long time window
window = 86400.0

# Length of the time series
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)

# Loop on LFE families
for i in range(0, len(families)):
    data = LFEtime.loc[LFEtime['ID'] == families[i]]
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
    # Plot figure
    plt.figure(1, figsize=(20, 10))
    params = {'xtick.labelsize':20,
          'ytick.labelsize':20}
    pylab.rcParams.update(params)
    # 2004 Parkfield earhquake
    plt.axvline(1269, color='red', linestyle='--')
    plt.stem(np.arange(0, len(X)), X, 'k-', markerfmt=' ', basefmt=' ')
    plt.xlim([-0.5, len(X) - 0.5])
    plt.xlabel('Time (days) since 2001/04/06', fontsize=24)
    plt.ylabel('Number of LFEs', fontsize=24)
    plt.title('Family {} ({:d} LFEs)'.format(families[i], np.sum(X)), \
        fontsize=24)
    plt.savefig('LFEdistribution/' + families[i] + '.eps', format='eps')
    plt.close(1)
