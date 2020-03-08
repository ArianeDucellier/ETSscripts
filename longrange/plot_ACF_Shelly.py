"""
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import sqrt

from ACVF import ACVF, ACF, PACF

h = 21

def aggregate(X, m):
    """
    Function to return aggregated time series

    Input:
        type X = numpy array
        X = Time series
        type m = integer
        m = Number of values for the aggregation
    Output:
        type Xm = numpy array
        Xm = Aggregated time series
    """
    N = len(X)
    N2 = int(N / m)
    X2 = X[0 : N2 * m]
    X2 = np.reshape(X2, (N2, int(m)))
    Xm = np.sum(X2, axis=1)
    return Xm

# Read the LFE file
LFEtime = pd.read_csv('../data/Shelly_2017/jgrb52060-sup-0002-datas1.txt', \
    delim_whitespace=True, header=None, skiprows=2)
LFEtime.columns = ['year', 'month', 'day', 's_of_day', 'hr', 'min', 'sec', \
    'ccsum', 'meancc', 'med_cc', 'seqday', 'ID', 'latitude', 'longitude', \
    'depth', 'n_chan']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

dirname = '../data/Shelly_2017/timeseries/'

params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)

for i in range(0, len(families)):
    filename = families[i]
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X = data[3]

    plt.figure(1, figsize=(27, 24))

    # One-minute-long time window
    N = np.shape(X)[0]
    acvf = ACVF(X, h)
    acf = ACF(acvf)
    pacf = PACF(acvf)[0]
    plt.subplot2grid((4, 3), (0, 0))
    plt.plot(np.arange(0, h), acf, 'bo', label='1 min')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.ylabel('ACF', fontsize=24)
    plt.legend(loc=1)
    plt.subplot2grid((4, 3), (0, 1))
    plt.plot(np.arange(1, h), pacf, 'bo', label='1 min')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.ylabel('PACF', fontsize=24)
    plt.legend(loc=1)
    plt.subplot2grid((4, 3), (0, 2))
    plt.plot(np.arange(1, h), 1.0 / pacf, 'bo', label='1 min')
    plt.ylabel('1/PACF', fontsize=24)
    plt.legend(loc=1)

    # One-hour-long time window
    X = aggregate(X, 60)
    N = np.shape(X)[0]
    acvf = ACVF(X, h)
    acf = ACF(acvf)
    pacf = PACF(acvf)[0]
    plt.subplot2grid((4, 3), (1, 0))
    plt.plot(np.arange(0, h), acf, 'bo', label='1 hour')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.ylabel('ACF', fontsize=24)
    plt.legend(loc=1)
    plt.subplot2grid((4, 3), (1, 1))
    plt.plot(np.arange(1, h), pacf, 'bo', label='1 hour')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.ylabel('PACF', fontsize=24)
    plt.legend(loc=1)
    plt.subplot2grid((4, 3), (1, 2))
    plt.plot(np.arange(1, h), 1.0 / pacf, 'bo', label='1 hour')
    plt.ylabel('1/PACF', fontsize=24)
    plt.legend(loc=1)

    # One-day-long time window
    X = aggregate(X, 24)
    N = np.shape(X)[0]
    acvf = ACVF(X, h)
    acf = ACF(acvf)
    pacf = PACF(acvf)[0]
    plt.subplot2grid((4, 3), (2, 0))
    plt.plot(np.arange(0, h), acf, 'bo', label='1 day')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.ylabel('ACF', fontsize=24)
    plt.legend(loc=1)
    plt.subplot2grid((4, 3), (2, 1))
    plt.plot(np.arange(1, h), pacf, 'bo', label='1 day')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.ylabel('PACF', fontsize=24)
    plt.legend(loc=1)
    plt.subplot2grid((4, 3), (2, 2))
    plt.plot(np.arange(1, h), 1.0 / pacf, 'bo', label='1 day')
    plt.ylabel('1/PACF', fontsize=24)
    plt.legend(loc=1)

    # One-week-long time window
    X = aggregate(X, 7)
    N = np.shape(X)[0]
    acvf = ACVF(X, h)
    acf = ACF(acvf)
    pacf = PACF(acvf)[0]
    plt.subplot2grid((4, 3), (3, 0))
    plt.plot(np.arange(0, h), acf, 'bo', label='1 week')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.xlabel('Time lag', fontsize=24)
    plt.ylabel('ACF', fontsize=24)
    plt.legend(loc=1)
    plt.subplot2grid((4, 3), (3, 1))
    plt.plot(np.arange(1, h), pacf, 'bo', label='1 week')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.xlabel('Time lag', fontsize=24)
    plt.ylabel('PACF', fontsize=24)
    plt.legend(loc=1)
    plt.subplot2grid((4, 3), (3, 2))
    plt.plot(np.arange(1, h), 1.0 / pacf, 'bo', label='1 week')
    plt.ylabel('1/PACF', fontsize=24)
    plt.legend(loc=1)

    # Finalize plot
    plt.suptitle('Family ' + filename, fontsize=30)
    plt.savefig('ACVF/' + filename + '.eps', format='eps')
    plt.close(1)
