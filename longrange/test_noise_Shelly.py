"""
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import sqrt

from ACVF import ACVF, ACF
from test_noise import remove_seasonality, portmanteau

dmax = 6
h = 21
alpha = 0.05
p = 0

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
    X = aggregate(X, 60 * 24)

    plt.figure(1, figsize=(18, (dmax - 1) * 4))

    # Loop on periodicity
    for d in range(2, (dmax + 1)):
        Xres = remove_seasonality(X, d)

        # Portmanteau test
        N = np.shape(Xres)[0]
        acvf = ACVF(Xres, h)
        acf = ACF(acvf)
        plt.subplot2grid((dmax - 1, 2), (d - 2, 0))
        plt.plot(np.arange(0, h), acf, 'bo', label='{} days'.format(d))
        plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
        plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
        plt.ylabel('ACF', fontsize=24)
        plt.legend(loc=1)
        (QLB, quantiles) = portmanteau(acf, N, alpha, p)
        plt.subplot2grid((dmax - 1, 2), (d - 2, 1))
        plt.plot(np.arange(p + 1, h), QLB[p : h], 'bo')
        for j in range(p + 1, h):
            if (j == p + 1):
                plt.plot(np.linspace(j - 0.25, j + 0.25, 2), np.repeat(quantiles[j - 1], 2), 'r-', \
                    label='{} quantile'.format(100 * (1 - alpha)))
            else:
                plt.plot(np.linspace(j - 0.25, j + 0.25, 2), np.repeat(quantiles[j - 1], 2), 'r-')
        plt.ylabel('QLB', fontsize=24)
        if (d == dmax):
            plt.xlabel('Time lag', fontsize=24)

    # Finalize plot
    plt.suptitle('Family ' + filename, fontsize=30)
    plt.savefig('Noise/' + filename + '.eps', format='eps')
    plt.close(1)
