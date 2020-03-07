"""
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from ACVF import ACVF, ACF
from test_noise import portmanteau

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
    Xm = np.mean(X2, axis=1)
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

for i in range(0, 1): #len(families)):
    filename = families[i]
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X_1 = data[3]
    X_2 = aggregate(X_1, 60)
    X_3 = aggregate(X_2, 24)
    X_4 = aggregate(X_3, 7)

    # Portmanteau test
    plt.figure(1, figsize=(9, 24))

    N = np.shape(X_1)[0]
    acvf = ACVF(X_1, h)
    acf = ACF(acvf)
    (QLB, quantiles) = portmanteau(acf, N, alpha, p)
    plt.subplot2grid((4, 1), (1, 0))
    plt.plot(np.arange(p + 1, h), QLB[p : h], 'bo', label='QLB - 1 min')
    for i in range(p + 1, h):
        if (i == p + 1):
            plt.plot(np.linspace(i - 0.25, i + 0.25, 2), np.repeat(quantiles[i - 1], 2), 'r-', \
                label='{} quantile'.format(100 * (1 - alpha)))
        else:
            plt.plot(np.linspace(i - 0.25, i + 0.25, 2), np.repeat(quantiles[i - 1], 2), 'r-')
    plt.ylabel('', fontsize=24)
    plt.legend(loc=1)

    # Finalize plot
    plt.suptitle('Family ' + filename, fontsize=30)
    plt.savefig('Noise/' + filename + '.eps', format='eps')
    plt.close(1)
