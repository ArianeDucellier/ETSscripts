"""
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import sqrt

from ACVF import ACVF, ACF
from test_noise import remove_seasonality, portmanteau, turning_point, difference_sign, rank, runs

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

    # Create figure for portmanteau test
    plt.figure(1, figsize=(18, (dmax - 1) * 4))

    # Create output files for other tests
    turning_file = open('Noise/' + filename + '_turningpoint.txt', 'a')
    difference_file = open('Noise/' + filename + '_differencesign.txt', 'a')
    rank_file = open('Noise/' + filename + '_rank.txt', 'a')
    runs_file = open('Noise/' + filename + '_runs.txt', 'a')

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
        if (d == dmax):
            plt.xlabel('Time lag', fontsize=24)
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

        # Turning point test
        (test_sum, p_value) = turning_point(Xres)
        turning_file.write('Deseasonalized data with period {:d} days\n'.format(d))
        turning_file.write('Probability of having more than {:d} turning points is {:f}\n'.format(test_sum, p_value))
        turning_file.write('\n')

        # Difference sign test
        (test_sum, p_value) = difference_sign(Xres)
        difference_file.write('Deseasonalized data with period {:d} days\n'.format(d))
        difference_file.write('Probability of having more than {:d} positive differences is {:f}\n'.format(test_sum, p_value))
        difference_file.write('\n')

        # Rank test
        (test_sum, p_value) = rank(Xres)
        rank_file.write('Deseasonalized data with period {:d} days\n'.format(d))
        rank_file.write('Probability of having more than {:d} pairs with same rank is {:f}\n'.format(test_sum, p_value))
        rank_file.write('\n')

        # Runs test
        (test_sum, p_value) = runs(Xres)
        runs_file.write('Deseasonalized data with period {:d} days\n'.format(d))
        runs_file.write('Probability of having more than {:d} runs is {:f}\n'.format(test_sum, p_value))
        runs_file.write('\n')

    # Finalize plot
    plt.tight_layout()
    plt.savefig('Noise/' + filename + '.eps', format='eps')
    plt.close(1)

    # Close files
    turning_file.close()
    difference_file.close()
    rank_file.close()
    runs_file.close()
