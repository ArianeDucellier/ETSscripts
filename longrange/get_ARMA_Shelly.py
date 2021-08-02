"""
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pickle

from math import sqrt

from ACVF import ACVF, ACF
from get_ARMA_parameters import yule_walker
from test_noise import portmanteau

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

family_list = ['1788sss', '1812sss', '15025s', '25879s', '32192s', '41767s']
p_list = [1, 1, 1, 1, 1, 1]

h = 21
alpha = 0.05

dirname = '../data/Shelly_2017/timeseries/'

params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)

for (family, p) in zip(family_list, p_list):
    data = pickle.load(open(dirname + family + '.pkl', 'rb'))
    X = data[3]
    X = aggregate(X, 60 * 24)
    X = X - np.mean(X)
    (phi, sigma2) = yule_walker(X, p)
    acvf = ACVF(X, h)
    acvf_hat = sigma2 / (1.0 - phi[0]**2.0) * np.power(np.repeat(phi[0], h), np.arange(0, h))
    Z = X[1:] - phi[0] * X[:-1]
    acvf_Z = ACVF(Z, h)
    # Plot ACVF and fitted ACF
    plt.figure(1, figsize=(9, 6))
    plt.plot(np.arange(0, h), acvf, 'bo', label='data')
    plt.plot(np.arange(0, h), acvf_hat, 'r-', label='model')
    plt.xlabel('Time lag', fontsize=24)
    plt.ylabel('ACVF', fontsize=24)
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig('Model/' + family + '.eps', format='eps')
    plt.close(1)
    # Plot ACVF of residuals
    plt.figure(2, figsize=(18, 6))
    N = np.shape(Z)[0]
    acf_Z = ACF(acvf_Z)
    plt.subplot2grid((1, 2), (0, 0))
    plt.plot(np.arange(0, h), acf_Z, 'bo')
    plt.axhline(1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.axhline(-1.96 / sqrt(N), linewidth=2, linestyle='--', color='red')
    plt.xlabel('Time lag', fontsize=24)
    plt.ylabel('ACF', fontsize=24)
    (QLB, quantiles) = portmanteau(acf_Z, N, alpha, p)
    plt.subplot2grid((1, 2), (0, 1))
    plt.plot(np.arange(p + 1, h), QLB[p : h], 'bo')
    for j in range(p + 1, h):
        if (j == p + 1):
            plt.plot(np.linspace(j - 0.25, j + 0.25, 2), np.repeat(quantiles[j - 1], 2), 'r-', \
                label='{} quantile'.format(100 * (1 - alpha)))
        else:
            plt.plot(np.linspace(j - 0.25, j + 0.25, 2), np.repeat(quantiles[j - 1], 2), 'r-')
    plt.xlabel('Time lag', fontsize=24)
    plt.ylabel('QLB', fontsize=24)
    plt.tight_layout()
    plt.savefig('Model/' + family + '_residuals.eps', format='eps')
    plt.close(2)
