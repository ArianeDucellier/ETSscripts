"""
"""

import numpy as np

from scipy.stats import chi2

def portmanteau(ACF, N, alpha, p):
    """
    ACF autocorrelation function
    N length of time series
    alpha level of test
    p number of parameters estimated
    """
    hmax = np.shape(ACF)[0] - 1
    QLB = np.zeros(hmax)
    quantiles = np.zeros(hmax)
    for h in range(1, hmax + 1):
        if (h > p):
            denom = np.arange(N - 1, N - h - 1, -1)
            QLB[h - 1] = N * (N + 2) * np.sum(np.power(ACF[1 : (h + 1)], 2.0) / denom)
            quantiles[h - 1] = chi2.ppf(1 - alpha, h - p)
    return(QLB, quantiles)
