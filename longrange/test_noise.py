"""
"""

import numpy as np

from math import sqrt
from scipy.stats import chi2, norm

def remove_seasonality(X, d):
    """
    X time series
    d period
    """
    # Remove seasonality to estimate trend
    if (d % 2 == 1):
        a = np.repeat(1.0 / d, d)
    else:
        a = np.repeat(1.0 / d, d + 1)
        a[0] = 1.0 / (2.0 * d)
        a[d] = 1.0 / (2.0 * d)
    n = np.shape(a)[0]
    N = np.shape(X)[0]
    Xpad = np.concatenate((np.zeros(n), X, np.zeros(n)))
    m = np.correlate(Xpad, a)
    m = m[int((n + 1) / 2) : (N + int((n + 1) / 2))]
    u = X - m
    w = np.zeros(d)
    for i in range(0, d):
        indices = np.arange(i, N, d)
        w[i] = np.mean(X[indices])
    wbar = np.mean(w)
    s0 = w - wbar
    s = np.tile(s0, int(N / d))
    if (N % d != 0):
        s = np.concatenate((s, np.zeros(N % d)))
    d = X - s
    return d
    
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

def turning_point(X):
    """
    """
    N = np.shape(X)[0]
    mu = 2.0 * (N - 2.0) / 3.0
    var = (16.0 * N - 29.0) / 90.0
    xtm1 = X[0 : (N - 2)]
    xt = X[1 : (N - 1)]
    xtp1 = X[2 : N]
    # Compute the number of turning points
    test_sum = np.sum(((xt > xtm1) & (xt > xtp1)) | ((xt < xtm1) & (xt < xtp1)))
    # Normalize
    test = abs(test_sum - mu) / sqrt(var)
    p_value = 2.0 * (1.0 - norm.cdf(test))
    return(test_sum, p_value)

def difference_sign(X):
    """
    """
    N = np.shape(X)[0]
    mu = (N - 1.0) / 2.0
    var = (N + 1.0) / 12.0
    xtm1 = X[0 : (N - 1)]
    xt = X[1 : N]
    # Number of times where X_t > X_t-1
    test_sum = np.sum(xt > xtm1)
    # Normalize
    test = abs(test_sum - mu) / sqrt(var)
    p_value = 2.0 * (1.0 - norm.cdf(test))
    return(test_sum, p_value)

def rank(X):
    """
    """
    N = np.shape(X)[0]
    mu = N * (N - 1.0) / 4.0
    var = N * (N - 1.0) * (2.0 * N + 5.0) / 72.0
    Xr = np.tile(X, N).reshape((N, -1))
    Xc = np.transpose(Xr)
    ir = np.tile(np.arange(0, N), N).reshape((N, -1))
    ic = np.transpose(ir)
    # Number of times where X_r > X_s and r > s
    test_sum = np.sum((Xr < Xc) & (ir < ic))    
    # Normalize
    test = abs(test_sum - mu) / sqrt(var)
    p_value = 2.0 * (1.0 - norm.cdf(test))
    return(test_sum, p_value)

def runs(X):
    """
    """
    Xnorm = X - np.mean(X)
    N = np.shape(Xnorm)[0]
    Npos = np.sum(Xnorm > 0)
    Nneg = N - Npos
    mu = 2.0 * Npos * Nneg / N + 1
    var = (mu - 1.0) * (mu - 2.0) / (N - 1)   
    Xbinary = np.copy(Xnorm)
    Xbinary[Xnorm > 0] = 1
    Xbinary[Xnorm < 0] = -1
    # Number of runs
    test_sum = np.sum(np.abs(np.diff(Xbinary)) > 0) + 1
    # Normalize
    test = abs(test_sum - mu) / sqrt(var)
    p_value = 2.0 * (1.0 - norm.cdf(test))
    return(test_sum, p_value)
