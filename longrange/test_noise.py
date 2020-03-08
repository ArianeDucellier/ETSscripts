"""
"""

import numpy as np

from scipy.stats import chi2

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
    mu = 2.0 * (N - 2) / 3.0
    var = (16.0 * N - 29.0) / 90.0
    x <- embed(ts, 3)
    # Compute the number of turning points
    test.sum <- sum((x[,2] > x[,1] & x[,2] > x[,3]) | (x[,2] < x[,1] & x[,2] < x[,3]))
    # Normalize
    test <- abs(test.sum - mu) / sqrt(var)
    p.value <- 2 * (1 - pnorm(test))
    structure(list(test.sum=test.sum, test=test, p.value=p.value, mu=mu, var=var))