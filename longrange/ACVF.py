"""
Scripts to compute ACVF and PACF of time series
"""

import numpy as np

def ACVF(X, h):
    """
    """
    ACVF = np.zeros(h)
    for i in range(0, h):
        N = np.shape(X)[0]
        Xbar = np.mean(X)
        Xt = X[0 : (N - i)]
        Xth = X[i : N]
        ACVF[i] = np.sum((Xth - Xbar) * (Xt - Xbar)) / N
    return(ACVF)

def ACF(ACVF):
    """
    """
    gamma0 = ACVF[0]
    ACF = ACVF / gamma0
    return(ACF)

def PACF(ACVF):
    """
    """
    p = np.shape(ACVF)[0] - 1
    # phi_1,1 = gamma(1) / gamma(0)
    phis = np.array([ACVF[1] / ACVF[0]])
    # v_0 = gamma(0)
    v = np.repeat(ACVF[0], p + 1)
    # PACF(1) = phi_1,1
    PACF = np.repeat(phis, p)
    # v_1 = v_0 (1 - phi_1,1^2)
    v[1] = v[0] * (1 - phis[0]**2)
    if (p > 1):
        for k in range(1, p):
            oldphis = np.copy(phis)
            # phi_k,k = gamma(k) - sum_k-1,j gamma(k-j) / v_k-1
            a = (ACVF[k + 1] - np.sum(np.dot(oldphis, np.flip(ACVF[1 : (k + 1)])))) / v[k]
            phis = np.repeat(a, k + 1)
            # phi_k,i = phi_k-1,i - phi_k,k 
            phis[0 : k] = oldphis - phis[k] * np.flip(oldphis)
            # PACF(k) = phi_k,k
            PACF[k] = phis[k]
            # v_k = v_k-1 (1 - phi_k,k^2)
            v[k + 1] = v[k] * (1 - phis[k]**2)
    return(PACF, v)
