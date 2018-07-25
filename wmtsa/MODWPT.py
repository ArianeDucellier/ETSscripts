"""
This module contains functions to compute the MODWPT of a time series
"""

import matplotlib.pyplot as plt
import numpy as np

import MODWT

def get_Wjn(h, g, j, n, Wjm1n):
    """
    Compute the MODWPT coefficients Wj,2n,t and Wj,2n+1,t at level j
    from the MODWPT coefficients Wj-1,n,t at level j - 1

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWPT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWPT scaling filter
        type j = integer
        j = Current level of the MODWPT decomposition
        type n = integer
        n = Index of the MODWPT vector Wj-1,n at level j - 1
        type Wjm1n = 1D numpy array
        Wjm1n = MODWPT coefficients Wj-1,n,t at level j - 1
    Output:
        type Wjn1 = 1D numpy array
        Wjn1 = MODWPT coefficients Wj,2n,t at level j
        type Wjn2 = 1D numpy array
        Wjn2 = MODWPT coefficients Wj,2n+1,t at level j
    """
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    N = len(Wjm1n)
    assert ((n >= 0) and (n <= 2 ** (j - 1) - 1)), \
        'Index n must be >= 0 and <= 2 ** (j - 1) - 1'
    Wjn1 = np.zeros(N)
    Wjn2 = np.zeros(N)
    L = np.shape(h)[0]
    if (n % 2 == 0):
        an = g
        bn = h
    else:
        an = h
        bn = g
    for t in range(0, N):
        for l in range(0, L):
            index = int((t - (2 ** (j - 1)) * l) % N)
            Wjn1[t] = Wjn1[t] + an[l] * Wjm1n[index]
            Wjn2[t] = Wjn2[t] + bn[l] * Wjm1n[index]
    return (Wjn1, Wjn2)

def get_Wj(h, g, j, Wjm1):
    """
    Compute the MODWPT coefficients Wj at level j
    from the MODWPT coefficients Wj-1 at level j - 1

    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the MODWPT wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the MODWPT scaling filter
        type j = integer
        j = Current level of the MODWPT decomposition
        type Wjm1 = list of n = 2 ** (j - 1) 1D numpy arrays
        Wjm1 = MODWPT coefficients Wj-1 at level j - 1
    Output:
        type Wj = list of n = 2 ** j 1D numpy arrays
        Wj = MODWPT coefficients Wj at level j
    """
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
    Wj = []
    for n in range(0, 2 ** (j - 1)):
        Wjm1n = Wjm1[n]
        (Wjn1, Wjn2) = get_Wjn(h, g, j, n, Wjm1n)
        Wj.append(Wjn1)
        Wj.append(Wjn2)
    return Wj

def get_MODWPT(X, name, J):
    """
    Compute the MODWPT of X up to level J
    
    Input:
        type X = 1D numpy array
        X = Time series
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial MODWPT
    Output:
        type W = list of J+1 lists 1D numpy arrays
        W = Vectors of MODWPT coefficients at levels 0, ... , J
    """
    assert (type(J) == int), \
        'Level of DWPT must be an integer'
    assert (J >= 1), \
        'Level of MODWPT must be higher or equal to 1'
    g = MODWT.get_scaling(name)
    h = MODWT.get_wavelet(g)
    W = [X]
    for j in range(1, J + 1):
        Wjm1 = W[-1]
        Wj = get_Wj(h, g, j, Wjm1)
        W.append(Wj)
    return W
