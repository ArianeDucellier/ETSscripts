"""
This module contains functions to compute the MODWPT of a time series
"""

import matplotlib.pyplot as plt
import numpy as np

from math import floor

import DWPT, MODWT

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
        type W = list of J+1 lists of 1D numpy arrays
        W = Vectors of MODWPT coefficients at levels 0, ... , J
    """
    assert (type(J) == int), \
        'Level of MODWPT must be an integer'
    assert (J >= 1), \
        'Level of MODWPT must be higher or equal to 1'
    g = MODWT.get_scaling(name)
    h = MODWT.get_wavelet(g)
    W = [[X]]
    for j in range(1, J + 1):
        Wjm1 = W[-1]
        Wj = get_Wj(h, g, j, Wjm1)
        W.append(Wj)
    return W

def get_u(name, J):
    """
    Compute the series of filters to carry out the MODWPT up to level J

    Input:
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial MODWPT
    Output:
        type u = list of J lists of 1D numpy arrays
        u = Vectors of filters
    """
    assert (type(J) == int), \
        'Level of MODWPT must be an integer'
    assert (J >= 1), \
        'Level of MODWPT must be higher or equal to 1'
    g = MODWT.get_scaling(name)
    h = MODWT.get_wavelet(g)
    L = np.shape(h)[0]
    u = [[g, h]]
    for j in range(2, J + 1):
        uj = []
        Lj = (2 ** j - 1) * (L - 1) + 1
        for n in range(0, 2 ** j):
            ujm1 = u[-1][int(floor(n / 2))]
            if ((n % 4 == 0) or (n % 4 == 3)):
                un = g
            else:
                un = h
            ujn = np.zeros(Lj)
            for l in range(0, Lj):
                for k in range(0, L):
                    index = int(l - (2 ** (j - 1)) * k)
                    if ((index >= 0) and (index < len(ujm1))):
                        ujn[l] = ujn[l] + un[k] * ujm1[index]
            uj.append(ujn)
        u.append(uj)
    return u
                
def get_D(W, name):
    """
    Compute the details of the MODWPT using the wavelet coefficients

    Input:
        type W = list of J+1 lists of 1D numpy arrays
        W = Vectors of MODWPT coefficients at levels 0, ... , J
        type name = string
        name = Name of the wavelet filter
    Output:
        type D = list of J lists of 1D numpy arrays
        D = Vectors of MODWPT details at levels 1, ... , J
    """
    g = MODWT.get_scaling(name)
    h = MODWT.get_wavelet(g)
    J = len(W) - 1
    u = get_u(name, J)
    L = np.shape(h)[0]   
    N = len(W[0][0])
    D = []
    for j in range(1, J + 1):
        Wj = W[j]
        Dj = []
        Lj = (2 ** j - 1) * (L - 1) + 1
        for n in range(0, 2 ** j):
            Wjn = Wj[n]
            Djn = np.zeros(N)
            for t in range(0, N):
                for l in range(0, Lj):
                    index = int((t + l) % N)
                    Djn[t] = Djn[t] + u[j - 1][n][l] * Wjn[index]
            Dj.append(Djn)
        D.append(Dj)
    return D
            
if __name__ == '__main__':

    # Test 1
    def test1():
        """
        Reproduce plot of Figure 235 from WMTSA

        Input:
            None
        Output:
            None
        """
        X = np.loadtxt('../tests/data/magS4096.dat')
        N = np.shape(X)[0]
        W = get_MODWPT(X, 'LA8', 4)
        W4 = W[4]
        nu = DWPT.get_nu('LA8', 4)
        L = 8
        Lj = (2 ** 4 - 1) * (L - 1) + 1
        dt = 1.0 / 24.0
        plt.figure(1, figsize=(15, 51))
        # Plot data
        plt.subplot2grid((17, 1), (16, 0))
        plt.plot(dt * np.arange(0, N), X, 'k', label='X')
        plt.xlim([0, dt * (N - 1)])
        plt.xlabel('t (days)')
        plt.legend(loc=1)
        # Plot 16 MODWPT vectors of coefficients at level 4
        for n in range(0, 16):
            W4n = W4[n]
            plt.subplot2grid((17, 1), (n, 0))
            tshift = np.zeros(N)
            for t in range(0, N):
                 tshift[t] = dt * ((t - abs(nu[3][n])) % N)
            torder = np.argsort(tshift)
            plt.plot(tshift[torder], W4n[torder], 'k', \
                label='T' + str(nu[3][n]) + 'W4,' + str(n))
            plt.axvline(dt * (Lj - 2 - abs(nu[3][n])), linewidth=1, \
                color='red')
            plt.axvline(dt * (N - abs(nu[3][n])), linewidth=1, color='red')
            plt.xlim([0, dt * (N - 1)])
            plt.legend(loc=1)
        plt.savefig('../tests/MODWPT/magS4096_W4.eps', format='eps')
        plt.close(1)

    # Compute LA8 MODWPT of the solar physics time series from WMTSA
    # See Figure 235 in WMTSA
    test1()

    # Test 2
    def test2():
        """
        Reproduce plot of Figure 238 from WMTSA

        Input:
            None
        Output:
            None
        """
        X = np.loadtxt('../tests/data/magS4096.dat')
        N = np.shape(X)[0]
        W = get_MODWPT(X, 'LA8', 4)
        D = get_D(W, 'LA8')
        D4 = D[3]
        dt = 1.0 / 24.0
        plt.figure(1, figsize=(15, 51))
        # Plot data
        plt.subplot2grid((17, 1), (16, 0))
        plt.plot(dt * np.arange(0, N), X, 'k', label='X')
        plt.xlim([0, dt * (N - 1)])
        plt.xlabel('t (days)')
        plt.legend(loc=1)
        # Plot 16 MODWPT vectors of coefficients at level 4
        for n in range(0, 16):
            D4n = D4[n]
            plt.subplot2grid((17, 1), (n, 0))
            plt.plot(np.arange(0, N), D4n, 'k', label='D4,' + str(n))
            plt.xlim([0, dt * (N - 1)])
            plt.legend(loc=1)
        plt.savefig('../tests/MODWPT/magS4096_D4.eps', format='eps')
        plt.close(1)

    # Compute LA8 MODWPT of the solar physics time series from WMTSA
    # See Figure 235 in WMTSA
    test2()
