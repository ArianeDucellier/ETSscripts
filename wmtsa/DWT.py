"""
This module contains functions to compute the DWT of a time series
We assume that we compute the partial DWT up to level J
and that the length of the time series is a multiple of 2**J
"""

import matplotlib.pyplot as plt
import numpy as np

def get_scaling(name):
    """
    Return the coefficients of the scaling filter
    
    Input:
        type name = string
        name = Name of the wavelet filter
    Output:
        type g = 1D numpy array
        g = Vector of coefficients of the scaling filter
    """
    if (name == 'Haar'):
        g = np.loadtxt('../ScalingCoefficients/Haar.dat')
    elif (name == 'D4'):
        g = np.loadtxt('../ScalingCoefficients/D4.dat')
    else:
        raise ValueError('{} has not been implemented yet'.format(name))
    return g

def get_wavelet(g):
    """
    Return the coefficients of the wavelet filter
    
    Input:
        type g = 1D numpy array
        g = Vector of coefficients of the scaling filter
    Output:
        type h = 1D numpy array
        h = Vector of coefficients of the wavelet filter
    """
    L = np.shape(g)[0]
    h = np.zeros(L)
    for l in range(0, L):
        h[l] = ((-1.0) ** l) * g[L - 1 - l]
    return h

def get_WV(h, g, X):
    """
    Level j of pyramid algorithm.
    Take V_(j-1) and return W_j and V_j
    
    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the scaling filter
        type X = 1D numpy array
        X = V_(j-1)
    Output:
        type W = 1D numpy array
        W = W_j
        type V = 1D numpy array
        V = V_j
    """
    N = np.shape(X)[0]
    assert (N % 2 == 0), 'Length of vector of scaling coefficients is odd'
    assert (np.shape(h)[0] == np.shape(g)[0]), 'Wavelet and scaling filters have different lengths'
    N2 = int(N / 2)
    W = np.zeros(N2)
    V = np.zeros(N2)
    L = np.shape(h)[0]
    for t in range(0, N2):
        for l in range(0, L):
            index = (2 * t + 1 - l) % N
            W[t] = W[t] + h[l] * X[index]
            V[t] = V[t] + g[l] * X[index]
    return (W, V)

def get_X(h, g, W, V):
    """
    Level j of inverse pyramid algorithm.
    Take W_j and V_j and return V_(j-1)
    
    Input:
        type h = 1D numpy array
        h = Vector of coefficients of the wavelet filter
        type g = 1D numpy array
        g = Vector of coefficients of the scaling filter
        type W = 1D numpy array
        W = W_j
        type V = 1D numpy array
        V = V_j
    Output:
        type X = 1D numpy array
        X = V_(j-1)
    """
    assert (np.shape(W)[0] == np.shape(V)[0]), 'Wj and Vj have different lengths'
    assert (np.shape(h)[0] == np.shape(g)[0]), 'Wavelet and scaling filters have different lengths'
    N = np.shape(W)[0]
    N2 = int(2 * N)
    X = np.zeros(N2)
    L = np.shape(h)[0]
    for t in range(0, N2):
        for l in range(0, L):
            index1 = (t + l) % N2
            if (index1 % 2 == 1):
                index2 = int((index1 - 1) / 2)
                X[t] = X[t] + h[l] * W[index2] + g[l] * V[index2]
    return X    
   
def pyramid(X, name, J):
    """
    Compute the DWT of X up to level J
    
    Input:
        type X = 1D numpy array
        X = Time series which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWT
    Output:
        type W = 1D numpy array
        W = Vector of DWT coefficients
    """
    assert (type(J) == int), 'Level of DWT must be an integer'
    assert (J >= 1), 'level of DWT must be higher or equal to 1'
    N = np.shape(X)[0]
    assert (N % (2 ** J) == 0), 'Length of time series is not a multiple of 2**J'
    g = get_scaling(name)
    h = get_wavelet(g)
    Vj = X
    W = np.zeros(N)
    indb = 0
    for j in range(1, (J + 1)):
        (Wj, Vj) = get_WV(h, g, Vj)
        inde = indb + int(N / (2 ** j))
        W[indb : inde] = Wj
        if (j == J):
            W[inde : N] = Vj
        indb = indb + int(N / (2 ** j))     
    return W

def inv_pyramid(W, name, J):
    """
    Compute the inverse DWT of W up to level J
    
    Input:
        type W = 1D numpy array
        W = Vector of DWT coefficients which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWT
    Output:
        type X = 1D numpy array
        X = Original time series
    """
    assert (type(J) == int), 'Level of DWT must be an integer'
    assert (J >= 1), 'level of DWT must be higher or equal to 1'
    N = np.shape(W)[0]
    assert (N % (2 ** J) == 0), 'Length of vector of DWT coefficients is not a multiple of 2**J'
    g = get_scaling(name)
    h = get_wavelet(g)
    for J in range(J, 0, -1):
        
    
if __name__ == '__main__':

    # Test 1
    # Compute DWT of the first time series from WMTSA
    # See upper plot of figure 62 in WMTSA
    X = np.loadtxt('../Data/ts16a.dat')
    N = np.shape(X)[0]
    W = pyramid(X, 'Haar', 4)
    plt.figure(1, figsize=(10, 5))
    for i in range(0, N):
        plt.plot(np.array([i, i]), np.array([0.0, W[i]]), 'k-')
        plt.plot(i, W[i], 'ko')
    plt.axhline(0, color='k')
    plt.xlabel('n', fontsize=24)
    xticks_labels = []
    for i in range(0, N):
        xticks_labels.append(str(i))
    plt.xticks(np.arange(0, N), xticks_labels)
    plt.ylim([-2, 2])
    plt.title('Haar DWT of first time series')
    plt.savefig('../Tests/ts16a.eps', format='eps')

    # Test 2
    # Compute DWT of the second time series from WMTSA
    # See lower plot of figure 62 in WMTSA
    X = np.loadtxt('../Data/ts16b.dat')
    N = np.shape(X)[0]
    W = pyramid(X, 'Haar', 4)
    plt.figure(2, figsize=(10, 5))
    for i in range(0, N):
        plt.plot(np.array([i, i]), np.array([0.0, W[i]]), 'k-')
        plt.plot(i, W[i], 'ko')
    plt.axhline(0, color='k')
    plt.xlabel('n', fontsize=24)
    xticks_labels = []
    for i in range(0, N):
        xticks_labels.append(str(i))
    plt.xticks(np.arange(0, N), xticks_labels)
    plt.ylim([-2, 2])
    plt.title('Haar DWT of second time series')
    plt.savefig('../Tests/ts16b.eps', format='eps')
