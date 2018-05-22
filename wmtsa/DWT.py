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
        g = np.loadtxt('../data/scalingcoeff/Haar.dat')
    elif (name == 'D4'):
        g = np.loadtxt('../data/scalingcoeff/D4.dat')
    elif (name == 'D6'):
        g = np.loadtxt('../data/scalingcoeff/D6.dat')
    elif (name == 'D8'):
        g = np.loadtxt('../data/scalingcoeff/D8.dat')
    elif (name == 'D10'):
        g = np.loadtxt('../data/scalingcoeff/D10.dat')
    elif (name == 'D12'):
        g = np.loadtxt('../data/scalingcoeff/D12.dat')
    elif (name == 'D14'):
        g = np.loadtxt('../data/scalingcoeff/D14.dat')
    elif (name == 'D16'):
        g = np.loadtxt('../data/scalingcoeff/D16.dat')
    elif (name == 'D18'):
        g = np.loadtxt('../data/scalingcoeff/D18.dat')
    elif (name == 'D20'):
        g = np.loadtxt('../data/scalingcoeff/D20.dat')
    elif (name == 'LA8'):
        g = np.loadtxt('../data/scalingcoeff/LA8.dat')
    elif (name == 'LA10'):
        g = np.loadtxt('../data/scalingcoeff/LA10.dat')
    elif (name == 'LA12'):
        g = np.loadtxt('../data/scalingcoeff/LA12.dat')
    elif (name == 'LA14'):
        g = np.loadtxt('../data/scalingcoeff/LA14.dat')
    elif (name == 'LA16'):
        g = np.loadtxt('../data/scalingcoeff/LA16.dat')
    elif (name == 'LA18'):
        g = np.loadtxt('../data/scalingcoeff/LA18.dat')
    elif (name == 'LA20'):
        g = np.loadtxt('../data/scalingcoeff/LA20.dat')
    elif (name == 'C6'):
        g = np.loadtxt('../data/scalingcoeff/C6.dat')
    elif (name == 'C12'):
        g = np.loadtxt('../data/scalingcoeff/C12.dat')
    elif (name == 'C18'):
        g = np.loadtxt('../data/scalingcoeff/C18.dat')
    elif (name == 'C24'):
        g = np.loadtxt('../data/scalingcoeff/C24.dat')
    elif (name == 'C30'):
        g = np.loadtxt('../data/scalingcoeff/C30.dat')
    elif (name == 'BL14'):
        g = np.loadtxt('../data/scalingcoeff/BL14.dat')
    elif (name == 'BL18'):
        g = np.loadtxt('../data/scalingcoeff/BL18.dat')
    elif (name == 'BL20'):
        g = np.loadtxt('../data/scalingcoeff/BL20.dat')
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
    assert (N % 2 == 0), \
        'Length of vector of scaling coefficients is odd'
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
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
    assert (np.shape(W)[0] == np.shape(V)[0]), \
        'Wj and Vj have different lengths'
    assert (np.shape(h)[0] == np.shape(g)[0]), \
        'Wavelet and scaling filters have different lengths'
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
    assert (N % (2 ** J) == 0), \
        'Length of time series is not a multiple of 2**J'
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
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    N = np.shape(W)[0]
    assert (N % (2 ** J) == 0), \
        'Length of vector of DWT coefficients is not a multiple of 2**J'
    g = get_scaling(name)
    h = get_wavelet(g)
    Vj = W[-int(N / (2 ** J)) : ]
    for j in range(J, 0, -1):
        Wj = W[-int(N / (2 ** (j - 1))) : -int(N / 2 ** j)]
        Vj = get_X(h, g, Wj, Vj)
    X = Vj
    return X

def get_DS(X, W, name, J):
    """
    Compute the details and the smooths of the time series
    using the DWT coefficients

    Input:
        type X = 1D numpy array
        X =  Time series which length is a multiple of 2**J
        type W = 1D numpy array
        W = Vector of DWT cofficients which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWT
    Output:
        type D = list of 1D numpy arrays
        D = List of details [D1, D2, ... , DJ]
        type S = list of 1D numpy arrays
        S = List of smooths [S0, S1, S2, ... , SJ]
    """
    assert (type(J) == int), \
        'Level of DWT must be an integer'
    assert (J >= 1), \
        'Level of DWT must be higher or equal to 1'
    N = np.shape(W)[0]
    assert (N % (2 ** J) == 0), \
        'Length of vector of DWT coefficients is not a multiple of 2**J'
    # Compute details
    D = []
    for j in range(1, J + 1):
        Wj = np.zeros(N)
        Wj[-int(N / (2 ** (j - 1))) : -int(N / 2 ** j)] = \
            W[-int(N / (2 ** (j - 1))) : -int(N / 2 ** j)]
        Dj = inv_pyramid(Wj, name, J)
        D.append(Dj)
    # Compute smooths
    S = [X]
    for j in range(0, J):
        Sj = S[-1] - D[j]
        S.append(Sj)
    return (D, S)

def NPES(W):
    """ Compute the normalized partial energy sequence
    of a time series

    Input:
        type W = 1D numpy array
        W = Time series (or wavelet coefficients)
    Output:
        type C = 1D numpy array
        C = NPES
    """
    N = np.shape(W)[0]
    U = np.flip(np.sort(np.power(np.abs(W), 2.0)), 0)
    C = np.zeros(N)
    for i in range(0, N):
        C[i] = np.sum(U[0 : i + 1]) / np.sum(U)
    return C

def get_nu(name, J):
    """ Compute the phase shift for LA filters

    Input:
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = maximum level for DWT
    Output:
        type nuH = list of J values
        nuH = Shifts for the wavelet filter
        type nuG = list of J values
        nuG = Shifts for the scaling filter
    """
    assert (name[0 : 2] == 'LA'), \
        'Wavelet filter must be Daubechies least asymmetric'
    L = int(name[2 : ])
    if (L == 14):
        nu = int(- L / 2 + 2)
    elif (int(L / 2) % 2 == 0):
        nu = int(- L / 2 + 1)
    else:
        nu = int(- L / 2)
    nuH = []
    nuG = []
    for j in range(1, J + 1):
        Lj = int((2 ** j - 1) * (L - 1) + 1)
        nuH.append(- int(Lj / 2 + L / 2 + nu - 1))
        nuG.append(- int((Lj - 1) * nu / (L - 1)))
    return (nuH, nuG)

def plot_W(X, name, J):
    """Plot the wavelet coefficients and the data
    
    Input:
        type X = 1D numpy array
        X = Time series which length is a multiple of 2**J
        type name = string
        name = Name of the wavelet filter
        type J = integer
        J = Level of partial DWT
    """
    W = pyramid(X, name, J)
    N = np.shape(X)[0]
    plt.figure(1, figsize=(3 * (J + 2), 15))
    plt.subplot2grid((J + 2, 1), (J + 1, 0))
    plt.plot(np.arange(0, N), X, 'k')
    for j in range(1, J + 1):
        Wj = W[-int(N / (2 ** (j - 1))) : -int(N / 2 ** j)]
        plt.subplot2grid((J + 2, 1), (J - j + 1, 0))
        for i in range(0, int(N / 2 ** j)):
            plt.plot((i * 2 ** j, i * 2 ** j), (0.0, Wj[i]), 'k')
    Vj = W[-int(N / (2 ** J)) : ]
    plt.subplot2grid((J + 2, 1), (0, 0))
    plt.plot((2 ** J) * np.arange(0, int(N / 2 ** J)), Vj, 'k')

if __name__ == '__main__':

    # Test 1
    def test1(name_input, name_output, title):
        """
        Reproduce plots of Figure 62 from WMTSA

        Input:
            type name_input = string
            name_input = Name of file containing time series
            type name_output = string
            name_output = Name of image file containing the plot
            type title = string
            title = Title to add to the plot
        Output:
            None
        """
        X = np.loadtxt('../tests/' + name_input)
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
        plt.title(title)
        plt.savefig('../tests/' + name_output, format='eps')
        plt.close()

    # Compute DWT of the first time series from WMTSA
    # See upper plot of Figure 62 in WMTSA
    test1('ts16a.dat', 'ts16a_W.eps', \
        'Haar DWT of first time series')

    # Compute DWT of the second time series from WMTSA
    # See lower plot of Figure 62 in WMTSA
    test1('ts16b.dat', 'ts16b_W.eps', \
        'Haar DWT of second time series')

    # Test 2
    def test2(name_input, name_output, title, name_filter):
        """
        Reproduce plots of Figures 64 and 65 from WMTSA

        Input:
            type name_input = string
            name_input = Name of file containing time series
            type name_output = string
            name_output = Name of image file containing the plot
            type title = string
            title = Title to add to the plot
            type name_filter = string
            name_filter = Name of the wavelet filter
        Output:
            None
        """
        X = np.loadtxt('../tests/' + name_input)
        N = np.shape(X)[0]
        W = pyramid(X, name_filter, 4)
        (D, S) = get_DS(X, W, name_filter, 4)
        xticks_labels = []
        for i in range(0, N):
            xticks_labels.append(str(i))
        plt.figure(3, figsize=(30, 25))
        # Plot details
        for j in range(1, 5):
            plt.subplot2grid((5, 3), (j, 0))
            for i in range(0, N):
                plt.plot(np.array([i, i]), np.array([0.0, D[j - 1][i]]), 'k-')
                plt.plot(i, D[j - 1][i], 'ko')
            plt.axhline(0, color='k')
            plt.xticks(np.arange(0, N), xticks_labels)
            plt.ylim([-2, 2])
            if (j == 4):
                plt.xlabel('n', fontsize=24)
        # Plot smooths
        for j in range(0, 5):
            plt.subplot2grid((5, 3), (j, 1))
            for i in range(0, N):
                plt.plot(np.array([i, i]), np.array([0.0, S[j][i]]), 'k-')
                plt.plot(i, S[j][i], 'ko')
            plt.axhline(0, color='k')
            plt.xticks(np.arange(0, N), xticks_labels)
            plt.ylim([-2, 2])
            if (j == 4):
                plt.xlabel('n', fontsize=24)
        # Plot roughs
        for j in range(0, 5):
            plt.subplot2grid((5, 3), (j, 2))
            for i in range(0, N):
                plt.plot(np.array([i, i]), \
                    np.array([0.0, X[i] - S[j][i]]), 'k-')
                plt.plot(i, X[i] - S[j][i], 'ko')
            plt.axhline(0, color='k')
            plt.xticks(np.arange(0, N), xticks_labels)
            plt.ylim([-2, 2])
            if (j == 4):
                plt.xlabel('n', fontsize=24)
        plt.suptitle(title, fontsize=30)
        plt.savefig('../tests/' + name_output, format='eps')
        plt.close() 

    # Compute details, smooths and roughs of the first time series
    # from WMTSA using the Haar wavelet filter
    # See upper plot of Figure 64 in WMTSA
    test2('ts16a.dat', 'ts16a_DSR_Haar.eps', \
          'Haar DWT of first time series', 'Haar')

    # Compute details, smooths and roughs of the second time series
    # from WMTSA using the Haar wavelet filter
    # See lower plot of Figure 64 in WMTSA
    test2('ts16b.dat', 'ts16b_DSR_Haar.eps', \
          'Haar DWT of second time series', 'Haar')

    # Compute details, smooths and roughs of the first time series
    # from WMTSA using the D(4) wavelet filter
    # See upper plot of Figure 65 in WMTSA
    test2('ts16a.dat', 'ts16a_DSR_D4.eps', \
          'D(4) DWT of first time series', 'D4')

    # Compute details, smooths and roughs of the second time series
    # from WMTSA using the D(4) wavelet filter
    # See lower plot of Figure 65 in WMTSA
    test2('ts16b.dat', 'ts16b_DSR_D4.eps', \
          'D(4) DWT of second time series', 'D4')
