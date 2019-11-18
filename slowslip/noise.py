""" Script to look at MODWT of noise """

import matplotlib.pyplot as plt
import numpy as np

import DWT, MODWT
from MODWT import get_DS, pyramid

# Set random seed
np.random.seed(0)

# Generate random noise time series
N = 2048
X = noise = np.random.normal(0.0, 1.0, N)

# Parameters
names = ['Haar', 'D4', 'D6', 'D8', 'D10', 'D12', 'D14', 'D16', 'D18', 'D20', \
    'LA8', 'LA10', 'LA12', 'LA14', 'LA16', 'LA18', 'LA20', 'C6', 'C12', \
    'C18', 'C24', 'C30', 'BL14', 'BL18', 'BL20']
J = 6

# Loop on wavelet filters
for name in names:
    # Length of filter
    g = MODWT.get_scaling(name)
    L = len(g)
    # MODWT
    (W, V) = pyramid(X, name, J)
    if ((name[0 : 2] == 'LA') or (name[0 : 1] == 'C')):
        (nuH, nuG) = DWT.get_nu(name, J)
    (D, S) = get_DS(X, W, name, J)
    # Maxima wavelet coefficients
    maxvalue = [np.max(np.abs(X))]
    for j in range(0, J):
        maxvalue.append(np.max(np.abs(W[j])))
    maxvalue.append(np.max(np.abs(V)))
    ymax = max(maxvalue)
    # Plot wavelet coefficients
    plt.figure(1, figsize=(15, (J + 2) * 4))
    plt.subplot2grid((J + 2, 1), (J + 1, 0))
    t = np.arange(0, N)
    plt.plot(t, X, 'k', label='X')
    plt.xlim([np.min(t), np.max(t)])
    plt.ylim([- ymax, ymax])
    plt.legend(loc=1)
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j, 0))
        if ((name[0 : 2] == 'LA') or (name[0 : 1] == 'C')):
            plt.plot(t, np.roll(W[j], nuH[j]), 'k', label='T' + str(nuH[j]) + 'W' + str(j + 1))
        else:
            plt.plot(t, W[j], 'k', label='W' + str(j + 1))
        Lj = (2 ** (j + 1) - 1) * (L - 1) + 1
        if ((name[0 : 2] == 'LA') or (name[0 : 1] == 'C')):
            plt.axvline(Lj - 2 - abs(nuH[j]), linewidth=1, color='red')
            plt.axvline(N - abs(nuH[j]), linewidth=1, color='red')
        else:
            plt.axvline(Lj - 2, linewidth=1, color='red')
            plt.axvline(N - 1, linewidth=1, color='red')
        plt.xlim([np.min(t), np.max(t)])
        plt.ylim([- ymax, ymax])
        plt.legend(loc=1)
    plt.subplot2grid((J + 2, 1), (0, 0))
    if ((name[0 : 2] == 'LA') or (name[0 : 1] == 'C')):
        plt.plot(t, np.roll(V, nuG[J - 1]), 'k', label='T' + str(nuG[J - 1]) + 'V' + str(J))
    else:
        plt.plot(t, V, 'k', label='V' + str(J))
    Lj = (2 ** J - 1) * (L - 1) + 1
    if ((name[0 : 2] == 'LA') or (name[0 : 1] == 'C')):
        plt.axvline(Lj - 2 - abs(nuG[J - 1]), linewidth=1, color='red')
        plt.axvline(N - abs(nuG[J - 1]), linewidth=1, color='red')
    else:
        plt.axvline(Lj - 2, linewidth=1, color='red')
        plt.axvline(N - 1, linewidth=1, color='red')
    plt.xlim([np.min(t), np.max(t)])
    plt.ylim([- ymax, ymax])
    plt.legend(loc=1)
    plt.title('MODWT of random noise')
    plt.savefig('noise/' + name + '_WV.eps', format='eps')
    plt.close(1)
    # Maxima details
    maxvalue = [np.max(np.abs(X))]
    for j in range(0, J):
        maxvalue.append(np.max(np.abs(D[j])))
    maxvalue.append(np.max(np.abs(S[J])))
    ymax = max(maxvalue)
    # Plot details and smooth
    plt.figure(2, figsize=(15, (J + 2) * 4))
    plt.subplot2grid((J + 2, 1), (J + 1, 0))
    t = np.arange(0, N)
    plt.plot(t, X, 'k', label='X')
    plt.xlim([np.min(t), np.max(t)])
    plt.ylim([- ymax, ymax])
    plt.legend(loc=1)
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j, 0))
        plt.plot(t, D[j], 'k', label='D' + str(j + 1))
        Lj = (2 ** (j + 1) - 1) * (L - 1) + 1
        plt.axvline(Lj - 2, linewidth=1, color='red')
        plt.axvline(N - Lj + 1, linewidth=1, color='red')
        plt.xlim([np.min(t), np.max(t)])
        plt.ylim([- ymax, ymax])
        plt.legend(loc=1)
    plt.subplot2grid((J + 2, 1), (0, 0))
    plt.plot(t, S[J], 'k', label='S' + str(J))
    Lj = (2 ** 6 - 1) * (L - 1) + 1
    plt.axvline(Lj - 2, linewidth=1, color='red')
    plt.axvline(N - Lj + 1, linewidth=1, color='red')
    plt.xlim([np.min(t), np.max(t)])
    plt.ylim([- ymax, ymax])
    plt.legend(loc=1)
    plt.title('MODWT of random noise')
    plt.savefig('noise/' + name + '_DS.eps', format='eps')
    plt.close(2)
