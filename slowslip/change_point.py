"""
This module contains a function to compute the MODWT of the slow slip,
and see whether there are changes from one day to another:
    - 1. Look at the maxima and minima of the wavelet coefficient
    - 2. Look at inflexion points in the variance
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from math import cos, pi, sin, sqrt

import DWT
import MODWT
import date

# Preprocessing of the GPS data
# Choose the station
station = 'PGC5'
direction = 'lon'
dataset = 'cleaned'
filename = '../data/PANGA/' + dataset + '/' + station + '.' + direction
# Load the data
data = np.loadtxt(filename, skiprows=26)
time = data[:, 0]
disp = data[:, 1]
# Correct for the repeated value
dt = np.diff(time)
gap = np.where(dt < 1.0 / 365.0 - 0.001)[0]
time[gap[0] + 1] = time[gap[0] + 2]
time[gap[0] + 2] = 0.5 * (time[gap[0] + 2] + time[gap[0] + 3])
# Look for gaps greater than 2 days
days = 2
dt = np.diff(time)
gap = np.where(dt > days / 365.0 + 0.001)[0]
# Select a subset of the data without big gaps
ibegin = 2189
iend = 5102
time_sub = time[ibegin + 1 : iend + 1]
disp_sub = disp[ibegin + 1 : iend + 1]
# Fill the missing values by interpolation
dt = np.diff(time_sub)
gap = np.where(dt > 1.0 / 365.0 + 0.001)[0]
for i in range(0, len(gap)):
    time_sub = np.insert(time_sub, gap[i] + 1, time_sub[gap[i]] + 1.0 / 365.0)
    disp_sub = np.insert(disp_sub, gap[i] + 1, \
        0.5 * (disp_sub[gap[i]] + disp_sub[gap[i] + 1]))
    gap[i : ] = gap[i : ] + 1

# Preprocessing of the tremor data
# Load tremor data from the catalog downloaded from the PNSN
filename = '../data/timelags/08_01_2009_11_26_2014.txt'
day = np.loadtxt(filename, dtype=np.str, usecols=[0], skiprows=16)
hour = np.loadtxt(filename, dtype=np.str, usecols=[1], skiprows=16)
nt = np.shape(day)[0]
time_tremor = np.zeros(nt)
for i in range(0, nt):
    time_tremor[i] = date.string2day(day[i], hour[i])
location = np.loadtxt('../data/timelags/08_01_2009_11_26_2014.txt', \
    usecols=[2, 3], skiprows=16)
lat_tremor = location[:, 0]
lon_tremor = location[:, 1]
# Compute the tremor time for different distances from the GPS station
a = 6378.136
e = 0.006694470
lat0 = 48.65
lon0 = -123.45
dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / \
    sqrt(1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / \
    ((1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
dist = np.sqrt(np.power((lat_tremor - lat0) * dy, 2.0) + \
               np.power((lon_tremor - lon0) * dx, 2.0))
dists = [20.0, 40.0, 60.0, 80.0, 100.0]
times = []
for i in range(0, len(dists)):
    k = dist <= dists[i]
    times.append(time_tremor[k])

def compute_var(time, disp, J, name, time_ETS, times, dists):
    """
    """
    # Initialization
    g = DWT.get_scaling(name)
    L = len(g)
    dt = 1.0 / 365.0
    N = len(disp)
    # Compute MODWT
    (W, V) = MODWT.pyramid(disp, name, J)
    # Compute time derivative of wavelet coefficients
    D = []
    for j in range(0, J):
        Wj = W[j]
        Dj = np.diff(Wj)
        D.append(Dj)
    # Compute time when we have both:
    #     - 1. Change in the sign of the derivative
    #     - 2. The maximum or minimum is more than mean + 2 standard deviation
    M = []
    for j in range (0, J):
        Dj = D[j]
        index = np.sign(Dj[1 :]) - np.sign(Dj[0 : -1]) != 0
        index = np.insert(index, 0, False)
        index = np.append(index, False)
        Mj = time[(index)] # & (abs(Wj) > np.mean(Wj) + np.std(Wj))]
        M.append(Mj)
    # Compute the cumulative variance
    C = []
    for j in range(0, J):
        Wj = W[j]
        Cj = np.cumsum(np.power(Wj, 2.0))
        C.append(Cj)
    # Compute time when we have:
    #     - 1. Chqnge in the sign of the second derivative
    I = []
    for j in range(0, J):
        Cj = C[j]
        D1 = np.diff(Cj)
        D2 = np.diff(D1)
        index = np.sign(D2[1 :]) - np.sign(D2[0 : -1]) != 0
        index = np.insert(index, 0, False)
        index = np.append(index, False)
        index = np.append(index, False)
        Ij = time[(index)] + dt / 2.0
        I.append(Ij)
    # Figure 1: Look at the wavelet coefficients
    plt.figure(1, figsize=(15, (J + 2) * 4))
    # Plot tremor
    plt.subplot2grid((J + 2, 1), (J + 1, 0))
    for i in range(0, len(time_ETS)):
        plt.axvline(time_ETS[i], linewidth=2, color='grey')
    colors = cm.rainbow(np.linspace(0, 1, len(dists)))
    for time_dist, distance, c in zip(times, dists, colors):
        nt = np.shape(time_dist)[0]
        plt.plot(np.sort(time_dist), (1.0 / nt) * \
            np.arange(0, len(time_dist)), color=c, \
            label='distance <= {:2.0f} km'.format(distance))
    plt.xlim(np.min(time), np.max(time))
    plt.xlabel('Time (year)')
    plt.title('Cumulative number of tremor')
    plt.legend(loc=4)   
    # Plot data
    plt.subplot2grid((J + 2, 1), (J, 0))
    for i in range(0, len(time_ETS)):
        plt.axvline(time_ETS[i], linewidth=2, color='grey')
    plt.plot(time, disp, 'k', label='Data')
    plt.xlim(np.min(time), np.max(time))
    plt.legend(loc=1)
    # Plot wavelet coefficients at each level
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j - 1, 0))
        Wj = W[j]
        plt.plot(time, Wj, 'k-', label ='W' + str(j + 1))
        Mj = M[j]
        for i in range (0, len(Mj)):
            plt.axvline(Mj[i], linewidth=1, color='grey')
        Lj = (2 ** (j + 1) - 1) * (L - 1)
        if (Lj < N):
            plt.axvline(time[Lj], linewidth=2, color='red')
        plt.xlim(np.min(time), np.max(time))
        plt.legend(loc=1)
    # Save figure
    plt.suptitle('Wavelet coefficients', fontsize=24)
    filename = 'wavelet_' + name + '.eps'
    plt.savefig(filename, format='eps')
    plt.close(1)
    # Figure 2: Look at the derivative of the wavelet coefficients
    plt.figure(2, figsize=(15, (J + 2) * 4))
    # Plot tremor
    plt.subplot2grid((J + 2, 1), (J + 1, 0))
    for i in range(0, len(time_ETS)):
        plt.axvline(time_ETS[i], linewidth=2, color='grey')
    colors = cm.rainbow(np.linspace(0, 1, len(dists)))
    for time_dist, distance, c in zip(times, dists, colors):
        nt = np.shape(time_dist)[0]
        plt.plot(np.sort(time_dist), (1.0 / nt) * \
            np.arange(0, len(time_dist)), color=c, \
            label='distance <= {:2.0f} km'.format(distance))
    plt.xlim(np.min(time), np.max(time))
    plt.xlabel('Time (year)')
    plt.title('Cumulative number of tremor')
    plt.legend(loc=4)   
    # Plot data
    plt.subplot2grid((J + 2, 1), (J, 0))
    for i in range(0, len(time_ETS)):
        plt.axvline(time_ETS[i], linewidth=2, color='grey')
    plt.plot(time, disp, 'k', label='Data')
    plt.xlim(np.min(time), np.max(time))
    plt.legend(loc=1)
    # Plot derivative of the wavelet coefficients at each level
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j - 1, 0))
        Dj = D[j]
        plt.plot(time[0 : -1] + dt / 2.0, Dj, 'k-', label ='D' + str(j + 1))
        Mj = M[j]
        for i in range (0, len(Mj)):
            plt.axvline(Mj[i], linewidth=1, color='grey')
        Lj = (2 ** (j + 1) - 1) * (L - 1)
        if (Lj < N):
            plt.axvline(time[Lj], linewidth=2, color='red')
        plt.xlim(np.min(time), np.max(time))
        plt.legend(loc=1)
    # Save figure
    plt.suptitle('Derivative of the wavelet coefficients', fontsize=24)
    filename = 'derivative_' + name + '.eps'
    plt.savefig(filename, format='eps')
    plt.close(2)
    # Figure 3: Look at the cumulative variance
    plt.figure(3, figsize=(15, (J + 2) * 4))
    # Plot tremor
    plt.subplot2grid((J + 2, 1), (J + 1, 0))
    for i in range(0, len(time_ETS)):
        plt.axvline(time_ETS[i], linewidth=2, color='grey')
    colors = cm.rainbow(np.linspace(0, 1, len(dists)))
    for time_dist, distance, c in zip(times, dists, colors):
        nt = np.shape(time_dist)[0]
        plt.plot(np.sort(time_dist), (1.0 / nt) * \
            np.arange(0, len(time_dist)), color=c, \
            label='distance <= {:2.0f} km'.format(distance))
    plt.xlim(np.min(time), np.max(time))
    plt.xlabel('Time (year)')
    plt.title('Cumulative number of tremor')
    plt.legend(loc=4)   
    # Plot data
    plt.subplot2grid((J + 2, 1), (J, 0))
    for i in range(0, len(time_ETS)):
        plt.axvline(time_ETS[i], linewidth=2, color='grey')
    plt.plot(time, disp, 'k', label='Data')
    plt.xlim(np.min(time), np.max(time))
    plt.legend(loc=1)
    # Plot cumulative variance at each level
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j - 1, 0))
        Cj = C[j]
        plt.plot(time, Cj, 'k-', label ='C' + str(j + 1))
        Ij = I[j]
        for i in range (0, len(Ij)):
            plt.axvline(Ij[i], linewidth=1, color='grey')
        Lj = (2 ** (j + 1) - 1) * (L - 1)
        if (Lj < N):
            plt.axvline(time[Lj], linewidth=2, color='red')
        plt.xlim(np.min(time), np.max(time))
        plt.legend(loc=1)
    # Save figure
    plt.suptitle('Cumulative variance', fontsize=24)
    filename = 'variance_' + name + '.eps'
    plt.savefig(filename, format='eps')
    plt.close(3)
    return (M, I)

time_ETS = [2000.9583, 2002.1250, 2003.1250, 2004.0417, 2004.5417, 2005.7083, \
    2007.0833, 2008.375, 2009.375, 2010.6667, 2011.6667, 2012.7083, \
    2013.7500, 2014.9167, 2016.0000, 2017.1667]

(M, I) = compute_var(time_sub, disp_sub, 8, 'ID4', time_ETS, times, dists)
