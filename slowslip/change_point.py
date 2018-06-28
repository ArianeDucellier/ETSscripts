"""
This module contains a function to compute the MODWT of the slow slip
every day, and see whether there are changes from one day to another
"""
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from math import cos, pi, sin, sqrt

import MODWT
import date

# Preprocessing
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
# Compute the tremor tme for different distances from the GPS station
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

def compute_last_coeff(time, disp, J, name, time_ETS, times, dists):
    """
    Add one value to the time series at each time step, then compute and plot
    the last wavelet coefficient of the MODWT for each time step.

    Input:
        type time = Numpy 1D array
        time = Day when data was recorded
        type disp = Numpy 1D array
        disp = Displacement measured at the GPS station
        type J = integer
        J = Level of the MODWT
        type name = string
        name = Wavelet filter (use extremal phase D4 to D20)
        type time_ETS = list of floats
        time_ETS = Time of main ETS events
        type times = list of 1D Numpy arrays
        times = Time at which there is a tremor at less than a given distance
        type dists = list of floats
        dists = Maximum distance from tremor to GPS station
    Output:
        None
    """
    # Initialization
    N = np.shape(disp)[0]
    L = int(name[1 :])
    t0 = time[0]
    dt = 1.0 / 365.0
    Wlast = np.zeros((N - 2, J))
    # Compute MODWT
    for i in range(2, N):
        (W0, V0) = MODWT.pyramid(disp[0 : i], name, J)
        for j in range(0, J):
            Wlast[i - 2, j] = W0[j][-1]
    # Plot figure
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
    plt.plot(t0 + dt * np.arange(0, N), disp, 'k', label='Data')
    plt.xlim(np.min(time), np.max(time))
    plt.legend(loc=1)
    # Plot last wavelet coefficient at each level
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j - 1, 0))
        for i in range(0, len(time_ETS)):
            plt.axvline(time_ETS[i], linewidth=2, color='grey')
        plt.plot(t0 + dt * np.arange(2, N), Wlast[:, j], 'k', \
            label = 'Level ' + str(j + 1))
        i0 = (2 ** (j + 1) - 1) * (L - 1)
        if (i0 < N):
            plt.axvline(t0 + dt * i0, linewidth=2, color='red')
        plt.xlim(np.min(time), np.max(time))
        plt.legend(loc=1)
    # Save figure
    plt.suptitle('Lats wavelet coefficient of the MODWT', fontsize=24)
    filename = 'lastcoeff_' + name + '.eps'
    plt.savefig(filename, format='eps')
    plt.close(1)

def slidingMODWT(time, disp, J, name, time_ETS, times, dists, ibegin, iend):
        # Initialization
    N = np.shape(disp)[0]
    t0 = time[0]
    dt = 1.0 / 365.0
    W = []
    for i in range(ibegin, iend):
        (W0, V0) = MODWT.pyramid(disp[0 : i], name, J)
        W.append(W0)
    # Plot figure
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
    plt.plot(t0 + dt * np.arange(0, N), disp, 'k', label='Data')
    plt.xlim(np.min(time), np.max(time))
    plt.legend(loc=1)
    colors = cm.rainbow(np.linspace(0, 1, iend - ibegin))
    # Plot wavelet coefficient at each level
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j - 1, 0))
        for i in range(0, len(time_ETS)):
            plt.axvline(time_ETS[i], linewidth=2, color='grey')
        for i in range(ibegin, iend):
            W0 = W[iend - i - 1]
            if (i == ibegin):
                plt.plot(t0 + dt * np.arange(0, len(W0[j])), W0[j], \
                    color=colors[iend - i - 1], label = 'W' + str(j + 1))
            else:
                plt.plot(t0 + dt * np.arange(0, len(W0[j])), W0[j], \
                    color=colors[iend - i - 1])
        plt.xlim(np.min(time), np.max(time))
        plt.legend(loc=1)
    # Save figure
    plt.suptitle('Wavelet coefficients of the MODWT', fontsize=24)
    filename = 'wavelet_' + name + '.eps'
    plt.savefig(filename, format='eps')
    plt.close(1)

def deriveMODWT(time, disp, J, name, time_ETS, times, dists):
    N = np.shape(disp)[0]
    t0 = time[0]
    dt = 1.0 / 365.0
    (W, V) = MODWT.pyramid(disp, name, J)
    chsgn = []
    # Plot figure
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
    plt.plot(t0 + dt * np.arange(0, N), disp, 'k', label='Data')
    plt.xlim(np.min(time), np.max(time))
    plt.legend(loc=1)
    # Plot wavelet coefficient at each level
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j - 1, 0))
        Wj = W[j]
        deriv = np.diff(Wj)
        index = np.sign(deriv[1 :]) - np.sign(deriv[0 : -1]) != 0
        index = np.insert(index, 0, False)
        index = np.append(index, False)
        chsgn0 = time[(index) & (abs(Wj) > np.mean(Wj) + np.std(Wj))]
        chsgn.append(chsgn0)
        plt.plot(time, Wj, 'k-', label ='W' + str(j + 1))
        for i in range (0, len(chsgn0)):
            plt.axvline(chsgn0[i], linewidth=1, color='grey')
        plt.xlim(np.min(time), np.max(time))
        plt.legend(loc=1)
    # Save figure
    plt.suptitle('Wavelet coefficients of the MODWT', fontsize=24)
    filename = 'wavelet_' + name + '.eps'
    plt.savefig(filename, format='eps')
    plt.close(1)
    return chsgn

time_ETS = [2000.9583, 2002.1250, 2003.1250, 2004.0417, 2004.5417, 2005.7083, \
    2007.0833, 2008.375, 2009.375, 2010.6667, 2011.6667, 2012.7083, \
    2013.7500, 2014.9167, 2016.0000, 2017.1667]

#compute_last_coeff(time_sub, disp_sub, 8, 'D14', time_ETS, times, dists)

#slidingMODWT(time_sub, disp_sub, 8, 'D4', time_ETS, times, dists, 2466, 2919)

chsgn = deriveMODWT(time_sub, disp_sub, 8, 'LA8', time_ETS, times, dists)
