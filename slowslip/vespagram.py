"""
Script to plot a vespagram-like figure of slow slip
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import DWT
from MODWT import get_DS, get_scaling, pyramid

stations = ['CHCM', 'CLRS', 'CUSH', 'ELSR', 'ESM1', 'FRID', 'PGC5', 'PNCL', 'SQIM', 'UFDA']
direction = 'lon'
dataset = 'cleaned'
lats = [48.8203, 47.4233, 48.5350, 48.6485, 48.1014, 48.1168, 48.0825]
lons = [-124.1309, -123.2199, -123.0180, -123.4511, -123.4152, -123.4943, \
    -123.1018]

# Parameters
name = 'LA8'
g = get_scaling(name)
L = len(g)
J = 6
(nuH, nuG) = DWT.get_nu(name, J)
slowness = np.arange(-0.1, 0.105, 0.005)

times = []
disps = []
Ws = []
Vs = []
Ds = []
Ss = []

# Data
for station in stations:
    filename = '../data/PANGA/' + dataset + '/' + station + '.' + direction
    # Load the data
    data = np.loadtxt(filename, skiprows=26)
    time = data[:, 0]
    disp = data[:, 1]
    error = data[:, 2]
    sigma = np.std(disp)
    # Correct for the repeated values
    dt = np.diff(time)
    gap = np.where(dt < 1.0 / 365.0 - 0.0001)[0]
    for i in range(0, len(gap)):
        if ((time[gap[i] + 2] - time[gap[i] + 1] > 2.0 / 365.0 - 0.0001) \
        and (time[gap[i] + 2] - time[gap[i] + 1] < 2.0 / 365.0 + 0.0001)):
            time[gap[i] + 1] = 0.5 * (time[gap[i] + 2] + time[gap[i]])
        elif ((time[gap[i] + 2] - time[gap[i] + 1] > 1.0 / 365.0 - 0.0001) \
          and (time[gap[i] + 2] - time[gap[i] + 1] < 1.0 / 365.0 + 0.0001) \
          and (time[gap[i] + 3] - time[gap[i] + 2] > 2.0 / 365.0 - 0.0001) \
          and (time[gap[i] + 3] - time[gap[i] + 2] < 2.0 / 365.0 + 0.0001)):
            time[gap[i] + 1] = time[gap[i] + 2]
            time[gap[i] + 2] = 0.5 * (time[gap[i] + 2] + time[gap[i] + 3])
    # Look for gaps greater than 1 day
    days = 2
    dt = np.diff(time)
    gap = np.where(dt > days / 365.0 - 0.0001)[0]
    duration = np.round((time[gap + 1] - time[gap]) * 365).astype(np.int)
    # Fill the gaps by interpolation
    for j in range(0, len(gap)):
        time = np.insert(time, gap[j] + 1, \
            time[gap[j]] + np.arange(1, duration[j]) / 365.0)
        disp = np.insert(disp, gap[j] + 1, \
            np.random.normal(0.0, sigma, duration[j] - 1))
        gap[j + 1 : ] = gap[j + 1 : ] + duration[j] - 1
    times.append(time)
    disps.append(disp)
    # MODWT
    [W, V] = pyramid(disp, name, J)
    Ws.append(W)
    Vs.append(V)
    (D, S) = get_DS(disp, W, name, J)
    Ds.append(D)
    Ss.append(S)

# Subset
tbegin = []
tend = []
for time in times:
    tbegin.append(np.min(time))
    tend.append(np.max(time))
tmin = max(tbegin)
tmax = min(tend)

# Loop on details
for j in range(0, J):
    times_subset = []
    Dj_subset = []
    for (time, D) in zip(times, Ds):
        ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
        iend = np.where(np.abs(time - tmax) < 0.001)[0]
        times_subset.append(time[ibegin[0] : iend[0] + 1])
        Dj_subset.append(D[j][ibegin[0] : iend[0] + 1])

    # Stack
    vespagram = np.zeros((len(slowness), len(times_subset[0])))
    latmin = min(lats)
    for i in range(0, len(slowness)):
        for (time, Dj, lat) in zip(times_subset, Dj_subset, lats):
            Dj_interp = np.interp(time + slowness[i] * (lat - latmin), time, Dj)
            vespagram[i, :] = vespagram[i, :] + Dj_interp

    plt.figure(1, figsize=(15, 5))
    plt.contourf(times_subset[0], slowness, vespagram, cmap=plt.get_cmap('seismic'))
    plt.colorbar(orientation='horizontal')
    plt.savefig('vespagram/D' + str(j + 1) + '.eps', format='eps')
    plt.close(1)

    plt.figure(2, figsize=(15, 5))
    plt.plot(times_subset[0], vespagram[int((len(slowness) - 1) / 2), :], 'k')
    plt.savefig('vespagram/times_D' + str(j + 1) + '.eps', format='eps')
    plt.close(2)
