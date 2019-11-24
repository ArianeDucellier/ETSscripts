"""
Script to plot a vespagram-like figure of slow slip
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from math import cos, pi, sin, sqrt

import date
import DWT
from MODWT import get_DS, get_scaling, pyramid

#stations = ['ALBH', 'CHCM', 'COUP', 'PGC5', 'PTAA', 'SACH', 'SC02', 'SC03', 'SEQM', 'UFDA', 'FRID', 'PNCL', 'POUL', 'SQIM']
stations = ['ALBH', 'CHCM', 'COUP', 'PGC5', 'SC02', 'SC03', 'UFDA', 'FRID', 'PNCL', 'SQIM']
direction = 'lon'
dataset = 'cleaned'
#lats = [48.2323, 48.0106, 48.2173, 48.6483, 48.1168, 48.5667, 48.5462, 47.8166, 48.0914, 47.7550, 48.5352, 48.1014, 47.7546, 48.0823]
#lons = [-123.2915, -122.7759, -122.6856, -123.4511, -123.4943, -123.4207, -123.0076, -123.7057, -123.1135, -122.6670, -123.0180, -123.4152, -122.6672, -123.1020]
lats = [48.2323, 48.0106, 48.2173, 48.6483, 48.5462, 47.8166, 47.7550, 48.5352, 48.1014, 48.0823]
lons = [-123.2915, -122.7759, -122.6856, -123.4511, -123.0076, -123.7057, -122.6670, -123.0180, -123.4152, -123.1020]
lat0 = 48.1168
lon0 = -123.4943

events = [2009.78, 2009.86, 2010.06, 2010.085, 2010.21, 2010.24, 2010.65, 2011, 2011.07, 2011.2, 2011.32, 2011.38, 2011.6]

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

# GPS data
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

# Load tremor data from the catalog downloaded from the PNSN
filename = '../data/timelags/08_01_2009_09_05_2018.txt'
day = np.loadtxt(filename, dtype=np.str, usecols=[0], skiprows=16)
hour = np.loadtxt(filename, dtype=np.str, usecols=[1], skiprows=16)
nt = np.shape(day)[0]
time_tremor = np.zeros(nt)
for i in range(0, nt):
    time_tremor[i] = date.string2day(day[i], hour[i])
location = np.loadtxt('../data/timelags/08_01_2009_09_05_2018.txt', usecols=[2, 3], skiprows=16)
lat_tremor = location[:, 0]
lon_tremor = location[:, 1]

# Compute the tremor time for sources located less than 75 km from the origin
a = 6378.136
e = 0.006694470
dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
dist = np.sqrt(np.power((lat_tremor - lat0) * dy, 2.0) + np.power((lon_tremor - lon0) * dx, 2.0))
k = (dist <= 100.0)
time_tremor = time_tremor[k]

# Number of tremors per day
ntremor = np.zeros(len(times[0]))
for i in range(0, len(times[0])):
    for j in range (0, len(time_tremor)):
        if ((time_tremor[j] >= times[0][i] - 0.5 / 365.0) and \
            (time_tremor[j] <= times[0][i] + 0.5 / 365.0)):
            ntremor[i] = ntremor[i] + 1

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
            Dj_interp = np.interp(time + slowness[i] * (lat - lat0), time, Dj)
            vespagram[i, :] = vespagram[i, :] + Dj_interp

    plt.figure(1, figsize=(15, 10))
    plt.subplot(211)
    plt.contourf(times_subset[0], slowness, vespagram, cmap=plt.get_cmap('seismic'))
    plt.colorbar(orientation='horizontal')
    plt.subplot(212)
    plt.plot(times[0], ntremor, 'k', label='Number of tremor / day')
    plt.xlim(tmin, tmax)
    plt.savefig('vespagram/D' + str(j + 1) + '.eps', format='eps')
    plt.close(1)

    plt.figure(2, figsize=(15, 5))
    plt.plot(times_subset[0], vespagram[int((len(slowness) - 1) / 2), :], 'k')
    plt.savefig('vespagram/times_D' + str(j + 1) + '.eps', format='eps')
    plt.close(2)
