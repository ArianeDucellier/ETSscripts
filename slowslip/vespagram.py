"""
Script to plot a vespagram-like figure of slow slip
"""

import datetime
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from math import cos, pi, sin, sqrt
from scipy.io import loadmat

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

#good_events = [2009.86448, 2010.0616, 2010.08624, 2010.20945, 2011.06092, 2011.32101, 2011.46886, 2011.95072]

#good_events = [2009.86174, 2010.05886, 2010.08624, 2010.20671, 2011.06092, 2011.32101, 2011.46886, 2011.95072]
good_events = [2010.05886, 2010.08624, 2010.20671, 2010.23682]

#bad_events = [2009.77687, 2010.23682, 2010.6475, 2011.00068, 2011.19507, 2011.38398, 2011.44422, 2011.49076, 2011.95072]

bad_events = [2007.78097, 2008.07392, 2008.1807, 2008.35592, 2009.10883, 2009.19644, 2009.3525, 2009.77413, 2010.23682, 2010.63655, 2011.00068, 2011.19507, 2011.38398, 2011.44148, 2011.49624, 2011.6167, 2011.95072]

t1 = 2007.79192
t2 = 2008.05202
t3 = 2008.2026
t4 = 2008.37235
t5 = 2008.47091
t6 = 2008.69268
t7 = 2008.84873
t8 = 2009.01574
t9 = 2009.33607
t10 = 2010.05339
t11 = 2010.20671
t12 = 2010.61739
t13 = 2010.81177
t14 = 2010.90212
t15 = 2011.22245
t16 = 2011.61396
t17 = 2011.90965
t18 = 2012.06023

t_maxima = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18]

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
#filename = '../data/timelags/08_01_2009_09_05_2018.txt'
#day = np.loadtxt(filename, dtype=np.str, usecols=[0], skiprows=16)
#hour = np.loadtxt(filename, dtype=np.str, usecols=[1], skiprows=16)
#nt = np.shape(day)[0]
#time_tremor = np.zeros(nt)
#for i in range(0, nt):
#    time_tremor[i] = date.string2day(day[i], hour[i])
#location = np.loadtxt('../data/timelags/08_01_2009_09_05_2018.txt', usecols=[2, 3], skiprows=16)
#lat_tremor = location[:, 0]
#lon_tremor = location[:, 1]

#filename = '../data/timelags/SummaryLatest.mat'
#data = loadmat(filename)
#TREMall = data['TREMall']
#day = TREMall[0][0][0]
#latitude = TREMall[0][0][1]
#longitude = TREMall[0][0][2]
#nt = np.shape(day)[0]
#time_tremor = np.zeros(nt)
#lat_tremor = np.zeros(nt)
#lon_tremor = np.zeros(nt)
#for i in range(0, nt):
#    myday = date.matlab2ymdhms(day[i][0])
#    t1 = datetime.date(myday[0], myday[1], myday[2])
#    t2 = datetime.date(myday[0], 1, 1)
#    time_tremor[i] = myday[0] + (t1 - t2).days / 365.25
#    lat_tremor[i] = latitude[i][0]
#    lon_tremor[i] = longitude[i][0]

# Compute the tremor time for sources located less than 75 km from the origin
a = 6378.136
e = 0.006694470
dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
#dist = np.sqrt(np.power((lat_tremor - lat0) * dy, 2.0) + np.power((lon_tremor - lon0) * dx, 2.0))
#k = (dist <= 100.0)
#time_tremor = time_tremor[k]

# Number of tremors per day
#ntremor = np.zeros(len(times[0]))
#for i in range(0, len(times[0])):
#    for j in range (0, len(time_tremor)):
#        if ((time_tremor[j] >= times[0][i] - 0.5 / 365.0) and \
#            (time_tremor[j] <= times[0][i] + 0.5 / 365.0)):
#            ntremor[i] = ntremor[i] + 1

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
    if (j == 2):
        vesp3 = vespagram
    if (j == 3):
        vesp4 = vespagram
    if (j == 4):
        vesp5 = vespagram
    if (j == 5):
        vesp6 = vespagram

#    plt.figure(1, figsize=(15, 10))
    plt.figure(1, figsize=(15, 5))
#    plt.subplot(211)
    plt.contourf(times_subset[0], slowness * 365.25 / dy, vespagram, cmap=plt.get_cmap('seismic'), vmin=-10.0, vmax=10.0)
#    for event in good_events:
#        plt.axvline(event, linewidth=1, color='grey')
#    for event in bad_events:
#        plt.axvline(event, linewidth=1, color='grey')
    for tm in t_maxima:
        plt.axvline(tm, linewidth=1, color='k')
    plt.xlabel('Time (year)')
    plt.ylabel('Slowness (day / km)')
    plt.colorbar(orientation='horizontal')
#    plt.subplot(212)
#    plt.axvline(good_events[0], linewidth=1, color='red')
#    plt.axvline(good_events[0], linewidth=1, color='green')
#    plt.axvline(good_events[1], linewidth=1, color='blue')
#    plt.axvline(good_events[2], linewidth=1, color='magenta')
#    plt.axvline(good_events[3], linewidth=1, color='cyan')
#    plt.axvline(good_events[5], linewidth=1, color='darkorchid')
#    plt.axvline(good_events[6], linewidth=1, color='sienna')
#    plt.axvline(good_events[7], linewidth=1, color='tan')
#    for event in bad_events:
#        plt.axvline(event, linewidth=1, color='grey')
#    plt.plot(times[0], ntremor, 'k')
#    plt.xlim(tmin, tmax)
#    plt.xlabel('Time (year)')
#    plt.ylabel('Number of tremor per day')
    plt.savefig('vespagram/D' + str(j + 1) + '.eps', format='eps')
    plt.close(1)

#plt.figure(1, figsize=(15, 10))
plt.figure(1, figsize=(15, 5))
#plt.subplot(211)
plt.contourf(times_subset[0], slowness * 365.25 / dy, vesp3 + vesp4, cmap=plt.get_cmap('seismic'), vmin=-20.0, vmax=20.0)
#for event in good_events:
#    plt.axvline(event, linewidth=1, color='grey')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
for tm in t_maxima:
    plt.axvline(tm, linewidth=1, color='k')
plt.xlabel('Time (year)')
plt.ylabel('Slowness (day / km)')
plt.colorbar(orientation='horizontal')
#plt.subplot(212)
#plt.axvline(good_events[0], linewidth=1, color='red')
#plt.axvline(good_events[1], linewidth=1, color='green')
#plt.axvline(good_events[2], linewidth=1, color='blue')
#plt.axvline(good_events[3], linewidth=1, color='magenta')
#plt.axvline(good_events[4], linewidth=1, color='cyan')
#plt.axvline(good_events[5], linewidth=1, color='darkorchid')
#plt.axvline(good_events[6], linewidth=1, color='sienna')
#plt.axvline(good_events[7], linewidth=1, color='tan')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
#plt.plot(times[0], ntremor, 'k')
#plt.xlim(tmin, tmax)
#plt.xlabel('Time (year)')
#plt.ylabel('Number of tremor per day')
plt.savefig('vespagram/D34.eps', format='eps')
plt.close(1)

#plt.figure(1, figsize=(15, 10))
plt.figure(1, figsize=(15, 5))
#plt.subplot(211)
plt.contourf(times_subset[0], slowness * 365.25 / dy, vesp3 + vesp4 + vesp5, cmap=plt.get_cmap('seismic'), vmin=-20.0, vmax=20.0)
#for event in good_events:
#    plt.axvline(event, linewidth=1, color='grey')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
for tm in t_maxima:
    plt.axvline(tm, linewidth=1, color='k')
plt.xlabel('Time (year)')
plt.ylabel('Slowness (day / km)')
plt.colorbar(orientation='horizontal')
#plt.subplot(212)
#plt.axvline(good_events[0], linewidth=1, color='red')
#plt.axvline(good_events[1], linewidth=1, color='green')
#plt.axvline(good_events[2], linewidth=1, color='blue')
#plt.axvline(good_events[3], linewidth=1, color='magenta')
#plt.axvline(good_events[4], linewidth=1, color='cyan')
#plt.axvline(good_events[5], linewidth=1, color='darkorchid')
#plt.axvline(good_events[6], linewidth=1, color='sienna')
#plt.axvline(good_events[7], linewidth=1, color='tan')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
#plt.plot(times[0], ntremor, 'k')
#plt.xlim(tmin, tmax)
#plt.xlabel('Time (year)')
#plt.ylabel('Number of tremor per day')
plt.savefig('vespagram/D345.eps', format='eps')
plt.close(1)

#plt.figure(1, figsize=(15, 10))
plt.figure(1, figsize=(15, 5))
#plt.subplot(211)
plt.contourf(times_subset[0], slowness * 365.25 / dy, vesp3 + vesp4 + vesp5 + vesp6, cmap=plt.get_cmap('seismic'), vmin=-30.0, vmax=30.0)
#for event in good_events:
#    plt.axvline(event, linewidth=1, color='grey')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
for tm in t_maxima:
    plt.axvline(tm, linewidth=1, color='k')
plt.xlabel('Time (year)')
plt.ylabel('Slowness (day / km)')
plt.colorbar(orientation='horizontal')
#plt.subplot(212)
#plt.axvline(good_events[0], linewidth=1, color='red')
#plt.axvline(good_events[1], linewidth=1, color='green')
#plt.axvline(good_events[2], linewidth=1, color='blue')
#plt.axvline(good_events[3], linewidth=1, color='magenta')
#plt.axvline(good_events[4], linewidth=1, color='cyan')
#plt.axvline(good_events[5], linewidth=1, color='darkorchid')
#plt.axvline(good_events[6], linewidth=1, color='sienna')
#plt.axvline(good_events[7], linewidth=1, color='tan')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
#plt.plot(times[0], ntremor, 'k')
#plt.xlim(tmin, tmax)
#plt.xlabel('Time (year)')
#plt.ylabel('Number of tremor per day')
plt.savefig('vespagram/D3456.eps', format='eps')
plt.close(1)

#plt.figure(1, figsize=(15, 10))
plt.figure(1, figsize=(15, 5))
#plt.subplot(211)
plt.contourf(times_subset[0], slowness * 365.25 / dy, vesp4 + vesp5 + vesp6, cmap=plt.get_cmap('seismic'), vmin=-24.0, vmax=24.0)
#for event in good_events:
#    plt.axvline(event, linewidth=1, color='grey')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
for tm in t_maxima:
    plt.axvline(tm, linewidth=1, color='k')
plt.xlabel('Time (year)')
plt.ylabel('Slowness (day / km)')
plt.colorbar(orientation='horizontal')
#plt.subplot(212)
#plt.axvline(good_events[0], linewidth=1, color='red')
#plt.axvline(good_events[1], linewidth=1, color='green')
#plt.axvline(good_events[2], linewidth=1, color='blue')
#plt.axvline(good_events[3], linewidth=1, color='magenta')
#plt.axvline(good_events[4], linewidth=1, color='cyan')
#plt.axvline(good_events[5], linewidth=1, color='darkorchid')
#plt.axvline(good_events[6], linewidth=1, color='sienna')
#plt.axvline(good_events[7], linewidth=1, color='tan')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
#plt.plot(times[0], ntremor, 'k')
#plt.xlim(tmin, tmax)
#plt.xlabel('Time (year)')
#plt.ylabel('Number of tremor per day')
plt.savefig('vespagram/D456.eps', format='eps')
plt.close(1)

#plt.figure(1, figsize=(15, 10))
plt.figure(1, figsize=(15, 5))
#plt.subplot(211)
plt.contourf(times_subset[0], slowness * 365.25 / dy, vesp4 + vesp5, cmap=plt.get_cmap('seismic'), vmin=-16.0, vmax=16.0)
#for event in good_events:
#    plt.axvline(event, linewidth=1, color='grey')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
for tm in t_maxima:
    plt.axvline(tm, linewidth=1, color='k')
plt.xlabel('Time (year)')
plt.ylabel('Slowness (day / km)')
plt.colorbar(orientation='horizontal')
#plt.subplot(212)
#plt.axvline(good_events[0], linewidth=1, color='red')
#plt.axvline(good_events[1], linewidth=1, color='green')
#plt.axvline(good_events[2], linewidth=1, color='blue')
#plt.axvline(good_events[3], linewidth=1, color='magenta')
#plt.axvline(good_events[4], linewidth=1, color='cyan')
#plt.axvline(good_events[5], linewidth=1, color='darkorchid')
#plt.axvline(good_events[6], linewidth=1, color='sienna')
#plt.axvline(good_events[7], linewidth=1, color='tan')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
#plt.plot(times[0], ntremor, 'k')
#plt.xlim(tmin, tmax)
#plt.xlabel('Time (year)')
#plt.ylabel('Number of tremor per day')
plt.savefig('vespagram/D45.eps', format='eps')
plt.close(1)

#plt.figure(1, figsize=(15, 10))
plt.figure(1, figsize=(15, 5))
#plt.subplot(211)
plt.contourf(times_subset[0], slowness * 365.25 / dy, vesp5 + vesp6, cmap=plt.get_cmap('seismic'), vmin=-25.0, vmax=25.0)
#for event in good_events:
#    plt.axvline(event, linewidth=1, color='grey')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
for tm in t_maxima:
    plt.axvline(tm, linewidth=1, color='k')
plt.ylabel('Slowness (day / km)')
plt.colorbar(orientation='horizontal')
#plt.subplot(212)
#plt.axvline(good_events[0], linewidth=1, color='red')
#plt.axvline(good_events[1], linewidth=1, color='green')
#plt.axvline(good_events[2], linewidth=1, color='blue')
#plt.axvline(good_events[3], linewidth=1, color='magenta')
#plt.axvline(good_events[4], linewidth=1, color='cyan')
#plt.axvline(good_events[5], linewidth=1, color='darkorchid')
#plt.axvline(good_events[6], linewidth=1, color='sienna')
#plt.axvline(good_events[7], linewidth=1, color='tan')
#for event in bad_events:
#    plt.axvline(event, linewidth=1, color='grey')
#plt.plot(times[0], ntremor, 'k')
#plt.xlim(tmin, tmax)
#plt.xlabel('Time (year)')
#plt.ylabel('Number of tremor per day')
plt.savefig('vespagram/D56.eps', format='eps')
plt.close(1)
