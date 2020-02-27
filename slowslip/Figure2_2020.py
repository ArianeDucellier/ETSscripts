'''
Script to correlate wavelet details with simple waveform
'''

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import pi, sin

import correlate
import MODWT

# Moving average function
def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

# Read tremor file and apply moving average
window = 21
df = pickle.load(open('time_tremor.pkl', 'rb'))
MA_100 = movingaverage(df['nt_100km'], window)
MA_75 = movingaverage(df['nt_75km'], window)
MA_50 = movingaverage(df['nt_50km'], window)
time_MA = df['time'][int((window - 1) / 2):- int((window - 1) / 2)]

N = 500
duration = 10
name = 'LA8'
J = 8

# Create time vector
time = np.arange(0, N + 1)

# Create displacement vector
disp = np.zeros(N + 1)
for i in range(0, N + 1):
    if (time[i] <= 0.5 * (N - duration)):
        disp[i] = time[i] / (N - duration)
    elif (time[i] >= 0.5 * (N + duration)):
        disp[i] = (time[i] - N) / (N - duration)
    else:
        disp[i] = (0.5 * N - time[i]) / duration           

# Compute MODWT
(W, V) = MODWT.pyramid(disp, name, J)
(D, S) = MODWT.get_DS(disp, W, name, J)

N0 = 100

# Take only the center
imin = int((N - N0) / 2)
imax = int((N + N0) / 2)
time0 = time[imin : (imax + 1)]
disp0 = disp[imin : (imax + 1)]
D0 = []
for Dj in D:
    D0.append(Dj[imin : (imax + 1)])
S0 = []
for Sj in S:
    S0.append(Sj[imin : (imax + 1)])

event_size = duration
stations = ['ALBH', 'CHCM', 'COUP', 'PGC5', 'SC02', 'SC03', 'UFDA', 'FRID', 'PNCL', 'SQIM']
direction = 'lon'
dataset = 'cleaned'
lats = [48.2323, 48.0106, 48.2173, 48.6483, 48.5462, 47.8166, 47.7550, 48.5352, 48.1014, 48.0823]
lat0 = 48.1168
slowness = 0.0

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
    [W, V] = MODWT.pyramid(disp, name, J)
    Ws.append(W)
    Vs.append(V)
    (D, S) = MODWT.get_DS(disp, W, name, J)
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

# Conversion latitude -> kms
a = 6378.136
e = 0.006694470
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)

# Initialize figure
params = {'legend.fontsize': 20, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)   
plt.figure(1, figsize=(24, 6))
plt.gcf().subplots_adjust(bottom=0.15)

# Correlate 7th level detail
plt.subplot2grid((1, 3), (0, 0))
times_subset = []
Dj_subset = []
for (time, D) in zip(times, Ds):
    ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
    iend = np.where(np.abs(time - tmax) < 0.001)[0]
    times_subset.append(time[ibegin[0] : iend[0] + 1])
    Dj_subset.append(D[6][ibegin[0] : iend[0] + 1])

# Stack
stack = np.zeros(len(times_subset[0]))
latmin = min(lats)
for (time, Dj, lat) in zip(times_subset, Dj_subset, lats):
    Dj_interp = np.interp(time + slowness * (lat - lat0), time, Dj)
    stack = stack + Dj_interp

# Correlate
cc = correlate.optimized(D0[6], stack)
M = len(D0[6])
N = len(stack)
index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
time = times_subset[0][index]

# Plot
plt.plot(time, cc, 'k', label='D7')
plt.fill_between(time_MA, np.repeat(-1.0, len(time_MA)), MA_75 * 0.004 - 1, color='blue', label='Tremor')
plt.xlim(np.min(time), np.max(time))
plt.ylim(- 1.0, 1.0)
plt.xlabel('Time (days)', fontsize=24)
plt.xticks(np.arange(2009, 2012, 1))
plt.legend(loc=1)

# Correlate 8th level detail
plt.subplot2grid((1, 3), (0, 1))
times_subset = []
Dj_subset = []
for (time, D) in zip(times, Ds):
    ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
    iend = np.where(np.abs(time - tmax) < 0.001)[0]
    times_subset.append(time[ibegin[0] : iend[0] + 1])
    Dj_subset.append(D[7][ibegin[0] : iend[0] + 1])

# Stack
stack = np.zeros(len(times_subset[0]))
latmin = min(lats)
for (time, Dj, lat) in zip(times_subset, Dj_subset, lats):
    Dj_interp = np.interp(time + slowness * (lat - lat0), time, Dj)
    stack = stack + Dj_interp

# Correlate
cc = correlate.optimized(D0[7], stack)
M = len(D0[7])
N = len(stack)
index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
time = times_subset[0][index]

# Plot
plt.plot(time, cc, 'k', label='D8')
plt.fill_between(time_MA, np.repeat(-1.0, len(time_MA)), MA_75 * 0.004 - 1, color='blue', label='Tremor')
plt.xlim(np.min(time), np.max(time))
plt.ylim(- 1.0, 1.0)
plt.xlabel('Time (days)', fontsize=24)
plt.xticks(np.arange(2009, 2012, 1))
plt.legend(loc=1)

# Correlate sum of 7th 8th level details
plt.subplot2grid((1, 3), (0, 2))
times_subset = []
Dj_subset = []
for (time, D) in zip(times, Ds):
    ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
    iend = np.where(np.abs(time - tmax) < 0.001)[0]
    times_subset.append(time[ibegin[0] : iend[0] + 1])
    Dj_subset.append(D[6][ibegin[0] : iend[0] + 1] + D[7][ibegin[0] : iend[0] + 1])

# Stack
stack = np.zeros(len(times_subset[0]))
latmin = min(lats)
for (time, Dj, lat) in zip(times_subset, Dj_subset, lats):
    Dj_interp = np.interp(time + slowness * (lat - lat0), time, Dj)
    stack = stack + Dj_interp

# Correlate
cc = correlate.optimized(D0[6] + D0[7], stack)
M = len(D0[6] + D0[7])
N = len(stack)
index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
time = times_subset[0][index]

# Plot
plt.plot(time, cc, 'k', label='D7 + D8')
plt.fill_between(time_MA, np.repeat(-1.0, len(time_MA)), MA_75 * 0.004 - 1, color='blue', label='Tremor')
plt.xlim(np.min(time), np.max(time))
plt.ylim(- 1.0, 1.0)
plt.xlabel('Time (days)', fontsize=24)
plt.xticks(np.arange(2009, 2012, 1))
plt.legend(loc=1)

# Save figure
plt.suptitle('Event size = {:d} days'.format(event_size), fontsize=30)
plt.savefig('Figure2_2020.eps', format='eps')
plt.close(1)
