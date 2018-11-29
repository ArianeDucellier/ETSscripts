"""
NASA proposal - Figure 2
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

import get_MODWT_GeoNet

station = 'CKID'
direction1 = 'e'
direction2 = 'u'
named1 = 'East'
named2 = 'Vertical'

name = 'C6'
J = 8
cutoff = 0.025

# Create figure
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params)   
fig = plt.figure(1, figsize=(20, 10))

# Direction 1
(times, disps, gaps) = get_MODWT_GeoNet.read_data(station, direction1)
fdisps = get_MODWT_GeoNet.median_filtering(disps, 11)
(Ws, Vs) = get_MODWT_GeoNet.compute_wavelet(times, fdisps, gaps, J, name, \
    station, direction1, False, False, False)
dispts = get_MODWT_GeoNet.thresholding(times, disps, gaps, Ws, Vs, J, name, \
    station, direction1, [], [], False, False, False)
dispfs = get_MODWT_GeoNet.low_pass_filter(times, disps, gaps, station, direction1, cutoff)

ymin = []
ymax = []
for i in range(0, len(times)):
    disp = disps[i]
    dispt = dispts[i]
    dispf = dispfs[i]
    maxy = max(np.max(disp), np.max(dispt), np.max(dispf))
    miny = min(np.min(disp), np.min(dispt), np.min(dispf))
    ymax.append(maxy)
    ymin.append(miny)

plt.subplot2grid((2, 2), (0, 0))
xmin = []
xmax = []
for i in range(0, len(times)):
    time = times[i]
    disp = disps[i]
    if (i == 0):
         plt.plot(time, disp, 'k', label=named1)
    else:
        plt.plot(time, disp, 'k')
    xmin.append(np.min(time))
    xmax.append(np.max(time))
plt.xlim(min(xmin), max(xmax))
plt.ylim(min(ymin), max(ymax))
plt.legend(loc=2, fontsize=16)
plt.title('Original data', fontsize=16)

plt.subplot2grid((2, 2), (0, 1))
xmin = []
xmax = []
for i in range(0, len(times)):
    time = times[i]
    dispt = dispts[i]
    dispf = dispfs[i]
    if (i == 0):
        plt.plot(time, dispt, 'k', label='Denoised', linewidth=2)
        plt.plot(time, dispf, 'grey', label='Low-pass filtered')
    else:
        plt.plot(time, dispt, 'k', linewidth=2)
        plt.plot(time, dispf, 'grey')
    xmin.append(np.min(time))
    xmax.append(np.max(time))
plt.xlim(min(xmin), max(xmax))
plt.ylim(min(ymin), max(ymax))
plt.title('Denoised signal', fontsize=16)
plt.legend(loc=2, fontsize=16)

# Direction 2
(times, disps, gaps) = get_MODWT_GeoNet.read_data(station, direction2)
fdisps = get_MODWT_GeoNet.median_filtering(disps, 11)
(Ws, Vs) = get_MODWT_GeoNet.compute_wavelet(times, fdisps, gaps, J, name, \
    station, direction2, False, False, False)
dispts = get_MODWT_GeoNet.thresholding(times, disps, gaps, Ws, Vs, J, name, \
    station, direction2, [], [], False, False, False)
dispfs = get_MODWT_GeoNet.low_pass_filter(times, disps, gaps, station, direction1, cutoff)

ymin = []
ymax = []
for i in range(0, len(times)):
    disp = disps[i]
    dispt = dispts[i]
    dispf = dispfs[i]
    maxy = max(np.max(disp), np.max(dispt), np.max(dispf))
    miny = min(np.min(disp), np.min(dispt), np.min(dispf))
    ymax.append(maxy)
    ymin.append(miny)

plt.subplot2grid((2, 2), (1, 0))
xmin = []
xmax = []
for i in range(0, len(times)):
    time = times[i]
    disp = disps[i]
    if (i == 0):
         plt.plot(time, disp, 'k', label=named2)
    else:
        plt.plot(time, disp, 'k')
    xmin.append(np.min(time))
    xmax.append(np.max(time))
plt.xlim(min(xmin), max(xmax))
plt.ylim(min(ymin), max(ymax))
plt.xlabel('Time (years)', fontsize=16)
plt.legend(loc=2, fontsize=16)

plt.subplot2grid((2, 2), (1, 1))
xmin = []
xmax = []
for i in range(0, len(times)):
    time = times[i]
    dispt = dispts[i]
    dispf = dispfs[i]
    if (i == 0):
        plt.plot(time, dispt, 'k', label='Denoised', linewidth=2)
        plt.plot(time, dispf, 'grey', label='Low-pass filtered')
    else:
        plt.plot(time, dispt, 'k', linewidth=2)
        plt.plot(time, dispf, 'grey')
    xmin.append(np.min(time))
    xmax.append(np.max(time))
plt.xlim(min(xmin), max(xmax))
plt.ylim(min(ymin), max(ymax))
plt.xlabel('Time (years)', fontsize=16)
plt.legend(loc=2, fontsize=16)

plt.savefig('Figure2.eps', format='eps')
plt.close(1)
