"""
NASA proposal - Figure 2
"""

import datetime
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

import get_MODWT_GeoNet

station = 'PUKE'
direction1 = 'e'
direction2 = 'u'
named1 = 'East'
named2 = 'Vertical'

name = 'C6'
J = 8
cutoff = 0.1

events = [[2010, 1, 31], [2010, 3, 31], [2010, 6, 30], [2010, 8, 15], \
    [2011, 4, 30], [2011, 4, 30], [2011, 8, 31], [2011, 9, 30], \
    [2011, 12, 15], [2012, 3, 15], [2012, 8, 15], [2012, 12, 31], \
    [2013, 2, 28], [2013, 6, 30], [2013, 7, 31], [2013, 9, 30], \
    [2013, 12, 15], [2014, 5, 31], [2014, 9, 30], [2014, 11, 15], \
    [2014, 12, 31], [2014, 12, 31], [2015, 1, 31], [1015, 2, 15], \
    [2015, 6, 30]]
locations = [['MAHI', 'CKID'], ['GISB', 'MAHI'], ['PUKE'], ['MAHI'], \
    ['PUKE'], ['ANAU'], ['MAHI', 'CKID'], ['ANAU', 'PUKE'], ['GISB'], \
    ['PUKE', 'ANAU', 'GISB'], ['ANAU', 'PUKE', 'GISB'], \
    ['PUKE', 'ANAU', 'GISB'], ['CKID', 'MAHI'], ['GISB', 'MAHI'], \
    ['ANAU', 'PUKE'], ['MAHI'], ['PUKE', 'ANAU'], ['ANAU', 'GISB'], \
    ['GISB', 'MAHI'], ['ANAU'], ['MAHI'], ['CKID'], ['PUKE'], ['GISB'], \
    ['ANAU', 'PUKE']]

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
dispfs = get_MODWT_GeoNet.low_pass_filter(times, disps, gaps, station, \
    direction1, cutoff)

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
        plt.plot(time, dispf, 'grey', label='Low-pass filtered')
        plt.plot(time, dispt, 'k', label='Denoised', linewidth=2)
    else:
        plt.plot(time, dispf, 'grey')
        plt.plot(time, dispt, 'k', linewidth=2)
    xmin.append(np.min(time))
    xmax.append(np.max(time))
for event, location in zip(events, locations):
    for site in location:
        if (site == station):
            plt.axvline(datetime.date(year=event[0], month=event[1], \
				day=event[2]), color='grey', linestyle='--')
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
dispfs = get_MODWT_GeoNet.low_pass_filter(times, disps, gaps, station, \
    direction1, cutoff)

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
        plt.plot(time, dispf, 'grey', label='Low-pass filtered')
        plt.plot(time, dispt, 'k', label='Denoised', linewidth=2)
    else:
        plt.plot(time, dispf, 'grey')
        plt.plot(time, dispt, 'k', linewidth=2)
    xmin.append(np.min(time))
    xmax.append(np.max(time))
for event, location in zip(events, locations):
	for site in location:
	    if (site == station):
		    plt.axvline(datetime.date(year=event[0], month=event[1], \
				day=event[2]), color='grey', linestyle='--')
plt.xlim(min(xmin), max(xmax))
plt.ylim(min(ymin), max(ymax))
plt.xlabel('Time (years)', fontsize=16)
plt.legend(loc=2, fontsize=16)

plt.savefig('Figure2_' + station + '.eps', format='eps')
plt.close(1)
