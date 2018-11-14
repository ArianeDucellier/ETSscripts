"""
Script to run get_MODWT for different GPS stations and get nice figures
"""

import datetime
import matplotlib.cm as cm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

import get_MODWT_GeoNet

stations = ['PUKE', 'ANAU', 'GISB', 'MAHI', 'CKID']
direction = 'u'

lats = [-38.07141492, -38.268212852, -38.635336932, -39.152563727, \
    -39.65787439]
lons = [178.257359581, 178.291223348, 177.886034958, 177.90700192, \
    177.07635309]

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

names = ['Haar', 'D4', 'D6', 'D8', 'D10', 'D12', 'D14', 'D16', 'D18', 'D20', \
    'LA8', 'LA10', 'LA12', 'LA14', 'LA16', 'LA18', 'LA20', 'C6', 'C12', \
    'C18', 'C24', 'C30', 'BL14', 'BL18', 'BL20']

# Plot the details and smooth for each station and each wavelet filter
# and reconstruct the time series after thresholding
J = 8
for name in ['LA8']:
    for station in stations:
        (times, disps, gaps) = get_MODWT_GeoNet.read_data(station, direction)

#        disps = get_MODWT_GeoNet.median_filtering(disps, 3)

        (Ws, Vs) = get_MODWT_GeoNet.compute_wavelet(times, disps, gaps, J, \
        name, station, direction, False, False, False)

        (Ds, Ss) = get_MODWT_GeoNet.compute_details(times, disps, gaps, Ws, \
            J, name, station, direction, True, False, False)

        get_MODWT_GeoNet.thresholding(times, disps, gaps, Ws, Vs, J, name, \
            station, direction, events, locations, True, False, True)

# Plot the wavelet details as function of latitude
#J = 8
#level = 8
#amp = 0.05
#
#for name in names:
#    # Initialize figure
#    params = {'xtick.labelsize':24,
#              'ytick.labelsize':24}
#    pylab.rcParams.update(params)
#    fig = plt.figure(1, figsize=(20, 10))
#
#    xmin = []
#    xmax = []
#    colors = cm.rainbow(np.linspace(0, 1, len(stations)))
#
#    # Loop on stations
#    for station, lat, c in zip(stations, lats, colors):
#        # Get details
#        (times, disps, gaps) = get_MODWT_GeoNet.read_data(station, direction)
#        (Ws, Vs) = get_MODWT_GeoNet.compute_wavelet(times, disps, gaps, J, \
#            name, station, direction, False, False, False)
#        (Ds, Ss) = get_MODWT_GeoNet.compute_details(times, disps, gaps, Ws, \
#            J, name, station, direction, False, False, False)
#        # Plot details
#        for i in range(0, len(times)):
#            time = times[i]
#            D = Ds[i]
#            if (i == 0):
#                plt.plot(time, lat + amp * D[level - 1], color=c, \
#                    label=station)
#            else:
#                plt.plot(time, lat + amp * D[level - 1], color=c)
#        # Get limits of plot
#        for i in range(0, len(times)):
#            time = times[i]
#            xmin.append(np.min(time))
#            xmax.append(np.max(time))
#
#    # Loop on slow slip events
#    for event, location in zip(events, locations):
#        for station in location:
#            for i in range(0, len(stations)):
#                if (station == stations[i]):
#                    y = lats[i]
#            plt.plot(np.array([datetime.date(year=event[0], month=event[1], \
#                day=event[2]), datetime.date(year=event[0], month=event[1], \
#                day=event[2])]), np.array([y - 2 * amp, y + 2 * amp]), \
#                linewidth=2, color='grey')
#
#    # End figure
#    plt.xlim(min(xmin), max(xmax))
#    plt.ylim(min(lats) - 2.0 * amp, max(lats) + 2.0 * amp)
#    plt.xlabel('Time (years)', fontsize=20)
#    plt.ylabel('Latitude', fontsize=20)
#    plt.legend(loc=3, fontsize=20)
#    if (direction == 'e'):
#        title = str(level) + 'th level detail - Eastern direction'
#    elif (direction == 'n'):
#        title = str(level) + 'th level detail - North direction'
#    else:
#        title = str(level) + 'th level detail - Vertical direction'
#    plt.suptitle(title, fontsize=24)
#    # Save figure
#    plt.savefig('D' + str(level) + '_' + direction + '_' + name + '.eps', \
#        format='eps')
#    plt.close(1)
