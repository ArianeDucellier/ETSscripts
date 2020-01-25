"""
Script to run get_MODWT for different GPS stations and get nice figures
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import get_MODWT

#stations = ['CLRS', 'CUSH', 'FRID', 'PGC5', 'PNCL', 'PTAA', 'SQIM']
#direction = 'lon'
#dataset = 'cleaned'
#lats = [48.8203, 47.4233, 48.5350, 48.6485, 48.1014, 48.1168, 48.0825]
#lons = [-124.1309, -123.2199, -123.0180, -123.4511, -123.4152, -123.4943, -123.1018]

stations = ['ALBH', 'CHCM', 'COUP', 'PGC5', 'SC02', 'SC03', 'UFDA', 'FRID', 'PNCL', 'SQIM']
direction = 'lon'
dataset = 'cleaned'
lats = [48.2323, 48.0106, 48.2173, 48.6483, 48.5462, 47.8166, 47.7550, 48.5352, 48.1014, 48.0823]
lons = [-123.2915, -122.7759, -122.6856, -123.4511, -123.0076, -123.7057, -122.6670, -123.0180, -123.4152, -123.1020]

time_ETS = [2010.63, 2011.63, 2012.73, 2013.72, 2014.92, 2016.01, 2017.22, \
    2018.37]

# Plot the wavelet coefficients and the details and smooth for each station
for station in stations:
    (times, disps, gaps) = get_MODWT.read_data(station, direction, dataset)

    (Ws, Vs) = get_MODWT.compute_wavelet(times, disps, gaps, time_ETS, 8, \
        'LA8', station, direction, dataset, True, False, False)

    (Ds, Ss) = get_MODWT.compute_details(times, disps, gaps, Ws, time_ETS, 8, \
        'LA8', station, direction, dataset, True, False, False)

# Plot the cumulative number of tremor for each station
filename = '08_01_2009_09_05_2018.txt'
dists = [100.0, 80.0, 60.0, 40.0, 20.0]

for station, lat, lon in zip(stations, lats, lons):
    get_MODWT.draw_tremor(filename, dists, lat, lon, station, time_ETS)

# Plot the wavelet details as function of latitude
#J = 5
#amp = 0.1

# Initialize figure
#plt.figure(1, figsize=(20, 10))
#xmin = []
#xmax = []
#colors = cm.rainbow(np.linspace(0, 1, len(stations)))
# Loop on stations
#for station, lat, c in zip(stations, lats, colors):
#    # Get details
#    (times, disps, gaps) = get_MODWT.read_data(station, direction, dataset)
#    (Ws, Vs) = get_MODWT.compute_wavelet(times, disps, gaps, time_ETS, 6, \
#        'LA8', station, direction, dataset, False, False, False)
#    (Ds, Ss) = get_MODWT.compute_details(times, disps, gaps, Ws, time_ETS, 6, \
#        'LA8', station, direction, dataset, False, False, False)
#    # Plot details
#    for i in range(0, len(times)):
#        time = times[i]
#        D = Ds[i]
#        if (i == 0):
#           plt.plot(time, lat + amp * D[J - 1], color=c, label=station)
#        else:
#            plt.plot(time, lat + amp * D[J - 1], color=c)
#    # Get limits of plot
#    for i in range(0, len(times)):
#        time = times[i]
#        xmin.append(np.min(time))
#        xmax.append(np.max(time))
# Plot timing of ETS events      
#for i in range(0, len(time_ETS)):
#    plt.axvline(time_ETS[i], linewidth=2, color='grey')
# End figure
#plt.xlim(min(xmin), max(xmax))
#plt.ylim(min(lats) - 2.0 * amp, max(lats) + 2.0 * amp)
#plt.xlabel('Time (years)', fontsize=20)
#plt.ylabel('Latitude', fontsize=20)
#plt.legend(loc=3, fontsize=20)
#title = str(J) + 'th level detail - ' + direction + ' (' + dataset + ' data)'
#plt.suptitle(title, fontsize=24)
# Save figure
#plt.savefig('D' + str(J) + '_' + direction + '_' + dataset + '.eps', \
#    format='eps')
#plt.close(1)
