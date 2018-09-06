"""
Scipt to run get_MODWT for different GPS stations and get nice figures
"""

import get_MODWT

stations = ['CLRS', 'CUSH', 'FRID', 'PGC5', 'PNCL', 'PTAA', 'SQIM']
direction = 'rad'
dataset = 'cleaned'
lats = [48.8203, 47.4233, 48.5350, 48.6485, 48.1014, 48.1168, 48.0825]
lons = [-124.1309, -123.2199, -123.0180, -123.4511, -123.4152, -123.4943, \
    -123.1018]

time_ETS = [2000.9583, 2002.1250, 2003.1250, 2004.0417, 2004.5417, 2005.7083, \
            2007.0833, 2008.375, 2009.375, 2010.6667, 2011.6667, 2012.7083, \
            2013.7500, 2014.9167, 2016.0000, 2017.1667]

#for station in stations:
#    (times, disps, gaps) = get_MODWT.read_data(station, direction, dataset)
#
#    (Ws, Vs) = get_MODWT.compute_wavelet(times, disps, gaps, time_ETS, 6, \
#        'LA8', station, direction, dataset, True, False, False)
#
#    (Ds, Ss) = get_MODWT.compute_details(times, disps, gaps, Ws, time_ETS, 6, \
#        'LA8', station, direction, dataset, True, False, False)

filename = '08_01_2009_09_05_2018.txt'
dists = [100.0, 80.0, 60.0, 40.0, 20.0]

for station, lat, lon in zip(stations, lats, lons):
    get_MODWT.draw_tremor(filename, dists, lat, lon, station)
