"""
This module contains a function to get all the permanent stations less than
a given distance from the template epicenter
"""

import itertools
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt

# Maximum distance
dmax = 100.0

# To transform latitude and longitude into kilometers
a = 6378.136
e = 0.006694470

# List of all stations
stations = []

# Get the locations of the epicenters of the LFE templates
LFEloc = np.loadtxt('../data/Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Loop on templates
for ie in range(0, len(LFEloc)):
    station_list = []
    lat0 = LFEloc[ie][2]
    lon0 = LFEloc[ie][3]
    dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
        sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
        pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)

    # Open BK station file
    BK = pd.read_csv('../data/networks/BK.txt', \
        delim_whitespace=True, header=None, skiprows=0)
    BK.columns = ['station', 'network', 'latitude', 'longitude']
    x = dx * (BK['longitude'] - lon0)
    y = dy * (BK['latitude'] - lat0)
    d = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
    for ir in range(0, len(BK['station'])):
        if (d[ir] <= dmax):
            station_list.append(BK['station'][ir])
            stations.append(BK['station'][ir])

    # Open NC station file
    NC = pd.read_csv('../data/networks/NC.txt', \
        delim_whitespace=True, header=None, skiprows=0)
    NC.columns = ['station', 'network', 'latitude', 'longitude']
    x = dx * (NC['longitude'] - lon0)
    y = dy * (NC['latitude'] - lat0)
    d = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
    for ir in range(0, len(NC['station'])):
        if (d[ir] <= dmax):
            station_list.append(NC['station'][ir])
            stations.append(NC['station'][ir])

    # Open PB station file
    PB = pd.read_csv('../data/networks/PB.txt', \
        delim_whitespace=True, header=None, skiprows=0)
    PB.columns = ['station', 'network', 'latitude', 'longitude']
    x = dx * (PB['longitude'] - lon0)
    y = dy * (PB['latitude'] - lat0)
    d = np.sqrt(np.power(x, 2.0) + np.power(y, 2.0))
    for ir in range(0, len(PB['station'])):
        if (d[ir] <= dmax):
            station_list.append(PB['station'][ir])
            stations.append(PB['station'][ir])

    # Save list of stations in file
    filename = 'stations/' + LFEloc[ie][0].decode('utf-8') + '.pkl'
    pickle.dump(station_list, open(filename, 'wb'))

stations.sort()
stations = list(stations for stations,_ in itertools.groupby(stations))
print(stations)
