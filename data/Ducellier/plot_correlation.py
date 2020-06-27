import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib.colors as colors
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from math import floor

import correlate

params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)

# List of LFE families
templates = np.loadtxt('../Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Beginning and end of the period we are looking at
tbegin = datetime(2007, 7, 1, 0, 0, 0)
tend = datetime(2009, 7, 1, 0, 0, 0)

# We construct the time series by counting the number of LFEs
# per two-day-long time window
window = 172800.0

# Length of the time series
dt = tend - tbegin
duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
nw = int(duration / window)

# Initialize event counts
X = np.zeros((nw, np.shape(templates)[0]), dtype=int)

# Initialize latitude
latitude = np.zeros(np.shape(templates)[0])
longitude = np.zeros(np.shape(templates)[0])

# Loop on templates
for i in range(0, np.shape(templates)[0]):

    latitude[i] = templates[i][2]
    longitude[i] = templates[i][3]

    # Open LFE catalog
    namedir = 'catalogs/' + templates[i][0].astype(str)
    namefile = namedir + '/catalog_2007_2009.pkl'
    df = pickle.load(open(namefile, 'rb'))

    # Filter LFEs
    maxc = np.max(df['nchannel'])
    df = df.loc[df['cc'] * df['nchannel'] >= 0.1 * maxc]

    # Loop on LFEs
    for j in range(0, len(df)):
        myYear = df['year'].iloc[j]
        myMonth = df['month'].iloc[j]
        myDay = df['day'].iloc[j]
        myHour = df['hour'].iloc[j]
        myMinute = df['minute'].iloc[j]
        mySecond = int(floor(df['second'].iloc[j]))
        myMicrosecond = int(1000000.0 * (df['second'].iloc[j] - mySecond))
        t = datetime(myYear, myMonth, myDay, myHour, myMinute, mySecond, \
            myMicrosecond)
        # Add LFE to appropriate time window
        if ((tbegin <= t) and (t < tbegin + timedelta(seconds=nw * window))):
            dt = t - tbegin
            duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * \
                0.000001
            index = int(duration / window)
            X[index, i] = X[index, i] + 1        
    m = np.mean(X[:, i])
    s = np.std(X[:, i])
    X[:, i] = np.where(X[:, i] > m + 3.0 * s, 1.0, 0.0)

# Cross correlation
cc = np.zeros((np.shape(templates)[0], np.shape(templates)[0]))
index1 = np.zeros((np.shape(templates)[0], np.shape(templates)[0]))
index2 = np.zeros((np.shape(templates)[0], np.shape(templates)[0]))
for i in range(0, np.shape(templates)[0]):
    for j in range(0, np.shape(templates)[0]):
        index1[i, j] = i
        index2[i, j] = j
        prod = correlate.optimized(X[:, latitude.argsort()[i]], X[:, latitude.argsort()[j]])
        cc[i, j] = np.max(prod)

# Write output files
for i in range(0, np.shape(templates)[0]):
    filename = 'correlation/' + templates[latitude.argsort()[i]][0].astype(str) + '.txt'
    tfile = open(filename, 'w')
    for j in range(0, np.shape(templates)[0]):
        if (i != j):
            tfile.write('{} {} {}\n'.format(longitude[latitude.argsort()[j]], \
                latitude[latitude.argsort()[j]], cc[i, j]))
    tfile.close()

# Plot figure
plt.figure(1, figsize=(10, 10))
plt.scatter(index1, index2, c=cc, marker='s', cmap='Reds')
plt.title('Cross correlation', fontsize=20)
plt.xlabel('Index 1', fontsize=20)
plt.ylabel('Index 2', fontsize=20)
plt.tight_layout()
plt.savefig('correlation.eps', format='eps')
plt.close(1)

# Draw maps
plt.style.use('bmh')

CALIFORNIA_NORTH = 3311 # projection

lonmin = -125.0
lonmax = -121.0
latmin = 39.0
latmax = 42.0

shapename = 'ocean'
ocean_shp = shapereader.natural_earth(resolution='10m',
                                      category='physical',
                                      name=shapename)

shapename = 'land'
land_shp = shapereader.natural_earth(resolution='10m',
                                     category='physical',
                                     name=shapename)

for i in range(0, np.shape(templates)[0]):

    fig = plt.figure(1, figsize=(15, 15)) 
    ax = plt.axes(projection=ccrs.epsg(CALIFORNIA_NORTH))
    ax.set_extent([lonmin, lonmax, latmin, latmax], ccrs.Geodetic())
    ax.set_title(templates[latitude.argsort()[i]][0].astype(str), fontsize=24)
    ax.gridlines(linestyle=":")

#    for myfeature in shapereader.Reader(ocean_shp).geometries(): 
#        ax.add_geometries([myfeature], ccrs.PlateCarree(), facecolor='#E0FFFF', edgecolor='black', alpha=0.5)
#    for myfeature in shapereader.Reader(land_shp).geometries(): 
#        ax.add_geometries([myfeature], ccrs.PlateCarree(), facecolor='#FFFFE0', edgecolor='black', alpha=0.5)

    cm = plt.cm.get_cmap('YlOrRd')
    sc = plt.scatter(longitude[latitude.argsort()], latitude[latitude.argsort()], c=cc[i, :], \
        vmin=0, vmax=1, cmap='Reds', s=400, transform=ccrs.PlateCarree())
    plt.colorbar(sc)

    plt.savefig('correlation/' + templates[latitude.argsort()[i]][0].astype(str) + '.eps', format='eps')
    plt.close(1)
