"""
Script to plot tremor and LFEs as a function of latitude
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import floor

from date import ymdhms2day

# Read tremor file
tremor = pd.read_csv('tremor/tremor.xyddtt', sep=' ', header=None)
tremor.columns = ['longitude', 'latitude', 'downdip', 'alongstrike', 'year', 'epoch']
year_tremor = tremor['year']
lat_tremor = tremor['latitude']

# Start figure and plot tremor
params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)
plt.figure(1, figsize=(20, 10))
plt.plot(year_tremor, lat_tremor, 'ro', markersize=5)

# List of LFE families
templates = np.loadtxt('../Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Loop on templates
for i in range(0, np.shape(templates)[0]):
    # Get latitude
    latitude = templates[i][2]

    # Open LFE catalog
    namedir = 'catalogs/' + templates[i][0].astype(str)
    namefile = namedir + '/catalog_2007_2009.pkl'
    df = pickle.load(open(namefile, 'rb'))

    # Filter LFEs
    df = df.loc[df['cc'] >= 0.09]

    N = len(df)
    time = np.zeros(N)
    # Loop on LFEs
    for j in range(0, len(df)):
        myYear = df['year'].iloc[j]
        myMonth = df['month'].iloc[j]
        myDay = df['day'].iloc[j]
        myHour = df['hour'].iloc[j]
        myMinute = df['minute'].iloc[j]
        mySecond = int(floor(df['second'].iloc[j]))
        myMicrosecond = int(1000000.0 * (df['second'].iloc[j] - mySecond))
        time[j] = ymdhms2day(myYear, myMonth, myDay, myHour, myMinute, mySecond)
    plt.plot(time, np.repeat(latitude, N), 'bo', markersize=5)

# End figure
plt.xlim([2007.5, 2009.5])
plt.ylim([39.4, 41.8])
plt.xlabel('Time (years)', fontsize=24)
plt.ylabel('Latitude', fontsize=24)
plt.title('Tremor and LFEs', fontsize=24)
plt.savefig('LFEdistribution/tremor.eps', format='eps')
plt.close(1)
