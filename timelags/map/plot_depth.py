"""
Script to plot the depth of source of the tremor
"""

import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

arrayName = 'TB'
stackStation = 'PWS'
stackTremor = 'PWS'
variable = 'ntremor'
#title = 'Depth (km)'
#title = 'Cross correlation'
#title = 'Distance to plate boundary (km)'
#title = 'Ratio cc peak to RMS'
#title = 'Thickness (km)'
title = 'Number of tremor'

# Data range
#vmin = 20
#vmax = 50
#vmin = 0.0
#vmax = 0.01
#vmin = -15
#vmax = 15
#vmin = 0
#vmax = 200
#vmin = 0
#vmax = 20
vmin = 0
vmax = 300

# Set matplotlib style
plt.style.use('bmh')

# Choose projection
WASHINGTON_NORTH = 2926

# Set boundaries
lonmin = -123.9
lonmax = -122.5
latmin = 47.6
latmax = 48.4

# Load map of ocean
shapename = 'ocean'
ocean_shp = shapereader.natural_earth(resolution='10m',
                                       category='physical',
                                       name=shapename)

# Load map of land
shapename = 'land'
land_shp = shapereader.natural_earth(resolution='10m',
                                       category='physical',
                                       name=shapename)

# Load data
df = pickle.load(open(arrayName + '/table_' + stackStation + '_' + stackTremor + '.pkl', 'rb'))

# Load LFEs
df_sweet = pickle.load(open('../depth/LFEs_Sweet_2014.pkl', 'rb'))
df_chestler = pickle.load(open('../depth/LFEs_Chestler_2017.pkl', 'rb'))

# Draw figure
fig = plt.figure(figsize=(15, 15)) 
ax = plt.axes(projection=ccrs.epsg(WASHINGTON_NORTH))
ax.set_extent([lonmin, lonmax, latmin, latmax], ccrs.Geodetic())
ax.gridlines(linestyle=":")

# Background
for myfeature in shapereader.Reader(ocean_shp).geometries(): 
    ax.add_geometries([myfeature], ccrs.PlateCarree(), facecolor='#E0FFFF', edgecolor='black', alpha=0.5)
for myfeature in shapereader.Reader(land_shp).geometries(): 
    ax.add_geometries([myfeature], ccrs.PlateCarree(), facecolor='#FFFFE0', edgecolor='black', alpha=0.5)

# Data transform
X = np.array(df['longitude']).reshape((11, 11))
Y = np.array(df['latitude']).reshape((11, 11))
Z = np.array(df[variable]).reshape((11, 11))

X = 0.5 * (X[0:10, :] + X[1:11, :])
X = np.vstack((2.0 * X[0:1, :] - X[1:2, :], X, 2.0 * X[9:10, :] - X[8:9, :]))
X = np.hstack((X, X[:, 0:1]))
Y = 0.5 * (Y[:, 0:10] + Y[:, 1:11])
Y = np.hstack((2.0 * Y[:, 0:1] - Y[:, 1:2], Y, 2.0 * Y[:, 9:10] - Y[:, 8:9]))
Y = np.vstack((Y, Y[0:1, :]))

# Plot tremor related variable
mesh = plt.pcolormesh(X, Y, Z, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

# If we have plotted the depth
if (variable == 'depth'):
    ax.scatter(df_sweet['longitude'], df_sweet['latitude'], c=df_sweet['depth'], marker='o', s=100, \
        vmin=vmin, vmax=vmax, edgecolor='k', transform=ccrs.PlateCarree())
    ax.scatter(df_chestler['longitude'], df_chestler['latitude'], c=df_chestler['depth'], marker='o', s=100, \
        vmin=vmin, vmax=vmax, edgecolor='k', transform=ccrs.PlateCarree())

# If we have plotted the distance to the plate boundary
if (variable == 'd_to_pb'):
    p_sweet = ax.scatter(df_sweet['longitude'], df_sweet['latitude'], c=df_sweet['depth_pb'] - df_sweet['depth'], \
        marker='o', s=100, vmin=vmin, vmax=vmax, edgecolor='k', transform=ccrs.PlateCarree())
    p_chestler = ax.scatter(df_chestler['longitude'], df_chestler['latitude'], c=df_chestler['depth_pb'] - df_chestler['depth'], \
        marker='o', s=100, vmin=vmin,vmax=vmax, edgecolor='k', transform=ccrs.PlateCarree())

# Colorbar
cb = fig.colorbar(mesh, orientation='vertical', shrink=0.5)
cb.set_label(label=title, size=20)
cb.ax.tick_params(labelsize=16)

# Save figure
namefile = arrayName + '/' + variable + '_' + stackStation + '_' + stackTremor
fig.savefig(namefile + '.pdf', format='pdf')
