"""
Script to compute the depth of the source of the tremor using
the time lag between the direct P-wave and the direct S-wave
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import atan2, cos, pi, sin, sqrt

arrayName = 'TB'
lat0 = 47.9730357142857
lon0 = -123.138492857143
stackStation = 'PWS'
stackTremor = 'PWS'
Vs = 3.6
Vp = 6.4

df = pickle.load(open('../' + arrayName + '_timelag.pkl', 'rb'))

# Get depth of plate boundary around the array
depth_pb = pd.read_csv('../depth/' + arrayName + '_depth.txt', sep=' ', \
    header=None)
depth_pb.columns = ['x', 'y', 'depth']

# Earth's radius and ellipticity
a = 6378.136
e = 0.006694470
    
# Convert kilometers to latitude, longitude
dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
    sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
    pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
longitude = lon0 + df['x0'] / dx
latitude = lat0 + df['y0'] / dy

# Compute the depth
depth = pd.Series(np.zeros(len(df)))
cc = pd.Series(np.zeros(len(df)))
d_to_pb = pd.Series(np.zeros(len(df)))
ratio = pd.Series(np.zeros(len(df)))
thickness = pd.Series(np.zeros(len(df)))
azimuth = pd.Series(np.zeros(len(df)))

for n in range(0, len(df)):
    if (df['ntremor'][n] > 10):
        # Get the depth of the plate boundary
        myx = depth_pb['x'] == df['x0'][n]
        myy = depth_pb['y'] == df['y0'][n]
        myline = depth_pb[myx & myy]
        d0 = myline['depth'].iloc[0]
        # Get values
        ccEW = df['cc_' + stackStation + '_' + stackTremor + '_EW'][n]
        ccNS = df['cc_' + stackStation + '_' + stackTremor + '_NS'][n]
        if (ccEW >= ccNS):
            time = df['t_' + stackStation + '_' + stackTremor + '_EW_cluster'][n]
            dt = df['std_' + stackStation + '_' + stackTremor + '_EW'][n]
            cc[n] = ccEW
            ratio[n] = df['ratio_' + stackStation + '_' + stackTremor + '_EW'][n]
        else:
            time = df['t_' + stackStation + '_' + stackTremor + '_NS_cluster'][n]
            dt = df['std_' + stackStation + '_' + stackTremor + '_NS'][n]
            cc[n] = ccNS
            ratio[n] = df['ratio_' + stackStation + '_' + stackTremor + '_NS'][n]
        # Compute distances
        distance = (time / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - df['x0'][n] ** 2.0 \
            - df['y0'][n] ** 2.0
        d1 = ((time + dt) / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - df['x0'][n] ** 2.0 \
            - df['y0'][n] ** 2.0
        d2 = ((time - dt) / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - df['x0'][n] ** 2.0 \
            - df['y0'][n] ** 2.0
        if (distance >= 0.0):
            depth[n] = sqrt(distance)
            d_to_pb[n] = d0 + depth[n]
        else:
            depth[n] = np.nan
            d_to_pb[n] = np.nan
        if ((d1 >= 0.0) and (d2>= 0.0)):
            thickness[n] = sqrt(d1) - sqrt(d2)
        else:
            thickness[n] = np.nan    
    else:
        cc[n] = np.nan
        ratio[n] = np.nan
        depth[n] = np.nan
        thickness[n] = np.nan
        d_to_pb[n] = np.nan        
    azimuth[n] = atan2(df['y0'][n], df['x0'][n]) * 180.0 / pi
    
table = pd.DataFrame(data={'longitude':longitude, 'latitude':latitude, \
    'depth':depth, 'cc':cc, 'd_to_pb':d_to_pb, 'ratio':ratio, \
    'thickness':thickness, 'ntremor':df['ntremor'], 'azimuth':azimuth})

# Write to file
namefile = arrayName + '/table_' + stackStation + '_' + stackTremor + '.pkl'
pickle.dump(table, open(namefile, 'wb'))

# Normalize ratio for plotting
table['ratio'] = table['ratio'] / table['ratio'].max()

# Plot of distance to plate boundary
# versus number of tremor, ratio cc / RMS, and azimuth
plt.figure(1, figsize=(15, 6))
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params) 
ax1 = plt.subplot(131)
plt.plot(table['ntremor'], table['d_to_pb'], 'ko')
plt.title('Distance to plate boundary', fontsize=16)
plt.xlabel('Number of tremor', fontsize=16)
plt.ylabel('Distance (km)', fontsize=16)
ax2 = plt.subplot(132)
plt.plot(table['ratio'], table['d_to_pb'], 'ko')
plt.title('Distance to plate boundary', fontsize=16)
plt.xlabel('Ratio cc / RMS', fontsize=16)
ax2 = plt.subplot(133)
plt.plot(table['azimuth'], table['d_to_pb'], 'ko')
plt.title('Distance to plate boundary', fontsize=16)
plt.xlabel('Azimuth', fontsize=16)
plt.savefig(arrayName + '/quality_{}_{}.eps'.format(stackStation, stackTremor), format='eps')
plt.close(1)
