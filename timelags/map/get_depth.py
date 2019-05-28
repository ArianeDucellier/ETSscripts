"""
Script to compute the depth of the plate boundary using
the time lag between the direct P-wave and the direct S-wave
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt

arrayName = 'BS'
lat0 = 47.95728
lon0 = -122.92866
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

for n in range(0, len(df)):
    ccEW = df['cc_' + stackStation + '_' + stackTremor + '_EW'][n]
    ccNS = df['cc_' + stackStation + '_' + stackTremor + '_NS'][n]
    if (ccEW >= ccNS):
        time = df['t_' + stackStation + '_' + stackTremor + '_EW_cluster'][n]
        cc[n] = ccEW
        ratio[n] = df['ratio_' + stackStation + '_' + stackTremor + '_EW'][n]
    else:
        time = df['t_' + stackStation + '_' + stackTremor + '_NS_cluster'][n]
        cc[n] = ccNS
        ratio[n] = df['ratio_' + stackStation + '_' + stackTremor + '_NS'][n]
    distance = (time / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - df['x0'][n] ** 2.0 \
        - df['y0'][n] ** 2.0
    if (distance >= 0.0):
        depth[n] = sqrt(distance)
    else:
        depth[n] = 0.0
    # Get the depth of the plate boundary
    myx = depth_pb['x'] == df['x0'][n]
    myy = depth_pb['y'] == df['y0'][n]
    myline = depth_pb[myx & myy]
    d0 = myline['depth'].iloc[0]
    d_to_pb[n] = d0 + depth[n]

# Keep only points with enough tremor
table = pd.DataFrame(data={'longitude':longitude, 'latitude':latitude, \
    'depth':depth, 'cc':cc, 'd_to_pb':d_to_pb, 'ratio':ratio, \
    'ntremor':df['ntremor']})
enough = table['ntremor'] > 10
table = table[enough]
positive = table['depth'] > 0.0
table = table[positive]

# Normalize cross correlation and ratio
table['cc'] = table['cc'] / table['cc'].max()
table['ratio'] = table['ratio'] / table['ratio'].max()

# Write to file
table1 = table.drop(columns=['cc', 'd_to_pb', 'ratio', 'ntremor'])
tfile = open('depth_' + stackStation + '_' + stackTremor + '.txt', 'w')
tfile.write(table1.to_string(header=False, index=False))
tfile.close()

table2 = table.drop(columns=['depth', 'd_to_pb', 'ratio', 'ntremor'])
tfile = open('cc_' + stackStation + '_' + stackTremor + '.txt', 'w')
tfile.write(table2.to_string(header=False, index=False))
tfile.close()

table3 = table.drop(columns=['depth', 'cc', 'ratio', 'ntremor'])
tfile = open('d_to_pb_' + stackStation + '_' + stackTremor + '.txt', 'w')
tfile.write(table3.to_string(header=False, index=False))
tfile.close()

table4 = table.drop(columns=['depth', 'cc', 'd_to_pb', 'ntremor'])
tfile = open('ratio_' + stackStation + '_' + stackTremor + '.txt', 'w')
tfile.write(table4.to_string(header=False, index=False))
tfile.close()

# Plot of distance to plate boundary
# versus number of tremor and ratio cc / RMS
plt.figure(1, figsize=(10, 6))
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params) 
ax1 = plt.subplot(121)
plt.plot(table['ntremor'], table['d_to_pb'], 'ko')
plt.title('Distance to plate boundary', fontsize=16)
plt.xlabel('Number of tremor', fontsize=16)
plt.ylabel('Distance (km)', fontsize=16)
ax2 = plt.subplot(122)
plt.plot(table['ratio'], table['d_to_pb'], 'ko')
plt.title('Distance to plate boundary', fontsize=16)
plt.xlabel('Ratio cc / RMS', fontsize=16)
plt.savefig('quality_{}_{}.eps'.format( \
    stackStation, stackTremor), format='eps')
plt.close(1)
