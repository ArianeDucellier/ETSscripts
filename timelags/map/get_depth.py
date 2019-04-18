"""
Script to compute the depth of the plate boundary using
the time lag between the direct P-wave and the direct S-wave
"""

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
for n in range(0, len(df)):
    ccEW = df['cc_' + stackStation + '_' + stackTremor + '_EW_cluster'][n]
    ccNS = df['cc_' + stackStation + '_' + stackTremor + '_NS_cluster'][n]
    if (ccEW >= ccNS):
        time = df['t_' + stackStation + '_' + stackTremor + '_EW_cluster'][n]
    else:
        time = df['t_' + stackStation + '_' + stackTremor + '_NS_cluster'][n]
    distance = (time / (1.0 / Vs - 1.0 / Vp)) ** 2.0 - df['x0'][n] ** 2.0 - df['y0'][n] ** 2.0
    if (distance >= 0.0):
        depth[n] = sqrt(distance)
    else:
        depth[n] = 0.0

# Write to file
table = pd.DataFrame(data={'longitude':longitude, 'latitude':latitude, 'depth':depth})
tfile = open('depth_' + stackStation + '_' + stackTremor + '.txt', 'w')
tfile.write(table.to_string(header=False, index=False))
tfile.close()
