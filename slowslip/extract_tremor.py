"""Extract tremor"""

import datetime
import numpy as np
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt
from scipy.io import loadmat

import date

stations = ['ALBH', 'CHCM', 'COUP', 'PGC5', 'SC02', 'SC03', 'UFDA', 'FRID', 'PNCL', 'SQIM']
direction = 'lon'
dataset = 'cleaned'
lats = [48.2323, 48.0106, 48.2173, 48.6483, 48.5462, 47.8166, 47.7550, 48.5352, 48.1014, 48.0823]
lons = [-123.2915, -122.7759, -122.6856, -123.4511, -123.0076, -123.7057, -122.6670, -123.0180, -123.4152, -123.1020]
lat0 = 48.1168
lon0 = -123.4943

times = []
disps = []

# GPS data
for station in stations:
    filename = '../data/PANGA/' + dataset + '/' + station + '.' + direction
    # Load the data
    data = np.loadtxt(filename, skiprows=26)
    time = data[:, 0]
    disp = data[:, 1]
    error = data[:, 2]
    sigma = np.std(disp)
    # Correct for the repeated values
    dt = np.diff(time)
    gap = np.where(dt < 1.0 / 365.0 - 0.0001)[0]
    for i in range(0, len(gap)):
        if ((time[gap[i] + 2] - time[gap[i] + 1] > 2.0 / 365.0 - 0.0001) \
        and (time[gap[i] + 2] - time[gap[i] + 1] < 2.0 / 365.0 + 0.0001)):
            time[gap[i] + 1] = 0.5 * (time[gap[i] + 2] + time[gap[i]])
        elif ((time[gap[i] + 2] - time[gap[i] + 1] > 1.0 / 365.0 - 0.0001) \
          and (time[gap[i] + 2] - time[gap[i] + 1] < 1.0 / 365.0 + 0.0001) \
          and (time[gap[i] + 3] - time[gap[i] + 2] > 2.0 / 365.0 - 0.0001) \
          and (time[gap[i] + 3] - time[gap[i] + 2] < 2.0 / 365.0 + 0.0001)):
            time[gap[i] + 1] = time[gap[i] + 2]
            time[gap[i] + 2] = 0.5 * (time[gap[i] + 2] + time[gap[i] + 3])
    # Look for gaps greater than 1 day
    days = 2
    dt = np.diff(time)
    gap = np.where(dt > days / 365.0 - 0.0001)[0]
    duration = np.round((time[gap + 1] - time[gap]) * 365).astype(np.int)
    # Fill the gaps by interpolation
    for j in range(0, len(gap)):
        time = np.insert(time, gap[j] + 1, \
            time[gap[j]] + np.arange(1, duration[j]) / 365.0)
        disp = np.insert(disp, gap[j] + 1, \
            np.random.normal(0.0, sigma, duration[j] - 1))
        gap[j + 1 : ] = gap[j + 1 : ] + duration[j] - 1
    times.append(time)
    disps.append(disp)

filename = '../data/timelags/SummaryLatest.mat'
data = loadmat(filename)
TREMall = data['TREMall']
day = TREMall[0][0][0]
latitude = TREMall[0][0][1]
longitude = TREMall[0][0][2]
nt = np.shape(day)[0]
time_tremor = np.zeros(nt)
lat_tremor = np.zeros(nt)
lon_tremor = np.zeros(nt)
for i in range(0, nt):
    myday = date.matlab2ymdhms(day[i][0])
    t1 = datetime.date(myday[0], myday[1], myday[2])
    t2 = datetime.date(myday[0], 1, 1)
    time_tremor[i] = myday[0] + (t1 - t2).days / 365.25
    lat_tremor[i] = latitude[i][0]
    lon_tremor[i] = longitude[i][0]

# Compute the tremor time for sources located less than 75 km from the origin
a = 6378.136
e = 0.006694470
dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
dist = np.sqrt(np.power((lat_tremor - lat0) * dy, 2.0) + np.power((lon_tremor - lon0) * dx, 2.0))
k_100 = (dist <= 100.0)
k_75 = (dist <= 75.0)
k_50 = (dist <= 50.0)
time_tremor_100 = time_tremor[k_100]
time_tremor_75 = time_tremor[k_75]
time_tremor_50 = time_tremor[k_50]

# Number of tremors per day
ntremor_100 = np.zeros(len(times[0]))
ntremor_75 = np.zeros(len(times[0]))
ntremor_50 = np.zeros(len(times[0]))
for i in range(0, len(times[0])):
    for j in range (0, len(time_tremor_100)):
        if ((time_tremor_100[j] >= times[0][i] - 0.5 / 365.0) and \
            (time_tremor_100[j] <= times[0][i] + 0.5 / 365.0)):
            ntremor_100[i] = ntremor_100[i] + 1
    for j in range (0, len(time_tremor_75)):
        if ((time_tremor_75[j] >= times[0][i] - 0.5 / 365.0) and \
            (time_tremor_75[j] <= times[0][i] + 0.5 / 365.0)):
            ntremor_75[i] = ntremor_100[i] + 1
    for j in range (0, len(time_tremor_50)):
        if ((time_tremor_50[j] >= times[0][i] - 0.5 / 365.0) and \
            (time_tremor_50[j] <= times[0][i] + 0.5 / 365.0)):
            ntremor_50[i] = ntremor_100[i] + 1

df = pd.DataFrame(data={'time':times[0], 'nt_100km':ntremor_100, 'nt_75km':ntremor_75, 'nt_50km':ntremor_50})
pickle.dump(df, open('time_tremor.pkl', 'wb'))
