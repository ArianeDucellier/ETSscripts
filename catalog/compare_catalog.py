import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from math import floor

filename = '080328.09.029'

# Read our catalog
namefile = 'LFEs/' + filename + '/catalog.pkl'
df1 = pickle.load(open(namefile, 'rb'))
df1 = df1[['year', 'month', 'day', 'hour', 'minute', 'second', \
    'cc', 'nchannel']]
df1 = df1.astype({'year': int, 'month': int, 'day': int, \
    'hour': int, 'minute': int, 'second': float, \
    'cc': float, 'nchannel': int})
df1['second'] = df1['second'].round(3)

# Write our catalog to file
tfile = open('LFEs/' + filename + '/catalog1.txt', 'w')
tfile.write(df1.to_string())
tfile.close()

# Read catalog from Plourde et al. (2015)
namefile = '../data/Plourde_2015/detections/' \
    + filename + '_detect5_cull.txt'
file = open(namefile)
# Get number of channels
first_line = file.readline().strip()
staNames = first_line.split()
nsta = len(staNames)
# Get LFE times
df2 = pd.read_csv(namefile, delim_whitespace=True, header=None, \
    skiprows=2, names = ['id', 'date', 'hours', 'seconds', 'cc', 'nan'], \
    dtype={'id': 'str', 'date': 'str'})
df2 = df2.astype({'hours': int, 'seconds': float, 'cc': float})

# Modify Plourde's catalog for comparison
df2['year'] = df2['date'].str[0:2].astype('int') + 2000
df2['month'] = df2['date'].str[2:4].astype('int')
df2['day'] = df2['date'].str[4:6].astype('int')
df2['hour'] = df2['hours'] - 1
df2['minute'] = np.floor(df2['seconds'] / 60.0).astype('int')
df2['second'] = df2['seconds'] - 60.0 * df2['minute']
df2['nchannel'] = np.repeat(3 * nsta, len(df2))
df2 = df2.drop(columns=['id', 'date', 'hours', 'seconds', 'nan'])
df2 = df2[['year', 'month', 'day', 'hour', 'minute', 'second', \
    'cc', 'nchannel']]
df2 = df2.astype({'year': int, 'month': int, 'day': int, \
    'hour': int, 'minute': int, 'second': float, \
    'cc': float, 'nchannel': int})
df2['second'] = df2['second'].round(3)

# Write Plourde's catalog to file
tfile = open('LFEs/' + filename + '/catalog2.txt', 'w')
tfile.write(df2.to_string())
tfile.close()

# Shift our catalog to the right
df1r = df1.copy()
df1r['second'] = df1r['second'] + 0.025
df1r['second'] = df1r['second'].round(3)

# Shift our catalog to the left
df1l = df1.copy()
df1l['second'] = df1l['second'] - 0.025
df1l['second'] = df1l['second'].round(3)

# Shift Plourde's catalog to the right
df2r = df2.copy()
df2r['second'] = df2r['second'] + 0.025
df2r['second'] = df2r['second'].round(3)

# Shift Plourde's catalog to the left
df2l = df2.copy()
df2l['second'] = df2l['second'] - 0.025
df2l['second'] = df2l['second'].round(3)

# LFEs missing in our catalog (in df2 but not df1)
df1_all = pd.concat([df1, df1r, df1l], ignore_index=True)
df_all = pd.concat([df1_all, df2], ignore_index=True)
df_merge = df_all.merge(df1_all.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df_missing = df_merge[df_merge['_merge'] == 'left_only']
df_missing = df_missing.drop(columns=['cc_y', 'nchannel_y', '_merge'])
df_missing = df_missing.rename(columns={'cc_x': 'cc', 'nchannel_x': 'nchannel'})

# LFEs added in our catalog (in df1 but not df2)
df2_all = pd.concat([df2, df2r, df2l], ignore_index=True)
df_all = pd.concat([df2_all, df1], ignore_index=True)
df_merge = df_all.merge(df2_all.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df_added = df_merge[df_merge['_merge'] == 'left_only']
df_added = df_added.drop(columns=['cc_y', 'nchannel_y', '_merge'])
df_added = df_added.rename(columns={'cc_x': 'cc', 'nchannel_x': 'nchannel'})

# LFEs present in both catalogs (we use our shifted catalog
# and get cross-correlation values from Plourde's catalog)
df1_merge = df1.merge(df2.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df1r_merge = df1r.merge(df2.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df1l_merge = df1l.merge(df2.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df1_both = df1_merge[df1_merge['_merge'] == 'both']
df1r_both = df1r_merge[df1r_merge['_merge'] == 'both']
df1l_both = df1l_merge[df1l_merge['_merge'] == 'both']
df_both1 = pd.concat([df1_both, df1r_both, df1l_both], ignore_index=True)
df_both1 = df_both1.drop(columns=['cc_x', 'nchannel_x', '_merge'])
df_both1 = df_both1.rename(columns={'cc_y': 'cc', 'nchannel_y': 'nchannel'})
df_both1 = df_both1.astype({'year': int, 'month': int, 'day': int, \
    'hour': int, 'minute': int, 'second': float, \
    'cc': float, 'nchannel': int})

# LFEs present in both catalogs (we use Plourde's shifted catalog
# and get cross-correlation values from our catalog
df2_merge = df2.merge(df1.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df2r_merge = df2r.merge(df1.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df2l_merge = df2l.merge(df1.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df2_both = df2_merge[df2_merge['_merge'] == 'both']
df2r_both = df2r_merge[df2r_merge['_merge'] == 'both']
df2l_both = df2l_merge[df2l_merge['_merge'] == 'both']
df_both2 = pd.concat([df2_both, df2r_both, df2l_both], ignore_index=True)
df_both2 = df_both2.drop(columns=['cc_x', 'nchannel_x', '_merge'])
df_both2 = df_both2.rename(columns={'cc_y': 'cc', 'nchannel_y': 'nchannel'})
df_both2 = df_both2.astype({'year': int, 'month': int, 'day': int, \
    'hour': int, 'minute': int, 'second': float, \
    'cc': float, 'nchannel': int})

# Cross correlation values
# Difference between LFEs added in our catalog
# and LFEs present in both catalogs
plt.figure(1, figsize=(20, 10))
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params) 
ax1 = plt.subplot(121)
plt.hist(df_added['cc'], \
    bins=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, \
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22])
plt.xlabel('Cross-correlation value', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.title('LFEs added in our catalog', fontsize=30)
ax2 = plt.subplot(122)
plt.hist(df_both2['cc'], \
    bins=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, \
    0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22])
plt.xlabel('Cross-correlation value', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.title('LFEs present in both catalogs', fontsize=30)
plt.savefig('LFEs/' + filename + '/hist.eps', format='eps')
ax1.clear()
ax2.clear()
plt.close(1)

# Missing LFEs
# Time lag between missing LFE and closest LFE
# in Plourde's catalog
timediff = np.zeros(len(df_missing))
for i in range(0, len(df_missing)):
    timediff[i] = 86400.0
    date_i = datetime(df_missing['year'].iloc[i], df_missing['month'].iloc[i], \
        df_missing['day'].iloc[i], df_missing['hour'].iloc[i], \
        df_missing['minute'].iloc[i], int(floor(df_missing['second'].iloc[i])), \
        int(1000000 * (df_missing['second'].iloc[i] - floor(df_missing['second'].iloc[i]))))
    for j in range(0, len(df2)):
        date_j = datetime(df2['year'].iloc[j], df2['month'].iloc[j], \
            df2['day'].iloc[j], df2['hour'].iloc[j], \
            df2['minute'].iloc[j], int(floor(df2['second'].iloc[j])), \
            int(1000000 * (df2['second'].iloc[j] - floor(df2['second'].iloc[j]))))
        dt = date_i - date_j
        duration = abs(dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001)
        if ((duration > 0.0) and (duration < timediff[i])):
            timediff[i] = duration

plt.figure(2, figsize=(10, 10))
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params) 
plt.hist(timediff, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
plt.xlabel('Time to closest LFE', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.title('LFEs missing in our catalog', fontsize=30)
plt.savefig('LFEs/' + filename + '/timelag.eps', format='eps')
plt.close(2)

# Write number of LFEs
tfile = open('LFEs/' + filename + '/comparison.txt', 'w')
tfile.write('Number of LFEs in our catalog: {}\n'.format(len(df1)))
tfile.write('Number of LFEs in Plourde catalog: {}\n'.format(len(df2)))
tfile.write('Number of LFEs added in our catalog: {}\n'.format(len(df_added)))
tfile.write('Number of LFEs missing in our catalog: {}\n'.format(len(df_missing)))
tfile.write('Number of LFEs present in both catalogs: {}\n'.format(len(df_both1)))
tfile.close()
