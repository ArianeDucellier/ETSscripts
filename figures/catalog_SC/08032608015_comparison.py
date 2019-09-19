import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

filename = '080326.08.015'
catalog = 'catalog_200709-200906.pkl'
tbegin = '2007-9-26'
tend = '2009-6-8'
threshold = 0.09
figurename = '08032608015_comparison.eps'

# Permanent stations catalog
namefile = '../../catalog/LFEs_permanent/' + filename + '/' + catalog
df1 = pickle.load(open(namefile, 'rb'))
df1 = df1[['year', 'month', 'day', 'hour', 'minute', 'second', \
    'cc', 'nchannel']]
df1 = df1.astype({'year': int, 'month': int, 'day': int, \
    'hour': int, 'minute': int, 'second': float, \
    'cc': float, 'nchannel': int})
df1['second'] = df1['second'].round(3)

# FAME stations catalog
namefile = '../../catalog/LFEs_unknown/' + filename + '/' + catalog
df2 = pickle.load(open(namefile, 'rb'))
df2 = df2[['year', 'month', 'day', 'hour', 'minute', 'second', \
    'cc', 'nchannel']]
df2 = df2.astype({'year': int, 'month': int, 'day': int, \
    'hour': int, 'minute': int, 'second': float, \
    'cc': float, 'nchannel': int})
df2['second'] = df2['second'].round(3)

# Filter catalogs (cross correlation)
df1 = df1.loc[df1['cc'] >= threshold]
df2 = df2.loc[df2['cc'] >= threshold]

# Filter catalog (date)
df1_date = pd.DataFrame({'year': df1['year'], 'month': df1['month'], 'day': df1['day']})
df1_date = pd.to_datetime(df1_date)
df1['date'] = df1_date
df1 = df1.set_index(['date'])
df1 = df1.loc[tbegin:tend]

df2_date = pd.DataFrame({'year': df2['year'], 'month': df2['month'], 'day': df2['day']})
df2_date = pd.to_datetime(df2_date)
df2['date'] = df2_date
df2 = df2.set_index(['date'])
df2 = df2.loc[tbegin:tend]

# Write permanent stations catalog to file
tfile = open(filename + '_permanent.txt', 'w')
tfile.write(df1.to_string())
tfile.close()

# Write FAME stations catalog to file
tfile = open(filename + '_FAME.txt', 'w')
tfile.write(df2.to_string())
tfile.close()

# LFEs missing in permanent stations catalog (in df2 but not df1)
df_all = pd.concat([df1, df2], ignore_index=True)
df_merge = df_all.merge(df1.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df_missing = df_merge[df_merge['_merge'] == 'left_only']
df_missing = df_missing.drop(columns=['cc_y', 'nchannel_y', '_merge'])
df_missing = df_missing.rename(columns={'cc_x': 'cc', 'nchannel_x': 'nchannel'})

# LFEs added in permanent stations catalog (in df1 but not df2)
df_all = pd.concat([df2, df1], ignore_index=True)
df_merge = df_all.merge(df2.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df_added = df_merge[df_merge['_merge'] == 'left_only']
df_added = df_added.drop(columns=['cc_y', 'nchannel_y', '_merge'])
df_added = df_added.rename(columns={'cc_x': 'cc', 'nchannel_x': 'nchannel'})

# LFEs present in both catalogs
# Cross correlation from FAME stations catalog
df1_merge = df1.merge(df2.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df1_both = df1_merge[df1_merge['_merge'] == 'both']
df1_both = df1_both.drop(columns=['cc_x', 'nchannel_x', '_merge'])
df1_both = df1_both.rename(columns={'cc_y': 'cc', 'nchannel_y': 'nchannel'})
df1_both = df1_both.astype({'year': int, 'month': int, 'day': int, \
    'hour': int, 'minute': int, 'second': float, \
    'cc': float, 'nchannel': int})

# LFEs present in both catalogs
# Cross correlation from permanent stations catalog
df2_merge = df2.merge(df1.drop_duplicates(), \
    on=['year', 'month', 'day', 'hour', 'minute', 'second'], \
    how='left', indicator=True)
df2_both = df2_merge[df2_merge['_merge'] == 'both']
df2_both = df2_both.drop(columns=['cc_x', 'nchannel_x', '_merge'])
df2_both = df2_both.rename(columns={'cc_y': 'cc', 'nchannel_y': 'nchannel'})
df2_both = df2_both.astype({'year': int, 'month': int, 'day': int, \
    'hour': int, 'minute': int, 'second': float, \
    'cc': float, 'nchannel': int})

# Cross correlation values
# Difference between LFEs added in permanent stations catalog
# and LFEs present in both catalogs
plt.figure(1, figsize=(20, 10))
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params) 
ax1 = plt.subplot(221)
plt.hist(df_added['cc'], \
    bins=[0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12, 0.125, \
          0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, \
          0.18, 0.185, 0.19, 0.195, 0.20, 0.205, 0.21, 0.215, 0.22])
plt.ylabel('Number of LFEs', fontsize=24)
plt.figtext(0.25, 0.8, '{:d} false detections'.format(len(df_added)), fontsize=16)
plt.figtext(0.25, 0.75, '(LFEs were not in FAME catalog)', fontsize=16)
plt.title('Permanent networks catalog', fontsize=24)
ax2 = plt.subplot(223)
plt.hist(df2_both['cc'], \
    bins=[0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12, 0.125, \
          0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, \
          0.18, 0.185, 0.19, 0.195, 0.20, 0.205, 0.21, 0.215, 0.22])
plt.xlabel('Cross-correlation value', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.figtext(0.25, 0.4, '{:d} true detections'.format(len(df2_both)), fontsize=16)
plt.figtext(0.25, 0.35, '(LFEs were in FAME catalog)', fontsize=16)
ax3 = plt.subplot(222)
plt.hist(df_missing['cc'], \
    bins=[0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12, 0.125, \
          0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, \
          0.18, 0.185, 0.19, 0.195, 0.20, 0.205, 0.21, 0.215, 0.22])
plt.ylabel('Number of LFEs', fontsize=24)
plt.figtext(0.64, 0.8, '{:d} detections missed'.format(len(df_missing)), fontsize=16)
plt.figtext(0.64, 0.75, '(LFEs are not in permanent networks catalog)', fontsize=16)
plt.title('FAME catalog', fontsize=24)
ax4 = plt.subplot(224)
plt.hist(df1_both['cc'], \
    bins=[0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12, 0.125, \
          0.13, 0.135, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.175, \
          0.18, 0.185, 0.19, 0.195, 0.20, 0.205, 0.21, 0.215, 0.22])
plt.xlabel('Cross-correlation value', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.figtext(0.64, 0.4, '{:d} detections reproduced'.format(len(df1_both)), fontsize=16)
plt.figtext(0.64, 0.35, '(LFEs are in permanent networks catalog)', fontsize=16)
plt.suptitle('Family {}'.format(filename), fontsize=24)
plt.savefig(figurename, format='eps')
ax1.clear()
ax2.clear()
ax3.clear()
ax4.clear()
plt.close(1)
