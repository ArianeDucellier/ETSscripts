"""
Script to plot cumulative number of LFEs
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta

# Family 080326.08.015
filename = '080326.08.015'
catalog = 'catalog_200709-200906.pkl'
tbegin = datetime(2007, 9, 1)
nb_days = 638
x_arrows = np.array([16.5, 43.5, 69.5, 126.5, 206.5, 219.5, 232.5, 319.5, 376.5, 433.5, 466.5, 485.5, 511.5])
ind_arrows = np.asarray([17, 44, 70, 127, 207, 220, 233, 320, 377, 434, 467, 486, 512], dtype=int)

# Family 080421.14.048
#filename = '080421.14.048'
#catalog = 'catalog_200707-200912.pkl'
#tbegin = datetime(2007, 7, 1)
#nb_days = 701
#x_arrows = np.array([252.5, 295.5, 408.5, 470.5])
#ind_arrows = np.asarray([253, 297, 409, 471])

# Get data and filter
df_u = pickle.load(open('catalog_unknown/' + filename + '/' + catalog, 'rb'))
maxc = np.max(df_u['nchannel'])
df_u = df_u.loc[df_u['cc'] * df_u['nchannel'] >= 0.09 * maxc]

df_p = pickle.load(open('catalog_permanent/' + filename + '/' + catalog, 'rb'))
maxc = np.max(df_p['nchannel'])
df_p = df_p.loc[df_p['cc'] * df_p['nchannel'] >= 0.09 * maxc]

# Add date column
df = pd.DataFrame({'year': df_u['year'], 'month': df_u['month'], 'day': df_u['day']})
df = pd.to_datetime(df)
df_u['date'] = df

df = pd.DataFrame({'year': df_p['year'], 'month': df_p['month'], 'day': df_p['day']})
df = pd.to_datetime(df)
df_p['date'] = df

# Cumulative number of LFEs
cum_u = np.zeros(nb_days)
cum_p = np.zeros(nb_days)
for i in range(0, nb_days):
    df_sub = df_u[df_u['date'] < tbegin + timedelta(i + 1)]
    cum_u[i] = len(df_sub)
    df_sub = df_p[df_p['date'] < tbegin + timedelta(i + 1)]
    cum_p[i] = len(df_sub)

# Normalize
cum_u = cum_u / np.max(cum_u)
cum_p = cum_p / np.max(cum_p)

y_arrows = cum_p[ind_arrows] + 0.3
n_arrows = np.shape(x_arrows)[0]
dx_arrows = np.zeros(n_arrows)
dy_arrows = np.repeat(-0.1, n_arrows)

# Plot
params = {'legend.fontsize': 24, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)
plt.figure(1, figsize=(10, 10))
plt.plot(np.arange(0, nb_days), cum_u, 'r-', label='FAME ({:d} LFEs)'.format(len(df_u)))
plt.plot(np.arange(0, nb_days), cum_p + 0.1, 'b-', label='networks ({:d} LFEs)'.format(len(df_p)))
for (x, y, dx, dy) in zip(x_arrows, y_arrows, dx_arrows, dy_arrows):
    plt.arrow(x, y, dx, dy, fc='k', ec='k', head_width=10, head_length=0.03)
plt.yticks([], [])
plt.ylim([-0.05, 1.25])
plt.xlabel('Time (days) since {:02d}/{:02d}/{:04d}'. \
    format(tbegin.month, tbegin.day, tbegin.year), fontsize=24)
plt.ylabel('Normalized number of LFEs', fontsize=24)
plt.title('Cumulative number of LFEs', fontsize=24)
plt.legend(loc=4, fontsize=20)
plt.savefig('LFEdistribution_permanent/' + filename + '.eps', format='eps')
plt.close(1)
