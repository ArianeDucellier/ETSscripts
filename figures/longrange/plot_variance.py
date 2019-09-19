"""
Script to make figure illustrating the presence or absence of
long-range dependence for different time series
"""
import matplotlib.pylab as pylab
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle

filename = 'FARIMA'

# Initialization
data = np.loadtxt(filename + '.txt')
m = np.zeros(6)
variance = np.zeros(6)

# Aggregate one time window (initial time series)
N = len(data)
m[0] = 1
variance[0] = np.var(data)

# Aggregate 3 time windows
data1 = np.reshape(data, (1215, 3))
data1 = np.mean(data1, axis=1)
N1 = len(data1)
m[1] = 3
variance[1] = np.var(data1)

# Aggregate 9 time windows
data2 = np.reshape(data, (405, 9))
data2 = np.mean(data2, axis=1)
N2 = len(data2)
m[2] = 9
variance[2] = np.var(data2)

# Aggregate 27 time windows
data3 = np.reshape(data, (135, 27))
data3 = np.mean(data3, axis=1)
N3 = len(data3)
m[3] = 27
variance[3] = np.var(data3)

# Aggregate 81 time windows
data4 = np.reshape(data, (45, 81))
data4 = np.mean(data4, axis=1)
N4 = len(data4)
m[4] = 81
variance[4] = np.var(data4)

# Aggregate 243 time windows
data5 = np.reshape(data, (15, 243))
data5 = np.mean(data5, axis=1)
N5 = len(data5)
m[5] = 243
variance[5] = np.var(data5)

# Save variance
df = pd.DataFrame(data={'m':m, 'var':variance})
pickle.dump(df, open(filename + '.pkl', 'wb'))

# One big figure
plt.figure(1, figsize=(36, 10))
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params)

ax1 = plt.subplot(231)
for i in range(0, N):
    plt.plot(0.5 + np.array([i, i]), np.array([0, data[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.text(3645.5 / 2.0, np.max(data), 'Sample size = 1', \
    horizontalalignment='center', verticalalignment='top', fontsize=20)

ax2 = plt.subplot(232)
for i in range(0, N1):
    plt.plot(1.5 + 3 * np.array([i, i]), np.array([0, data1[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.text(3645.5 / 2.0, np.max(data1), 'Sample size = 3', \
    horizontalalignment='center', verticalalignment='top', fontsize=20)

ax3 = plt.subplot(233)
for i in range(0, N2):
    plt.plot(4.5 + 9 * np.array([i, i]), np.array([0, data2[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.text(3645.5 / 2.0, np.max(data2), 'Sample size = 9', \
    horizontalalignment='center', verticalalignment='top', fontsize=20)

ax4 = plt.subplot(234)
for i in range(0, N3):
    plt.plot(13.5 + 27 * np.array([i, i]), np.array([0, data3[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.text(3645.5 / 2.0, np.max(data3), 'Sample size = 27', \
    horizontalalignment='center', verticalalignment='top', fontsize=20)

ax5 = plt.subplot(235)
for i in range(0, N4):
    plt.plot(40.5 + 81 * np.array([i, i]), np.array([0, data4[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.text(3645.5 / 2.0, np.max(data4), 'Sample size = 81', \
    horizontalalignment='center', verticalalignment='top', fontsize=20)

ax6 = plt.subplot(236)
for i in range(0, N5):
    plt.plot(121.5 + 243 * np.array([i, i]), np.array([0, data5[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.text(3645.5 / 2.0, np.max(data5), 'Sample size = 243', \
    horizontalalignment='center', verticalalignment='top', fontsize=20)

plt.suptitle(filename + ' model', fontsize=24)
plt.savefig(filename + '.eps', format='eps')
ax1.clear()
ax2.clear()
ax3.clear()
ax4.clear()
ax5.clear()
ax6.clear()
plt.close(1)

# Six individual figures
params = {'xtick.labelsize':12,
          'ytick.labelsize':12}
pylab.rcParams.update(params)

plt.figure(1, figsize=(12, 5))
for i in range(0, N):
    plt.plot(0.5 + np.array([i, i]), np.array([0, data[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=16)
plt.ylabel('Average # of events', fontsize=16)
plt.xlim([0.0, 3645.0])
plt.title(filename + ' model - Sample size = 1', fontsize=20)
plt.savefig(filename + '_1.eps', format='eps')
plt.close(1)

plt.figure(2, figsize=(12, 5))
for i in range(0, N1):
    plt.plot(1.5 + 3 * np.array([i, i]), np.array([0, data1[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.title(filename + ' model - Sample size = 3', fontsize=20)
plt.savefig(filename + '_2.eps', format='eps')
plt.close(2)

plt.figure(3, figsize=(12, 5))
for i in range(0, N2):
    plt.plot(4.5 + 9 * np.array([i, i]), np.array([0, data2[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.title(filename + ' model - Sample size = 9', fontsize=20)
plt.savefig(filename + '_3.eps', format='eps')
plt.close(3)

plt.figure(4, figsize=(12, 5))
for i in range(0, N3):
    plt.plot(13.5 + 27 * np.array([i, i]), np.array([0, data3[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.title(filename + ' model - Sample size = 27', fontsize=20)
plt.savefig(filename + '_4.eps', format='eps')
plt.close(4)

plt.figure(5, figsize=(12, 5))
for i in range(0, N4):
    plt.plot(40.5 + 81 * np.array([i, i]), np.array([0, data4[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.title(filename + ' model - Sample size = 81', fontsize=20)
plt.savefig(filename + '_5.eps', format='eps')
plt.close(5)

plt.figure(6, figsize=(12, 5))
for i in range(0, N5):
    plt.plot(121.5 + 243 * np.array([i, i]), np.array([0, data5[i]]), 'k-')
plt.xlabel('Time (minutes)', fontsize=20)
plt.ylabel('Average # of events', fontsize=20)
plt.xlim([0.0, 3645.0])
plt.title(filename + ' model - Sample size = 243', fontsize=20)
plt.savefig(filename + '_6.eps', format='eps')
plt.close(6)
