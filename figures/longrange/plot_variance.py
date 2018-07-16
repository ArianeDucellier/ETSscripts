"""
Figure to illustrate long range dependence
"""

import matplotlib.pylab as plt
import numpy as np

# Part 1: Long range dependence / bursts of LFEs
plt.figure(1, figsize=(30, 10))

data = np.array([0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 2, 2, 3, 3, \
                 3, 4, 4, 4, 5, \
                 5, 10, 10, 10, 5, \
                 5, 4, 4, 4, 3, \
                 3, 3, 2, 2, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 2, 2, 3, 3, \
                 3, 4, 4, 4, 5, \
                 5, 10, 10, 10, 5, \
                 5, 4, 4, 4, 3, \
                 3, 3, 2, 2, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0, \
                 0, 0, 0, 0, 0])

# 125 time windows
N = len(data)
ax1 = plt.subplot(131)
for i in range(0, N):
    plt.plot(np.array([i, i]), np.array([0, data[i]]), 'k-')
plt.xlabel('Time', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.xlim([-0.5, 124.5])
v = np.var(data)
plt.title('Window size = 1 - Variance = {:4.2f}'.format(v), fontsize=24)

# 25 time windows
data2 = np.reshape(data, (25, 5))
data2 = np.sum(data2, axis=1)
N = len(data2)
ax2 = plt.subplot(132)
for i in range(0, N):
    plt.plot(2.5 + 5 * np.array([i, i]), np.array([0, data2[i]]), 'k-')
plt.xlabel('Time', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.xlim([-0.5, 124.5])
v = np.var(data2)
plt.title('Window size = 5 - Variance = {:4.2f}'.format(v), fontsize=24)

# 5 time windows
data3 = np.reshape(data, (5, 25))
data3 = np.sum(data3, axis=1)
N = len(data3)
ax3 = plt.subplot(133)
for i in range(0, N):
    plt.plot(12.5 + 25 * np.array([i, i]), np.array([0, data3[i]]), 'k-')
plt.xlabel('Time', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.xlim([-0.5, 124.5])
v = np.var(data3)
plt.title('Window size = 25 - Variance = {:4.2f}'.format(v), fontsize=24)

# End and sae figure
plt.suptitle('Long range dependence', fontsize=24)
plt.savefig('longrange.eps', format='eps')
ax1.clear()
ax2.clear()
ax3.clear()
plt.close(1)

# Part 2: Homogeneous Poisson process
plt.figure(2, figsize=(30, 10))

NLFE = np.concatenate((np.repeat(0, 25), \
                       np.repeat(1, 41), \
                       np.repeat(2, 32), \
                       np.repeat(3, 17), \
                       np.repeat(4, 7), \
                       np.repeat(5, 2), \
                       np.repeat(6, 1)))
np.random.seed(seed=0)
ILFE = np.random.permutation(125)

data = np.zeros(125, dtype=int)
N = len(data)
for i in range(0, N):
    index = int(ILFE[i])
    data[index] = int(NLFE[i])

# 125 time windows
ax1 = plt.subplot(131)
for i in range(0, N):
    plt.plot(np.array([i, i]), np.array([0, data[i]]), 'k-')
plt.xlabel('Time', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.xlim([-0.5, 124.5])
v = np.var(data)
plt.title('Window size = 1 - Variance = {:4.2f}'.format(v), fontsize=24)

# 25 time windows
data2 = np.reshape(data, (25, 5))
data2 = np.sum(data2, axis=1)
N = len(data2)
ax2 = plt.subplot(132)
for i in range(0, N):
    plt.plot(2.5 + 5 * np.array([i, i]), np.array([0, data2[i]]), 'k-')
plt.xlabel('Time', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.xlim([-0.5, 124.5])
v = np.var(data2)
plt.title('Window size = 5 - Variance = {:4.2f}'.format(v), fontsize=24)

# 5 time windows
data3 = np.reshape(data, (5, 25))
data3 = np.sum(data3, axis=1)
N = len(data3)
ax3 = plt.subplot(133)
for i in range(0, N):
    plt.plot(12.5 + 25 * np.array([i, i]), np.array([0, data3[i]]), 'k-')
plt.xlabel('Time', fontsize=24)
plt.ylabel('Number of LFEs', fontsize=24)
plt.xlim([-0.5, 124.5])
v = np.var(data3)
plt.title('Window size = 25 - Variance = {:4.2f}'.format(v), fontsize=24)

# End and sae figure
plt.suptitle('Homogeneous Poisson', fontsize=24)
plt.savefig('poisson.eps', format='eps')
ax1.clear()
ax2.clear()
ax3.clear()
plt.close(2)
