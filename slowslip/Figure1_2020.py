'''
Script to correlate wavelet details with simple waveform
'''

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from math import pi, sin

import correlate
import MODWT

N = 500
duration = 10
name = 'LA8'
J = 8

# Create time vector
time = np.arange(0, N + 1)

# Create displacement vector
disp = np.zeros(N + 1)
for i in range(0, N + 1):
    if (time[i] <= 0.5 * (N - duration)):
        disp[i] = time[i] / (N - duration)
    elif (time[i] >= 0.5 * (N + duration)):
        disp[i] = (time[i] - N) / (N - duration)
    else:
        disp[i] = (0.5 * N - time[i]) / duration           

# Compute MODWT
(W, V) = MODWT.pyramid(disp, name, J)
(D, S) = MODWT.get_DS(disp, W, name, J)

# Get maximum value
ymax = [np.max(np.abs(disp))]
for j in range(0, J):
    ymax.append(np.max(np.abs(D[j])))
ymax.append(np.max(np.abs(S[J])))

# Initialize figure
params = {'legend.fontsize': 20, \
          'xtick.labelsize':24, \
          'ytick.labelsize':24}
pylab.rcParams.update(params)
plt.figure(1, figsize=(24, 6))
plt.gcf().subplots_adjust(bottom=0.15)

# Draw time series
plt.subplot2grid((1, 3), (0, 0))
plt.axvline(250, color='grey')
plt.plot(time, disp, 'k', label='Data')
plt.xlim(np.min(time), np.max(time))
plt.ylim(- 1.1 * max(ymax), 1.1 * max(ymax))
plt.xlabel('Time (days)', fontsize=24)
plt.legend(loc=1)
        
# Plot 7th level detail
plt.subplot2grid((1, 3), (0, 1))
plt.axvline(250, color='grey')
plt.plot(time, D[6], 'k', label='D7')
plt.xlim(np.min(time), np.max(time))
plt.ylim(- 1.1 * max(ymax), 1.1 * max(ymax))
plt.xlabel('Time (days)', fontsize=24)
plt.legend(loc=1)

# Plot 8th level detail
plt.subplot2grid((1, 3), (0, 2))
plt.axvline(250, color='grey')
plt.plot(time, D[7], 'k', label='D8')
plt.xlim(np.min(time), np.max(time))
plt.ylim(- 1.1 * max(ymax), 1.1 * max(ymax))
plt.xlabel('Time (days)', fontsize=24)
plt.legend(loc=1)
           
# Save figure
plt.suptitle('Event size = {:d} days'.format(duration), fontsize=30)
plt.savefig('Figure1_2020.eps', format='eps')
plt.close(1)
