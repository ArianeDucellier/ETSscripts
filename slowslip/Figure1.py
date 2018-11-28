"""
NASA proposal - Figure 1
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from math import log, sqrt

import MODWT

# Durations of slow slip events
duration1 = 2
duration2 = 5

# Signal-to-noise ratios
SNR1 = 2.0
SNR2 = 4.0

# MODWT wavelet filter
name = 'LA8'

# Duration of recording
N = 500

# MODWT level
J = 8

# Create time vector
time = np.arange(0, N + 1)

# Set random seed
np.random.seed(0)

# Create figure
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params)   
fig = plt.figure(1, figsize=(15, 10))

# Duration 1 - Noise level 1
noise = np.random.normal(0.0, 1.0 / SNR1, N + 1)
signal = np.zeros(N + 1)
for i in range(0, N + 1):
    if (time[i] <= 0.5 * (N - duration1)):
        signal[i] = time[i] * SNR1 / (N - duration1)
    elif (time[i] >= 0.5 * (N + duration1)):
        signal[i] = (time[i] - N) * SNR1 / (N - duration1)
    else:
        signal[i] = (0.5 * N - time[i]) * SNR1 / duration1           
disp = signal + noise
(W, V) = MODWT.pyramid(disp, name, J)
(D, S) = MODWT.get_DS(disp, W, name, J)
sigE = 1.0 / SNR1
Wt = []
for j in range(1, J + 1):
    Wj = W[j - 1]
    deltaj = sqrt(2.0 * sigE * log(N + 1) / (2.0 ** j))
    Wjt = np.where(Wj >= deltaj, Wj, 0.0)
    if (j == J):
        Vt = np.where(V >= deltaj, V, 0.0)
    Wt.append(Wjt)
dispt = MODWT.inv_pyramid(Wt, Vt, name, J)
# Plot
plt.subplot2grid((2, 4), (0, 0))
plt.plot(time, disp, 'k', label='Data')
plt.legend(loc=3, fontsize=14)
plt.title('Duration = {}, SNR = {}'.format(duration1, SNR1), fontsize=14)
plt.subplot2grid((2, 4), (1, 0))
plt.plot(time, signal, 'grey', label='Signal')
plt.plot(time, dispt, 'k', label='Denoised')
plt.xlabel('Time', fontsize=14)
plt.legend(loc=3, fontsize=14)

# Duration 1 - Noise level 2
noise = np.random.normal(0.0, 1.0 / SNR2, N + 1)
signal = np.zeros(N + 1)
for i in range(0, N + 1):
    if (time[i] <= 0.5 * (N - duration1)):
        signal[i] = time[i] * SNR2 / (N - duration1)
    elif (time[i] >= 0.5 * (N + duration1)):
        signal[i] = (time[i] - N) * SNR2 / (N - duration1)
    else:
        signal[i] = (0.5 * N - time[i]) * SNR2 / duration1           
disp = signal + noise
(W, V) = MODWT.pyramid(disp, name, J)
(D, S) = MODWT.get_DS(disp, W, name, J)
sigE = 1.0 / SNR2
Wt = []
for j in range(1, J + 1):
    Wj = W[j - 1]
    deltaj = sqrt(2.0 * sigE * log(N + 1) / (2.0 ** j))
    Wjt = np.where(Wj >= deltaj, Wj, 0.0)
    if (j == J):
        Vt = np.where(V >= deltaj, V, 0.0)
    Wt.append(Wjt)
dispt = MODWT.inv_pyramid(Wt, Vt, name, J)
# Plot
plt.subplot2grid((2, 4), (0, 1))
plt.plot(time, disp, 'k', label='Data')
plt.legend(loc=3, fontsize=14)
plt.title('Duration = {}, SNR = {}'.format(duration1, SNR2), fontsize=14)
plt.subplot2grid((2, 4), (1, 1))
plt.plot(time, signal, 'grey', label='Signal')
plt.plot(time, dispt, 'k', label='Denoised')
plt.xlabel('Time', fontsize=14)
plt.legend(loc=3, fontsize=14)

# Duration 2 - Noise level 1
noise = np.random.normal(0.0, 1.0 / SNR1, N + 1)
signal = np.zeros(N + 1)
for i in range(0, N + 1):
    if (time[i] <= 0.5 * (N - duration2)):
        signal[i] = time[i] * SNR1 / (N - duration2)
    elif (time[i] >= 0.5 * (N + duration1)):
        signal[i] = (time[i] - N) * SNR1 / (N - duration2)
    else:
        signal[i] = (0.5 * N - time[i]) * SNR1 / duration2           
disp = signal + noise
(W, V) = MODWT.pyramid(disp, name, J)
(D, S) = MODWT.get_DS(disp, W, name, J)
sigE = 1.0 / SNR1
Wt = []
for j in range(1, J + 1):
    Wj = W[j - 1]
    deltaj = sqrt(2.0 * sigE * log(N + 1) / (2.0 ** j))
    Wjt = np.where(Wj >= deltaj, Wj, 0.0)
    if (j == J):
        Vt = np.where(V >= deltaj, V, 0.0)
    Wt.append(Wjt)
dispt = MODWT.inv_pyramid(Wt, Vt, name, J)
# Plot
plt.subplot2grid((2, 4), (0, 2))
plt.plot(time, disp, 'k', label='Data')
plt.legend(loc=3, fontsize=14)
plt.title('Duration = {}, SNR = {}'.format(duration2, SNR1), fontsize=14)
plt.subplot2grid((2, 4), (1, 2))
plt.plot(time, signal, 'grey', label='Signal')
plt.plot(time, dispt, 'k', label='Denoised')
plt.xlabel('Time', fontsize=14)
plt.legend(loc=3, fontsize=14)

# Duration 2 - Noise level 2
noise = np.random.normal(0.0, 1.0 / SNR2, N + 1)
signal = np.zeros(N + 1)
for i in range(0, N + 1):
    if (time[i] <= 0.5 * (N - duration2)):
        signal[i] = time[i] * SNR2 / (N - duration2)
    elif (time[i] >= 0.5 * (N + duration1)):
        signal[i] = (time[i] - N) * SNR2 / (N - duration2)
    else:
        signal[i] = (0.5 * N - time[i]) * SNR2 / duration2           
disp = signal + noise
(W, V) = MODWT.pyramid(disp, name, J)
(D, S) = MODWT.get_DS(disp, W, name, J)
sigE = 1.0 / SNR2
Wt = []
for j in range(1, J + 1):
    Wj = W[j - 1]
    deltaj = sqrt(2.0 * sigE * log(N + 1) / (2.0 ** j))
    Wjt = np.where(Wj >= deltaj, Wj, 0.0)
    if (j == J):
        Vt = np.where(V >= deltaj, V, 0.0)
    Wt.append(Wjt)
dispt = MODWT.inv_pyramid(Wt, Vt, name, J)
# Plot
plt.subplot2grid((2, 4), (0, 3))
plt.plot(time, disp, 'k', label='Data')
plt.legend(loc=3, fontsize=14)
plt.title('Duration = {}, SNR = {}'.format(duration2, SNR2), fontsize=14)
plt.subplot2grid((2, 4), (1, 3))
plt.plot(time, signal, 'grey', label='Signal')
plt.plot(time, dispt, 'k', label='Denoised')
plt.xlabel('Time', fontsize=14)
plt.legend(loc=3, fontsize=14)

plt.savefig('Figure1.eps', format='eps')
plt.close(1)
