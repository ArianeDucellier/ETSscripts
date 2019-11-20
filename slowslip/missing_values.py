""" Script to look at MODWT of GPS data
with defferent methods to fill in missing values """

import matplotlib.pyplot as plt
import numpy as np

from math import log, sqrt

import DWT, MODWT
from MODWT import get_DS, get_scaling, inv_pyramid, pyramid

# Choose the station
station = 'PGC5'
direction = 'lon'
dataset = 'cleaned'
filename = '../data/PANGA/' + dataset + '/' + station + '.' + direction
# Load the data
data = np.loadtxt(filename, skiprows=26)
time = data[:, 0]
disp = data[:, 1]
error = data[:, 2]
# Correct for the repeated value
dt = np.diff(time)
gap = np.where(dt < 1.0 / 365.0 - 0.001)[0]
time[gap[0] + 1] = time[gap[0] + 2]
time[gap[0] + 2] = 0.5 * (time[gap[0] + 2] + time[gap[0] + 3])
# Look for gaps greater than 1 day
days = 1
dt = np.diff(time)
gap = np.where(dt > days / 365.0 + 0.001)[0]
# Select a subset of the data without gaps
ibegin = 2943
iend = 4333
time = time[ibegin + 1 : iend + 1]
disp = disp[ibegin + 1 : iend + 1]
error = error[ibegin + 1 : iend + 1]
sigE2 = np.mean(np.square(error))
N = np.shape(disp)[0]

# Parameters
name = 'LA8'
g = MODWT.get_scaling(name)
L = len(g)
J = 6
nmax = 15

# Compute MODWT
[W0, V0] = pyramid(disp, name, J)
(nuH, nuG) = DWT.get_nu(name, J)
(D0, S0) = get_DS(disp, W0, name, J)
W0t = []
for j in range(1, J + 1):
    W0j = W0[j - 1]
    deltaj = sqrt(2.0 * sigE2 * log(N) / (2.0 ** j))
    W0jt = np.where(np.abs(W0j) >= deltaj, W0j, 0.0)
    if (j == J):
        V0t = np.where(np.abs(V0) >= deltaj, V0, 0.0)
    W0t.append(W0jt)
dispt = inv_pyramid(W0t, V0t, name, J)

# Loop on missing values
gaps = [614, 797]
for gap in gaps:
    # Loop on length of gap
    for n in range(1, nmax):
        disp_interp = np.copy(disp)
        # Remove points, replace by interpolation
        for i in range(0, n):
            disp_interp[gap + i] = disp_interp[gap - 1] + (disp_interp[gap + n] - disp_interp[gap - 1]) * (i + 1) / (n + 1)
        # Compute MODWT
        [W, V] = pyramid(disp_interp, name, J)
        (D, S) = get_DS(disp_interp, W, name, J)
        Wt = []
        for j in range(1, J + 1):
            Wj = W[j - 1]
            deltaj = sqrt(2.0 * sigE2 * log(N) / (2.0 ** j))
            Wjt = np.where(np.abs(Wj) >= deltaj, Wj, 0.0)
            if (j == J):
                Vt = np.where(np.abs(V) >= deltaj, V, 0.0)
            Wt.append(Wjt)
        dispt_interp = inv_pyramid(Wt, Vt, name, J)
        # Figure wavelet coefficients
        plt.figure(1, figsize=(15, 24))
        # Plot data
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        plt.plot(time, disp_interp, 'r')
        plt.plot(time, disp, 'k', label='Data')
        plt.xlim(np.min(time[gap - 50]), np.max(time[gap + n + 49]))
        plt.legend(loc=1)
        # Plot wavelet coefficients at each level
        for j in range(1, J + 1):
            plt.subplot2grid((J + 2, 1), (J + 1 - j, 0))
            Wj = W[j - 1]
            plt.plot(time, np.roll(Wj, nuH[j - 1]), 'r')
            Wj = W0[j - 1]
            plt.plot(time, np.roll(Wj, nuH[j - 1]), 'k', label = 'W' + str(j))
            Lj = (2 ** j - 1) * (L - 1) + 1
            plt.axvline(time[Lj - 2 - abs(nuH[j - 1])], linewidth=1, color='blue')
            plt.axvline(time[N - abs(nuH[j - 1])], linewidth=1, color='blue')
            plt.xlim(np.min(time[gap - 50]), np.max(time[gap + n + 49]))
            plt.legend(loc=1)
        # Plot scaling coefficients for the last level
        plt.subplot2grid((J + 2, 1), (0, 0))
        plt.plot(time, np.roll(V, nuG[J - 1]), 'r')
        plt.plot(time, np.roll(V0, nuG[J - 1]), 'k', label = 'V' + str(J))
        Lj = (2 ** J - 1) * (L - 1) + 1
        plt.axvline(time[Lj - 2 - abs(nuG[J - 1])], linewidth=1, color='blue')
        plt.axvline(time[N - abs(nuG[J - 1])], linewidth=1, color='blue')
        plt.xlim(np.min(time[gap - 50]), np.max(time[gap + n + 49]))
        plt.legend(loc=1)
        plt.savefig('missing_values/WV_' + str(gap) + '_' + str(n) + '.eps', format='eps')
        plt.close(1)
        # Figure wavelet details
        plt.figure(2, figsize=(15, 24))
        # Plot data
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        plt.plot(time, disp_interp, 'r')
        plt.plot(time, disp, 'k', label='Data')
        plt.xlim(np.min(time[gap - 50]), np.max(time[gap + n + 49]))
        plt.legend(loc=1)
        # Plot wavelet coefficients at each level
        for j in range(0, J):
            plt.subplot2grid((J + 2, 1), (J + 1 - j, 0))
            plt.plot(time, D[j], 'r')
            plt.plot(time, D0[j], 'k', label='D' + str(j + 1))
            Lj = (2 ** (j + 1) - 1) * (L - 1) + 1
            plt.axvline(time[Lj - 2], linewidth=1, color='blue')
            plt.axvline(time[N - Lj + 1], linewidth=1, color='blue')
            plt.xlim(np.min(time[gap - 50]), np.max(time[gap + n + 49]))
            plt.legend(loc=1)
        # Plot scaling coefficients for the last level
        plt.subplot2grid((J + 2, 1), (0, 0))
        plt.plot(time, S[J], 'r')
        plt.plot(time, S0[J], 'k', label='S' + str(J))
        Lj = (2 ** 6 - 1) * (L - 1) + 1
        plt.axvline(time[Lj - 2], linewidth=1, color='blue')
        plt.axvline(time[N - Lj + 1], linewidth=1, color='blue')
        plt.xlim(np.min(time[gap - 50]), np.max(time[gap + n + 49]))
        plt.legend(loc=1)
        plt.savefig('missing_values/DS_' + str(gap) + '_' + str(n) + '.eps', format='eps')
        plt.close(2)
        # Figure denoising
        plt.figure(2, figsize=(15, 6))
        # Plot data
        plt.subplot2grid((2, 1), (1, 0))
        plt.plot(time, disp_interp, 'r')
        plt.plot(time, disp, 'k', label='Data')
        plt.xlim(np.min(time[gap - 150]), np.max(time[gap + n + 149]))
        plt.legend(loc=1)
        # Plot denoised signal
        plt.subplot2grid((2, 1), (0, 0))
        plt.plot(time, dispt_interp, 'r')
        plt.plot(time, dispt, 'k', label='Denoised')
        plt.xlim(np.min(time[gap - 150]), np.max(time[gap + n + 149]))
        plt.legend(loc=1)
        plt.savefig('missing_values/denoised_' + str(gap) + '_' + str(n) + '.eps', format='eps')
        plt.close(3)
