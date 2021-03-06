"""
Script to make synthetic slow slip events and run MODWT on them
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from math import log, sqrt
from scipy.signal import butter, lfilter

import MODWT

# Durations of slow slip events
durations = [2, 5, 10, 20, 50, 100]
# Signal-to-noise ratios
SNRs = [1.0, 2.0 , 4.0]
# MODWT wavelet filter
names = ['Haar', 'D4', 'D6', 'D8', 'D10', 'D12', 'D14', 'D16', 'D18', 'D20', \
    'LA8', 'LA10', 'LA12', 'LA14', 'LA16', 'LA18', 'LA20', 'C6', 'C12', \
    'C18', 'C24', 'C30', 'BL14', 'BL18', 'BL20']
# Duration of recording
N = 500
# MODWT level
J = 8

# Create time vector
time = np.arange(0, N + 1)

# Set random seed
np.random.seed(0)

# Create low-pass filter
b, a = butter(4, 0.1, btype='low')

# Loop on signal-to-noise ratios
for SNR in SNRs:
    noise = np.random.normal(0.0, 1.0 / SNR, N + 1)
    # Loop on wavelet filters
    for name in names:
        # Create displacements vectors
        signals = []
        disps = []
        for duration in durations:
            signal = np.zeros(N + 1)
            for i in range(0, N + 1):
                if (time[i] <= 0.5 * (N - duration)):
                    signal[i] = time[i] / (N - duration)
                elif (time[i] >= 0.5 * (N + duration)):
                    signal[i] = (time[i] - N) / (N - duration)
                else:
                    signal[i] = (0.5 * N - time[i]) / duration           
            disp = signal + noise
            signals.append(signal)
            disps.append(disp)
        # Compute MODWT
        Ws = []
        Vs = []
        Ds = []
        Ss = []
        ymin = []
        ymax = []
        for i in range(0, len(disps)):
            disp = disps[i]
            (W, V) = MODWT.pyramid(disp, name, J)
            (D, S) = MODWT.get_DS(disp, W, name, J)
            Ws.append(W)
            Vs.append(V)
            Ds.append(D)
            Ss.append(S)
            maxD = max([np.max(Dj) for Dj in D])
            minD = min([np.min(Dj) for Dj in D])
            ymax.append(max(maxD, np.max(S[J])))
            ymin.append(min(minD, np.min(S[J])))
        # Thresholding of MODWT wavelet coefficients
        dispts = []
        for i in range(0, len(disps)):
            W = Ws[i]
            V = Vs[i]
            sigE = 1.0 / SNR
            Wt = []
            for j in range(1, J + 1):
                Wj = W[j - 1]
                deltaj = sqrt(2.0 * (sigE ** 2.0) * log(N + 1) / (2.0 ** j))
                Wjt = np.where(np.abs(Wj) >= deltaj, Wj, 0.0)
                if (j == J):
                    Vt = np.where(np.abs(V) >= deltaj, V, 0.0)
                Wt.append(Wjt)
            dispt = MODWT.inv_pyramid(Wt, Vt, name, J)
            dispts.append(dispt)
        # Low-pass filter
        dispfs = []
        for i in range(0, len(disps)):
            disp = disps[i]
            dispf = lfilter(b, a, disp)
            dispfs.append(dispf)
        # Plot MODWT
        params = {'xtick.labelsize':24,
                  'ytick.labelsize':24}
        pylab.rcParams.update(params)   
        fig = plt.figure(1, figsize=(15 * len(durations), 5 * (J + 3)))
        # Plot denoised data
        for i in range(0, len(dispts)):
            plt.subplot2grid((J + 3, len(durations)), (J + 2, i))
            signal = signals[i]
            dispt = dispts[i]
            dispf = dispfs[i]
            plt.plot(time, signal, 'k', label='Signal', linewidth=3)
            plt.plot(time, dispt, 'r', label='Denoised', linewidth=3)
            plt.plot(time, dispf, 'grey', label='Low-pass filtered')
            plt.xlabel('Time (days)', fontsize=30)
            plt.ylim(-2.0, 2.0)
            plt.legend(loc=3, fontsize=14)
        # Plot data
        for i in range(0, len(disps)):
            plt.subplot2grid((J + 3, len(durations)), (J + 1, i))
            disp = disps[i]
            plt.plot(time, disp, 'k', label='Data')
            plt.ylim(-2.0, 2.0)
            plt.legend(loc=3, fontsize=14)
        # Plot details
        for i in range(0, len(disps)):
            for j in range(0, J):
                plt.subplot2grid((J + 3, len(durations)), (J - j, i))
                D = Ds[i]
                plt.plot(time, D[j], 'k', label='D' + str(j + 1))
                plt.ylim(-2.0, 2.0)
                plt.legend(loc=3, fontsize=14)
        # Plot smooth
        for i in range(0, len(disps)):
            plt.subplot2grid((J + 3, len(durations)), (0, i))
            S = Ss[i]
            plt.plot(time, S[J], 'k', label='S' + str(J))
            plt.ylim(-2.0, 2.0)
            plt.legend(loc=3, fontsize=14)
            plt.title('Duration = ' + str(durations[i]), fontsize=30)
        # Save figure
        title = 'Wavelet = ' + name + ' - SNR = ' + str(SNR)
        plt.suptitle(title, fontsize=50)
        plt.savefig('synthetics/' + name + '_' + str(SNR) + '_DS.eps', \
            format='eps')
        plt.close(1)
