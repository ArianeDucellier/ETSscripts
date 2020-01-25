'''
Script to correlate wavelet details with simple waveform
'''

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

from math import pi, sin

import correlate
import MODWT

def synthetic(duration, name, N, J):
    """
    """
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
    plt.figure(1, figsize=(15, 3 * (J + 2)))

    # Draw time series
    plt.subplot2grid((J + 2, 1), (J + 1, 0))
    plt.plot(time, disp, 'k', label='Data')
    plt.xlim(np.min(time), np.max(time))
    plt.ylim(- 1.1 * max(ymax), 1.1 * max(ymax))
    plt.xlabel('Time (days)', fontsize=24)
    plt.legend(loc=1)
        
    # Plot details at each level
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j, 0))
        plt.plot(time, D[j], 'k', label='D' + str(j + 1))
        plt.xlim(np.min(time), np.max(time))
        plt.ylim(- 1.1 * max(ymax), 1.1 * max(ymax))
        plt.legend(loc=1)

    # Plot smooth for the last level
    plt.subplot2grid((J + 2, 1), (0, 0))
    plt.plot(time, S[J], 'k', label='S' + str(J))
    plt.xlim(np.min(time), np.max(time))
    plt.ylim(- 1.1 * max(ymax), 1.1 * max(ymax))
    plt.legend(loc=1)
        
    # Save figure
    plt.suptitle('Event size = {:d} days'.format(duration), fontsize=30)
    plt.savefig('correlation/synthetic_' + str(duration) + '_DS.eps', \
        format='eps')
    plt.close(1)

    # Return times series and details
    return(time, disp, D, S)

def correlation(time0, disp0, D0, S0, stations, direction, dataset, \
    lats, lat0, name, J, slowness, event_size, window, D0_78, D0_678, D0_67):
    """
    """
    times = []
    disps = []
    Ws = []
    Vs = []
    Ds = []
    Ss = []

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
        # MODWT
        [W, V] = MODWT.pyramid(disp, name, J)
        Ws.append(W)
        Vs.append(V)
        (D, S) = MODWT.get_DS(disp, W, name, J)
        Ds.append(D)
        Ss.append(S)

    # Subset
    tbegin = []
    tend = []
    for time in times:
        tbegin.append(np.min(time))
        tend.append(np.max(time))
    tmin = max(tbegin)
    tmax = min(tend)

    # Conversion latitude -> kms
    a = 6378.136
    e = 0.006694470
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)

    # Initialize figure
    params = {'legend.fontsize': 20, \
              'xtick.labelsize':24, \
              'ytick.labelsize':24}
    pylab.rcParams.update(params)   
    plt.figure(1, figsize=(15, 3 * (J + 2)))

    # Correlate time series
    plt.subplot2grid((J + 2, 1), (J + 1, 0))

    times_subset = []
    disp_subset = []
    for (time, disp) in zip(times, disps):
        ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
        iend = np.where(np.abs(time - tmax) < 0.001)[0]
        times_subset.append(time[ibegin[0] : iend[0] + 1])
        disp_subset.append(disp[ibegin[0] : iend[0] + 1])

    # Stack
    stack = np.zeros(len(times_subset[0]))
    latmin = min(lats)
    for (time, disp, lat) in zip(times_subset, disp_subset, lats):
        disp_interp = np.interp(time + slowness * (lat - lat0), time, disp)
        stack = stack + disp_interp

    # Correlate
    cc = correlate.optimized(disp0, stack)
    M = len(disp0)
    N = len(stack)
    index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
    time = times_subset[0][index]

    # Plot
    plt.plot(time, cc, 'k', label='Data')
    plt.xlim(np.min(time), np.max(time))
    plt.ylim(- 1.0, 1.0)
    plt.xlabel('Time (days)', fontsize=24)
    plt.legend(loc=1)

    # Correlate details
    for j in range(0, J):
        plt.subplot2grid((J + 2, 1), (J - j, 0))

        times_subset = []
        Dj_subset = []
        for (time, D) in zip(times, Ds):
            ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
            iend = np.where(np.abs(time - tmax) < 0.001)[0]
            times_subset.append(time[ibegin[0] : iend[0] + 1])
            Dj_subset.append(D[j][ibegin[0] : iend[0] + 1])

        # Stack
        stack = np.zeros(len(times_subset[0]))
        latmin = min(lats)
        for (time, Dj, lat) in zip(times_subset, Dj_subset, lats):
            Dj_interp = np.interp(time + slowness * (lat - lat0), time, Dj)
            stack = stack + Dj_interp

        # Correlate
        cc = correlate.optimized(D0[j], stack)
        M = len(D0[j])
        N = len(stack)
        index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
        time = times_subset[0][index]

        # Plot
        plt.plot(time, cc, 'k', label='D' + str(j + 1))
        plt.xlim(np.min(time), np.max(time))
        plt.ylim(- 1.0, 1.0)
        plt.legend(loc=1)

    # Correlate smooth
    plt.subplot2grid((J + 2, 1), (0, 0))

    times_subset = []
    SJ_subset = []
    for (time, S) in zip(times, Ss):
        ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
        iend = np.where(np.abs(time - tmax) < 0.001)[0]
        times_subset.append(time[ibegin[0] : iend[0] + 1])
        SJ_subset.append(S[J][ibegin[0] : iend[0] + 1])

    # Stack
    stack = np.zeros(len(times_subset[0]))
    latmin = min(lats)
    for (time, SJ, lat) in zip(times_subset, SJ_subset, lats):
        SJ_interp = np.interp(time + slowness * (lat - lat0), time, SJ)
        stack = stack + SJ_interp

    # Correlate
    cc = correlate.optimized(S0[J], stack)
    M = len(S0[J])
    N = len(stack)
    index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
    time = times_subset[0][index]

    # Plot
    plt.plot(time, cc, 'k', label='S' + str(J))
    plt.xlim(np.min(time), np.max(time))
    plt.ylim(- 1.0, 1.0)
    plt.legend(loc=1)

    # Save figure
    plt.suptitle('Event size = {:d} days'.format(event_size), fontsize=30)
    plt.savefig('correlation/correlation_' + str(event_size) + \
        '_' + str(window) + '_DS.eps', format='eps')
    plt.close(1)

    # Plot sum of details 7 and 8
    params = {'legend.fontsize': 20, \
              'xtick.labelsize':24, \
              'ytick.labelsize':24}
    pylab.rcParams.update(params)   
    plt.figure(1, figsize=(15, 3))

    times_subset = []
    D_78_subset = []
    for (time, D) in zip(times, Ds):
        ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
        iend = np.where(np.abs(time - tmax) < 0.001)[0]
        times_subset.append(time[ibegin[0] : iend[0] + 1])
        D_78_subset.append(D[6][ibegin[0] : iend[0] + 1] + D[7][ibegin[0] : iend[0] + 1])

    # Stack
    stack = np.zeros(len(times_subset[0]))
    latmin = min(lats)
    for (time, D_78, lat) in zip(times_subset, D_78_subset, lats):
        D_78_interp = np.interp(time + slowness * (lat - lat0), time, D_78)
        stack = stack + D_78_interp

    # Correlate
    cc = correlate.optimized(D0_78, stack)
    M = len(D0_78)
    N = len(stack)
    index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
    time = times_subset[0][index]

    # Plot
    plt.plot(time, cc, 'k', label='D7 + D8')
    plt.xlim(np.min(time), np.max(time))
    plt.ylim(- 1.0, 1.0)
    plt.xlabel('Time (days)', fontsize=24)
    plt.legend(loc=1)

    # Save figure
    plt.suptitle('Event size = {:d} days'.format(event_size), fontsize=30)
    plt.savefig('correlation/correlation_' + str(event_size) + \
        '_' + str(window) + '_D78.eps', format='eps')
    plt.close(1)

    # Plot sum of details 6, 7 and 8
    params = {'legend.fontsize': 20, \
              'xtick.labelsize':24, \
              'ytick.labelsize':24}
    pylab.rcParams.update(params)   
    plt.figure(1, figsize=(15, 3))

    times_subset = []
    D_678_subset = []
    for (time, D) in zip(times, Ds):
        ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
        iend = np.where(np.abs(time - tmax) < 0.001)[0]
        times_subset.append(time[ibegin[0] : iend[0] + 1])
        D_678_subset.append(D[5][ibegin[0] : iend[0] + 1] + D[6][ibegin[0] : iend[0] + 1] + D[7][ibegin[0] : iend[0] + 1])

    # Stack
    stack = np.zeros(len(times_subset[0]))
    latmin = min(lats)
    for (time, D_678, lat) in zip(times_subset, D_678_subset, lats):
        D_678_interp = np.interp(time + slowness * (lat - lat0), time, D_678)
        stack = stack + D_678_interp

    # Correlate
    cc = correlate.optimized(D0_678, stack)
    M = len(D0_678)
    N = len(stack)
    index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
    time = times_subset[0][index]

    # Plot
    plt.plot(time, cc, 'k', label='D6 + D7 + D8')
    plt.xlim(np.min(time), np.max(time))
    plt.ylim(- 1.0, 1.0)
    plt.xlabel('Time (days)', fontsize=24)
    plt.legend(loc=1)

    # Save figure
    plt.suptitle('Event size = {:d} days'.format(event_size), fontsize=30)
    plt.savefig('correlation/correlation_' + str(event_size) + \
        '_' + str(window) + '_D678.eps', format='eps')
    plt.close(1)

    # Plot sum of details 6 and 7
    params = {'legend.fontsize': 20, \
              'xtick.labelsize':24, \
              'ytick.labelsize':24}
    pylab.rcParams.update(params)   
    plt.figure(1, figsize=(15, 3))

    times_subset = []
    D_67_subset = []
    for (time, D) in zip(times, Ds):
        ibegin = np.where(np.abs(time - tmin) < 0.001)[0]
        iend = np.where(np.abs(time - tmax) < 0.001)[0]
        times_subset.append(time[ibegin[0] : iend[0] + 1])
        D_67_subset.append(D[5][ibegin[0] : iend[0] + 1] + D[6][ibegin[0] : iend[0] + 1])

    # Stack
    stack = np.zeros(len(times_subset[0]))
    latmin = min(lats)
    for (time, D_67, lat) in zip(times_subset, D_67_subset, lats):
        D_67_interp = np.interp(time + slowness * (lat - lat0), time, D_67)
        stack = stack + D_67_interp

    # Correlate
    cc = correlate.optimized(D0_67, stack)
    M = len(D0_67)
    N = len(stack)
    index = int((M - 1) / 2) + np.arange(0, N - M + 1, dtype='int')
    time = times_subset[0][index]

    # Plot
    plt.plot(time, cc, 'k', label='D6 + D7')
    plt.xlim(np.min(time), np.max(time))
    plt.ylim(- 1.0, 1.0)
    plt.xlabel('Time (days)', fontsize=24)
    plt.legend(loc=1)

    # Save figure
    plt.suptitle('Event size = {:d} days'.format(event_size), fontsize=30)
    plt.savefig('correlation/correlation_' + str(event_size) + \
        '_' + str(window) + '_D67.eps', format='eps')
    plt.close(1)

if __name__ == '__main__':

    # Parameters
    name = 'LA8'
    N = 500
    J = 8
    stations = ['ALBH', 'CHCM', 'COUP', 'PGC5', 'SC02', 'SC03', 'UFDA', 'FRID', 'PNCL', 'SQIM']
    direction = 'lon'
    dataset = 'cleaned'
    lats = [48.2323, 48.0106, 48.2173, 48.6483, 48.5462, 47.8166, 47.7550, 48.5352, 48.1014, 48.0823]
    lat0 = 48.1168
    slowness = 0.0

    for duration in [1, 2, 5, 10, 20, 50]:

        # Synthetic wavelet
        (time, disp, D, S) = synthetic(duration, name, N, J)

        for N0 in [20, 50, 100, 200]:

            # Take only the center
            imin = int((N - N0) / 2)
            imax = int((N + N0) / 2)
            time0 = time[imin : (imax + 1)]
            disp0 = disp[imin : (imax + 1)]
            D0 = []
            for Dj in D:
                D0.append(Dj[imin : (imax + 1)])
            S0 = []
            for Sj in S:
                S0.append(Sj[imin : (imax + 1)])

            # Sum of details
            D0_78 = D0[6] + D0[7]
            D0_678 = D0[5] + D0[6] + D0[7]
            D0_67 = D0[5] + D0[6]

            # Correlate with stack of wavelet details
            correlation(time0, disp0, D0, S0, stations, direction, dataset, \
                lats, lat0, name, J, slowness, duration, N0, D0_78, D0_678, D0_67)
