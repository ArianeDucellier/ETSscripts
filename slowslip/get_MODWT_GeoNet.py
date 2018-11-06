"""
This module contains functions to carry out a MODWT analysis of the
displacement measured at a GPS station
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn import linear_model

import DWT, MODWT

def read_data(station, direction):
    """
    Read the GPS data, and divide into segments with only short gaps (less
    than two days), fill the missing values by interpolation

    Input:
        type station = string
        station = Name of the GPS station
        type direction = string
        direction = Component of the displacement (e, n or u)
    Output:
        type times = list of 1D numpy arrays
        times = Time when there is data recorded
        type disps = list of 1D numpy arrays
        disps = Corresponding displacement recorded
        type gaps = list of 1D numpy arrays (integers)
        gaps = Indices where a missing value has been filled by interpolation
    """
    filename = '../data/GeoNet/FITS-' + station + '-' + direction + '.csv'
    # Load the data
    data = pd.read_csv(filename)
    data['date-time'] = pd.to_datetime(data['date-time'])
    time = np.array(data['date-time'].apply( \
        lambda x: datetime.date.toordinal(x)))
    cname = ' ' + direction + ' (mm)'
    disp = np.array(data[cname])
    # Detrend the data
    x = np.reshape(time, (len(time), 1))
    y = np.reshape(disp, (len(disp), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    disp = np.reshape(np.array(y - y_pred), (len(disp)))
    # Look for gaps greater than 2 days
    dt = np.diff(time)
    gap = np.where(dt > 2)[0]
    # Select a subset of the data without big gaps
    ibegin = np.insert(gap, 0, -1)
    iend = np.insert(gap, len(gap), len(time) - 1)
    times = []
    disps = []
    gaps = []
    for i in range(0, len(ibegin)):
        time_sub = time[ibegin[i] + 1 : iend[i] + 1]
        disp_sub = disp[ibegin[i] + 1 : iend[i] + 1]
        # Fill the missing values by interpolation
        dt = np.diff(time_sub)
        gap = np.where(dt > 1)[0]
        for j in range(0, len(gap)):
            time_sub = np.insert(time_sub, gap[j] + 1, \
                time_sub[gap[j]] + 1)
            disp_sub = np.insert(disp_sub, gap[j] + 1, \
                0.5 * (disp_sub[gap[j]] + disp_sub[gap[j] + 1]))
            gap[j : ] = gap[j : ] + 1
        # Add final times, displacements, and gaps to list
        time_sub = np.array([datetime.date.fromordinal(x) for x in time_sub])
        times.append(time_sub)
        disps.append(disp_sub)
        gaps.append(gap)
    return (times, disps, gaps)

def compute_wavelet(times, disps, gaps, J, name, station, direction, \
    draw=True, draw_gaps=False, draw_BC=False):
    """
    Compute the MODWT wavelet coefficients for each times series

    Input:
        type times = list of 1D numpy arrays
        times = Time when there is data recorded
        type disps = list of 1D numpy arrays
        disps = Corresponding displacement recorded
        type gaps = list of 1D numpy arrays (integers)
        gaps = Indices where a missing value has been filled by interpolation
        type J = integer
        J = Level of MODWT
        type name = string
        name = Name of wavelet filter
        type station = string
        station = Name of the GPS station
        type direction = string
        direction = Component of the displacement (e, n, or u)
        type draw = boolean
        draw = Do we draw the wavelet coefficients?
        type draw_gaps = boolean
        draw_gaps = Do we draw a red line where there is a missing value that
            has been filled by interpolation?
        type draw_BC = boolean
        draw_BC = Do we draw blue and green line where the wavelet coefficients
            are affected by the circularity assumption?
    Output:
        type Ws = list of lists of 1D numpy arrays (length J)
        Ws = List of lists of vectors of MODWT wavelet coefficients
        type Vs = list of 1D numpy arrays
        Vs = List of vectors of MODWT scaling coefficients at level J
    """
    # Length of wavelet filter
    g = MODWT.get_scaling(name)
    L = len(g)

    # Draw time series
    if (draw == True):
        plt.figure(1, figsize=(15, 3 * (J + 2)))
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        if (draw_gaps==True):
            for i in range(0, len(gaps)):
                time = times[i]
                gap = gaps[i]
                for j in range(0, len(gap)):
                    plt.axvline(time[gap[j]], linewidth=1, color='red')
        xmin = []
        xmax = []
        for i in range(0, len(times)):
            time = times[i]
            disp = disps[i]
            if (i == 0):
                plt.plot(time, disp, 'k', label='Data')
            else:
                plt.plot(time, disp, 'k')
            xmin.append(np.min(time))
            xmax.append(np.max(time))
        plt.xlim(min(xmin), max(xmax))
        plt.xlabel('Time (years)')
        plt.legend(loc=1)

    # Compute wavelet coefficients
    Ws = []
    Vs = []
    for i in range(0, len(disps)):
        disp = disps[i]
        (W, V) = MODWT.pyramid(disp, name, J)
        if (i == 0):
            (nuH, nuG) = DWT.get_nu(name, J)
        Ws.append(W)
        Vs.append(V)

    # Plot wavelet coefficients at each level
    if (draw == True):
        for j in range(1, J + 1):
            plt.subplot2grid((J + 2, 1), (J + 1 - j, 0))
            for i in range(0, len(times)):
                time = times[i]
                gap = gaps[i]
                W = Ws[i]
                Wj = W[j - 1]
                N = len(time)
                if (draw_gaps == True):
                    for k in range(0, len(gap)):
                        plt.axvline(time[gap[k]], linewidth=1, color='red')
                if (i == 0):
                    plt.plot(time, np.roll(Wj, nuH[j - 1]), 'k', \
                        label = 'W' + str(j))
                else:
                    plt.plot(time, np.roll(Wj, nuH[j - 1]), 'k')
                if (draw_BC == True):
                    Lj = (2 ** j - 1) * (L - 1) + 1
                    if (Lj - 2 - abs(nuH[j - 1]) >= len(time)):
                        plt.axvline(time[-1], linewidth=1, color='blue')
                    else:
                        plt.axvline(time[Lj - 2 - abs(nuH[j - 1])], \
                            linewidth=1, color='blue')
                    if (N - abs(nuH[j - 1]) < 0):
                        plt.axvline(time[0], linewidth=1, color='green')
                    else:
                        plt.axvline(time[N - abs(nuH[j - 1])], linewidth=1, \
                            color='green')
            plt.xlim(min(xmin), max(xmax))
            plt.legend(loc=1)

    # Plot scaling coefficients for the last level
    if (draw == True):
        plt.subplot2grid((J + 2, 1), (0, 0))
        for i in range(0, len(times)):
            time = times[i]
            gap = gaps[i]
            V = Vs[i]
            N = len(time)
            if (draw_gaps == True):
                for k in range(0, len(gap)):
                    plt.axvline(time[gap[k]], linewidth=1, color='red')
            if (i == 0):
                plt.plot(time, np.roll(V, nuG[J - 1]), 'k', \
                    label = 'V' + str(J))
            else:
                plt.plot(time, np.roll(V, nuG[J - 1]), 'k')
            if (draw_BC == True):
                Lj = (2 ** J - 1) * (L - 1) + 1
                if (Lj - 2 - abs(nuG[J - 1]) >= len(time)):
                    plt.axvline(time[-1], linewidth=1, color='blue')
                else:
                    plt.axvline(time[Lj - 2 - abs(nuG[J - 1])], linewidth=1, \
                        color='blue')
                if (N - abs(nuG[J - 1]) < 0):
                    plt.axvline(time[0], linewidth=1, color='green')
                else:
                    plt.axvline(time[N - abs(nuG[J - 1])], linewidth=1, \
                        color='green')
        plt.xlim(min(xmin), max(xmax))
        plt.legend(loc=1)

    # Save figure
    if (draw == True):
        namedir = station
        if not os.path.exists(namedir):
            os.makedirs(namedir)
        title = station + ' - ' + direction
        plt.suptitle(title, fontsize=30)
        plt.savefig(namedir + '/' + direction + '_W.eps', \
            format='eps')
        plt.close(1)

    # Return wavelet coefficients
    return (Ws, Vs)

def compute_details(times, disps, gaps, Ws, J, name, station, direction, \
    draw=True, draw_gaps=False, draw_BC=False):
    """
    Compute the MODWT wavelet details and smooths for each times series

    Input:
        type times = list of 1D numpy arrays
        times = Time when there is data recorded
        type disps = list of 1D numpy arrays
        disps = Corresponding displacement recorded
        type gaps = list of 1D numpy arrays (integers)
        gaps = Indices where a missing value has been filled by interpolation
        type Ws = list of lists of 1D numpy arrays (length J)
        Ws = List of lists of vectors of MODWT wavelet coefficients
        type J = integer
        J = Level of MODWT
        type name = string
        name = Name of wavelet filter
        type station = string
        station = Name of the GPS station
        type direction = string
        direction = Component of the displacement (lat, lon or rad)
        type draw = boolean
        draw = Do we draw the wavelet details and smooths?
        type draw_gaps = boolean
        draw_gaps = Do we draw a red line where there is a missing value that
            has been filled by interpolation?
        type draw_BC = boolean
        draw_BC = Do we draw blue and green line where the wavelet details
            are affected by the circularity assumption?
    Output:
        type Ds = list of lists of 1D numpy arrays (length J)
        Ds = List of lists of details [D1, D2, ... , DJ]
        type Ss = list of lists of 1D numpy arrays (length J+1)
        Ss = List of lists of smooths [S0, S1, S2, ... , SJ]
    """
    # Length of wavelet filter
    g = MODWT.get_scaling(name)
    L = len(g)

    # Draw time series
    if (draw == True):
        plt.figure(1, figsize=(15, 3 * (J + 2)))
        plt.subplot2grid((J + 2, 1), (J + 1, 0))
        if (draw_gaps == True):
            for i in range(0, len(gaps)):
                time = times[i]
                gap = gaps[i]
                for j in range(0, len(gap)):
                    plt.axvline(time[gap[j]], linewidth=1, color='red')
        xmin = []
        xmax = []
        for i in range(0, len(times)):
            time = times[i]
            disp = disps[i]
            if (i == 0):
                plt.plot(time, disp, 'k', label='Data')
            else:
                plt.plot(time, disp, 'k')
            xmin.append(np.min(time))
            xmax.append(np.max(time))
        plt.xlim(min(xmin), max(xmax))
        plt.xlabel('Time (years)')
        plt.legend(loc=1)
        
    # Compute details and smooth
    Ds = []
    Ss = []
    for i in range(0, len(disps)):
        disp = disps[i]
        W = Ws[i]
        (D, S) = MODWT.get_DS(disp, W, name, J)
        Ds.append(D)
        Ss.append(S)

    # Plot details at each level
    if (draw == True):
        for j in range(0, J):
            plt.subplot2grid((J + 2, 1), (J - j, 0))
            for i in range(0, len(times)):
                time = times[i]
                gap = gaps[i]
                D = Ds[i]
                N = len(time)
                if (draw_gaps == True):
                    for k in range(0, len(gap)):
                        plt.axvline(time[gap[k]], linewidth=1, color='red')
                if (i == 0):
                    plt.plot(time, D[j], 'k', label='D' + str(j + 1))
                else:
                    plt.plot(time, D[j], 'k')
                if (draw_BC == True):
                    Lj = (2 ** (j + 1) - 1) * (L - 1) + 1
                    if (Lj - 2 >= len(time)):
                        plt.axvline(time[-1], linewidth=1, color='blue')
                    else:
                        plt.axvline(time[Lj - 2], linewidth=1, color='blue')
                    if (N - Lj + 1 < 0):
                        plt.axvline(time[0], linewidth=1, color='green')
                    else:
                        plt.axvline(time[N - Lj + 1], linewidth=1, \
                            color='green')
            plt.xlim(min(xmin), max(xmax))
            plt.legend(loc=1)

    # Plot smooth for the last level
    if (draw == True):
        plt.subplot2grid((J + 2, 1), (0, 0))
        for i in range(0, len(times)):
            time = times[i]
            gap = gaps[i]
            S = Ss[i]
            N = len(time)
            if (draw_gaps == True):
                for k in range(0, len(gap)):
                    plt.axvline(time[gap[k]], linewidth=1, color='red')
            if (i == 0):
                plt.plot(time, S[J], 'k', label='S' + str(J))
            else:
                plt.plot(time, S[J], 'k')
            if (draw_BC == True):
                Lj = (2 ** J - 1) * (L - 1) + 1
                if (Lj - 2 >= len(time)):
                    plt.axvline(time[-1], linewidth=1, color='blue')
                else:
                    plt.axvline(time[Lj - 2], linewidth=1, color='blue')
                if (N - Lj + 1 < 0):
                    plt.axvline(time[0], linewidth=1, color='green')
                else:
                    plt.axvline(time[N - Lj + 1], linewidth=1, color='green')
        plt.xlim(min(xmin), max(xmax))
        plt.legend(loc=1)
        
    # Save figure
    if (draw == True):
        namedir = station
        if not os.path.exists(namedir):
            os.makedirs(namedir)
        title = station + ' - ' + direction
        plt.suptitle(title, fontsize=30)
        plt.savefig(namedir + '/' + direction + '_DS.eps', \
            format='eps')
        plt.close(1)

    # Return details and smooths
    return (Ds, Ss)
