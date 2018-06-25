"""
This module contains a function to plot the stack of the cross correlations
computed with stack_ccorr_tremor using only selected tremor windows
"""

import obspy
from obspy.core.stream import Stream
from obspy.signal.cross_correlation import correlate

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle

from stacking import linstack, powstack, PWstack

def plot_stack(arrayName, x0, y0, type_stack, w, cc_stack, ncor, Tmin, Tmax, \
        xmax, ymax, Emax, Nmax, E0, N0, Et, Nt):
    """
    This function stacks the cross correlation over selected tremor windows
    and plot the stack

    Input:
        type arrayName = string
        arrayName = Name of seismic array
        type x0 = float
        x0 = Distance of the center of the cell from the array (east)
        type y0 = float
        y0 = Distance of the center of the cell from the array (north)
        type type_stack = string
        type_stack = Type of stack ('lin', 'pow', 'PWS')
        type w = float
        w = Power of the stack (for 'pow' and 'PWS')
        type cc_stack = string
        cc_stack = Type of stack ('lin', 'pow', 'PWS') over tremor windows
        type ncor = integer
        ncor = Number of points for the cross correlation with the stack
        type Tmin = float
        Tmin = Minimum time lag for comparing cross correlation with the stack
        type Tmax = float
        Tmax = Maximum time lag for comparing cross correlation with the stack
        type xmax = float
        xmax = Horizontal axis limit for plot
        type ymax = float
        ymax = Vertical axis limit for plot
        type Emax = list of floats
        Emax = Minimum values of max cross correlation (east)
        type Nmax = list of floats
        Nmax = Minimum values of max cross correlation (north)
        type E0 = list of floats
        E0 = Minimum values of cross correlation at zeros time lag (east)
        type N0 = list of floats
        N0 = Minimum values of cross correlation at zeros time lag (north)
        type Et = list of floats
        Et = Minimum values of time delay (east)
        type Nt = list of floats
        Nt = Minimum values of time delay (north)
    """
    assert (len(Emax) == len(Nmax)), \
        'Emax and Nmax must have the same length'
    assert (len(E0) == len(N0)), \
        'E0 and N0 must have the same length'
    assert (len(Et) == len(Nt)), \
        'Et and Nt must have the same length'
    # Read file containing data from stack_ccorr_tremor
    filename = 'cc/{}_{:03d}_{:03d}_{}.pkl'.format(arrayName, int(x0), \
        int(y0), type_stack)
    data = pickle.load(open(filename, 'rb'))
    EW_UD = data[6]
    NS_UD = data[7]
    # Stack over all tremor windows
    if (cc_stack == 'lin'):
        EW = linstack([EW_UD], normalize=False)[0]
        NS = linstack([NS_UD], normalize=False)[0]
    elif (cc_stack == 'pow'):
        EW = powstack([EW_UD], w, normalize=False)[0]
        NS = powstack([NS_UD], w, normalize=False)[0]
    elif (cc_stack == 'PWS'):
        EW = PWstack([EW_UD], w, normalize=False)[0]
        NS = PWstack([NS_UD], w, normalize=False)[0]
    else:
        raise ValueError( \
            'Type of stack must be lin, pow, or PWS')
    # Initialize indicators of cross correlation fit
    nt = len(EW_UD)
    ccmaxEW = np.zeros(nt)
    cc0EW = np.zeros(nt)
    timedelayEW = np.zeros(nt)
    ccmaxNS = np.zeros(nt)
    cc0NS = np.zeros(nt)
    timedelayNS = np.zeros(nt)
    # Windows of the cross correlation to look at
    i0 = int((len(EW) - 1) / 2)
    ibegin = i0 + int(Tmin / EW.stats.delta)
    iend = i0 + int(Tmax / EW.stats.delta) + 1
    for i in range(0, nt):
        # Cross correlate cc for EW with stack       
        cc_EW = correlate(EW_UD[i][ibegin : iend], EW[ibegin : iend], ncor)
        ccmaxEW[i] = np.max(cc_EW)
        cc0EW[i] = cc_EW[ncor]
        timedelayEW[i] = (np.argmax(cc_EW) - ncor) * EW.stats.delta
        # Cross correlate cc for NS with stack
        cc_NS = correlate(NS_UD[i][ibegin : iend], NS[ibegin : iend], ncor)
        ccmaxNS[i] = np.max(cc_NS)
        cc0NS[i] = cc_NS[ncor]
        timedelayNS[i] = (np.argmax(cc_NS) - ncor) * NS.stats.delta
    # Plot
    plt.figure(1, figsize=(45, 20))
    npts = int((EW.stats.npts - 1) / 2)
    dt = EW.stats.delta
    t = dt * np.arange(- npts, npts + 1)
    # EW / Vertical. Select with max cross correlation
    ax1 = plt.subplot(231)
    plt.plot(t, EW.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Emax)))
    for j in range(0, len(Emax)):
        EWselect = Stream()
        for i in range(0, nt):
            if ((ccmaxEW[i] >= Emax[j]) and (ccmaxNS[i] >= Nmax[j])):
                EWselect.append(EW_UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            EWstack = linstack([EWselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            EWstack = powstack([EWselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            EWstack = PWstack([EWselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, EWstack.data, color = colors[j], \
            label='Emax = {:3.2f}, Nmax = {:3.2f}'.format(Emax[j], Nmax[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('EW / Vertical (Max cross correlation)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # NS/ Vertical. Select with max cross correlation
    ax2 = plt.subplot(234)
    plt.plot(t, NS.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Nmax)))
    for j in range(0, len(Nmax)):
        NSselect = Stream()
        for i in range(0, nt):
            if ((ccmaxEW[i] >= Emax[j]) and (ccmaxNS[i] >= Nmax[j])):
                NSselect.append(NS_UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            NSstack = linstack([NSselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            NSstack = powstack([NSselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            NSstack = PWstack([NSselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, NSstack.data, color = colors[j], \
            label='Emax = {:3.2f}, Nmax = {:3.2f}'.format(Emax[j], Nmax[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('NS / Vertical (Max cross correlation)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # EW / Vertical. Select with cross correlation at zero time lag
    ax3 = plt.subplot(232)
    plt.plot(t, EW.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(E0)))
    for j in range(0, len(E0)):
        EWselect = Stream()
        for i in range(0, nt):
            if ((cc0EW[i] >= E0[j]) and (cc0NS[i] >= N0[j])):
                EWselect.append(EW_UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            EWstack = linstack([EWselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            EWstack = powstack([EWselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            EWstack = PWstack([EWselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, EWstack.data, color = colors[j], \
            label='E0 = {:3.2f}, N0 = {:3.2f}'.format(E0[j], N0[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('EW / Vertical (Cross correlation at 0)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # NS / Vertical. Select with cross correlation at zero time lag
    ax4 = plt.subplot(235)
    plt.plot(t, NS.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(N0)))
    for j in range(0, len(N0)):
        NSselect = Stream()
        for i in range(0, nt):
            if ((cc0EW[i] >= E0[j]) and (cc0NS[i] >= N0[j])):
                NSselect.append(NS_UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            NSstack = linstack([NSselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            NSstack = powstack([NSselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            NSstack = PWstack([NSselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, NSstack.data, color = colors[j], \
            label='E0 = {:3.2f}, N0 = {:3.2f}'.format(E0[j], N0[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('NS / Vertical (Cross correlation at 0)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # EW / Vertical. Select with time delay
    ax5 = plt.subplot(233)
    plt.plot(t, EW.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Et)))
    for j in range(0, len(Et)):
        EWselect = Stream()
        for i in range(0, nt):
            if ((timedelayEW[i] <= Et[j]) and (timedelayNS[i] <= Nt[j])):
                EWselect.append(EW_UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            EWstack = linstack([EWselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            EWstack = powstack([EWselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            EWstack = PWstack([EWselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, EWstack.data, color = colors[j], \
            label='Et = {:3.2f}, Nt = {:3.2f}'.format(Et[j], Nt[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('EW / Vertical (Time delay)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # NS / Vertical. Select with time delay
    ax6 = plt.subplot(236)
    plt.plot(t, NS.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Nt)))
    for j in range(0, len(Nt)):
        NSselect = Stream()
        for i in range(0, nt):
            if ((timedelayEW[i] <= Et[j]) and (timedelayNS[i] <= Nt[j])):
                NSselect.append(NS_UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            NSstack = linstack([NSselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            NSstack = powstack([NSselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            NSstack = PWstack([NSselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, NSstack.data, color = colors[j], \
            label='Et = {:3.2f}, Nt = {:3.2f}'.format(Et[j], Nt[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('NS / Vertical (Time delay)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
   # End figure
    plt.suptitle('{} at {} km, {} km ({} - {})'.format(arrayName, x0, y0, \
        type_stack, cc_stack), fontsize=24)
    plt.savefig('cc/{}_{:03d}_{:03d}_{}_{}_select.eps'.format(arrayName, \
        int(x0), int(y0), type_stack, cc_stack), format='eps')
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    plt.close(1)

if __name__ == '__main__':

    # Set the parameters
    arrayName = 'BS'
    x0 = 0.0
    y0 = 0.0
    w = 2.0
    ncor = 120
    Tmin = 2.0
    Tmax = 8.0
    xmax = 15.0
    Emax = [0.3, 0.4, 0.5, 0.6]
    Nmax = [0.3, 0.4, 0.5, 0.6]
    E0 = [0.2, 0.3, 0.4, 0.5]
    N0 = [0.2, 0.3, 0.4, 0.5]
    Et = [0.0, 0.05, 0.1, 0.15, 0.2]
    Nt = [0.0, 0.05, 0.1, 0.15, 0.2]

    plot_stack(arrayName, x0, y0, 'lin', w, 'lin', ncor, Tmin, Tmax, \
        xmax, 0.1, Emax, Nmax, E0, N0, Et, Nt)
    plot_stack(arrayName, x0, y0, 'lin', w, 'pow', ncor, Tmin, Tmax, \
        xmax, 0.2, Emax, Nmax, E0, N0, Et, Nt)
    plot_stack(arrayName, x0, y0, 'lin', w, 'PWS', ncor, Tmin, Tmax, \
        xmax, 0.05, Emax, Nmax, E0, N0, Et, Nt)
    plot_stack(arrayName, x0, y0, 'pow', w, 'lin', ncor, Tmin, Tmax, \
        xmax, 0.2, Emax, Nmax, E0, N0, Et, Nt)
    plot_stack(arrayName, x0, y0, 'pow', w, 'pow', ncor, Tmin, Tmax, \
        xmax, 1.0, Emax, Nmax, E0, N0, Et, Nt)
    plot_stack(arrayName, x0, y0, 'pow', w, 'PWS', ncor, Tmin, Tmax, \
        xmax, 0.15, Emax, Nmax, E0, N0, Et, Nt)
    plot_stack(arrayName, x0, y0, 'PWS', w, 'lin', ncor, Tmin, Tmax, \
        xmax, 0.02, Emax, Nmax, E0, N0, Et, Nt)
    plot_stack(arrayName, x0, y0, 'PWS', w, 'pow', ncor, Tmin, Tmax, \
        xmax, 0.2, Emax, Nmax, E0, N0, Et, Nt)
    plot_stack(arrayName, x0, y0, 'PWS', w, 'PWS', ncor, Tmin, Tmax, \
        xmax, 0.01, Emax, Nmax, E0, N0, Et, Nt)
