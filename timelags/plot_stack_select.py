"""
This module contains a function to plot the stack of the cross correlations
computed with stack_ccorr_tremor using only selected tremor windows
"""

import obspy
from obspy.core.stream import Stream
from obspy.signal.cross_correlation import correlate

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
        type Emax = float
        Emax = Minimum value of max cross correlation (east)
        type Nmax = float
        Nmax = Minimum value of max cross correlation (north)
        type E0 = float
        E0 = Minimum value of cross correlation at zeros time lag (east)
        type N0 = float
        N0 = Minimum value of cross correlation at zeros time lag (north)
        type Et = float
        Et = Minimum value of time delay (east)
        type Nt = float
        Nt = Minimum value of time delay (north)
    """
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
    # Initialize streams of selected traces
    EW1 = Stream()
    EW2 = Stream()
    EW3 = Stream()
    NS1 = Stream()
    NS2 = Stream()
    NS3 = Stream()
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
        # Keep only selected traces
        if ((ccmaxEW[i] > Emax) and (ccmaxNS[i] > Nmax)):
            EW1.append(EW_UD[i])
            NS1.append(NS_UD[i])
        if ((cc0EW[i] > E0) and (cc0NS[i] > N0)):
            EW2.append(EW_UD[i])
            NS2.append(NS_UD[i])
        if ((abs(timedelayEW[i]) <= Et) and (abs(timedelayNS[i]) <= Nt)):
            EW3.append(EW_UD[i])
            NS3.append(NS_UD[i])
    # Stack over selected tremor windows
    if (cc_stack == 'lin'):
        EW1stack = linstack([EW1], normalize=False)[0]
        EW2stack = linstack([EW2], normalize=False)[0]
        EW3stack = linstack([EW3], normalize=False)[0]
        NS1stack = linstack([NS1], normalize=False)[0]
        NS2stack = linstack([NS2], normalize=False)[0]
        NS3stack = linstack([NS3], normalize=False)[0]
    elif (cc_stack == 'pow'):
        EW1stack = powstack([EW1], w, normalize=False)[0]
        EW2stack = powstack([EW2], w, normalize=False)[0]
        EW3stack = powstack([EW3], w, normalize=False)[0]
        NS1stack = powstack([NS1], w, normalize=False)[0]
        NS2stack = powstack([NS2], w, normalize=False)[0]
        NS3stack = powstack([NS3], w, normalize=False)[0]
    elif (cc_stack == 'PWS'):
        EW1stack = PWstack([EW1], w, normalize=False)[0]
        EW2stack = PWstack([EW2], w, normalize=False)[0]
        EW3stack = PWstack([EW3], w, normalize=False)[0]
        NS1stack = PWstack([NS1], w, normalize=False)[0]
        NS2stack = PWstack([NS2], w, normalize=False)[0]
        NS3stack = PWstack([NS3], w, normalize=False)[0]
    else:
        raise ValueError( \
            'Type of stack must be lin, pow, or PWS')
    # Plot
    plt.figure(1, figsize=(45, 20))
    npts = int((EW.stats.npts - 1) / 2)
    dt = EW.stats.delta
    t = dt * np.arange(- npts, npts + 1)
    # Select with max cross correlation
    ax1 = plt.subplot(231)
    plt.plot(t, EW.data, 'k-', label='All')
    plt.plot(t, EW1stack.data, 'r-', label='Selected')
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Max cross correlation (east)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    ax2 = plt.subplot(234)
    plt.plot(t, NS.data, 'k-', label='All')
    plt.plot(t, NS1stack.data, 'r-', label='Selected')
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Max cross correlation (north)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # Select with cross correlation at zero time lag
    ax3 = plt.subplot(232)
    plt.plot(t, EW.data, 'k-', label='All')
    plt.plot(t, EW2stack.data, 'r-', label='Selected')
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Cross correlation at 0 (east)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    ax4 = plt.subplot(235)
    plt.plot(t, NS.data, 'k-', label='All')
    plt.plot(t, NS2stack.data, 'r-', label='Selected')
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Cross correlation at 0 (north)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # Select with time delay
    ax5 = plt.subplot(233)
    plt.plot(t, EW.data, 'k-', label='All')
    plt.plot(t, EW3stack.data, 'r-', label='Selected')
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Time delay (east)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    ax6 = plt.subplot(236)
    plt.plot(t, NS.data, 'k-', label='All')
    plt.plot(t, NS3stack.data, 'r-', label='Selected')
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Time delay (north)', fontsize=24)
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
    Emax = 0.5
    Nmax = 0.5
    E0 = 0.4
    N0 = 0.4
    Et = 0.1
    Nt = 0.1

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
