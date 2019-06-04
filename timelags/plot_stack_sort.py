"""
This module contains a function to plot the cross correlations computed with
stack_ccorr_tremor and the autocorrelations computed with
stack_acorr_tremor sorted by different criteria
"""

import obspy
from obspy.signal.cross_correlation import correlate

import matplotlib.pyplot as plt
import numpy as np
import pickle

from stacking import linstack, powstack, PWstack

def plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, type_sort, \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax):
    """
    This function plots the cross correlations computed with
    stack_ccorr_tremor and the autocorrelations computed with
    stack_acorr_tremor sorted by different criteria

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
        type Tmax = float
        Tmax = Maximum time lag for cross correlation plot
        type amp = float
        amp = Amplification factor of cross correlation for plotting
        type n1 = integer
        n1 = Index of first tremor to be plotted
        type n2 = integer
        n2 = Index of last tremor to be plotted
        type ncor = integer
        ncor = Number of points for the cross correlation with the stack
        type tmin = float
        tmin = Minimum time lag for comparing cross correlation with the stack
        type tmax = float
        tmax = Maximum time lag for comparing cross correlation with the stack
        type RMSmin = float
        RMSmin = Minimum time lag to compute the RMS
        type RMSmax = float
        RMSmax = Maximum time lag to compute the RMS
    """
    # Read file containing data from stack_ccorr_tremor
    filename = 'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}.pkl'.format( \
        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), \
        type_stack)
    data = pickle.load(open(filename, 'rb'))
    EW_UD = data[6]
    NS_UD = data[7]
    # Read file containing data from stack_acorr_tremor
    filename = 'ac/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}.pkl'.format( \
        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), \
        type_stack)
    data = pickle.load(open(filename, 'rb'))
    EW = data[6]
    NS = data[7]
    UD = data[8]
    # Stack over all tremor windows
    if (cc_stack == 'lin'):
        EWstack = linstack([EW_UD], normalize=False)[0]
        NSstack = linstack([NS_UD], normalize=False)[0]
    elif (cc_stack == 'pow'):
        EWstack = powstack([EW_UD], w, normalize=False)[0]
        NSstack = powstack([NS_UD], w, normalize=False)[0]
    elif (cc_stack == 'PWS'):
        EWstack = PWstack([EW_UD], w, normalize=False)[0]
        NSstack = PWstack([NS_UD], w, normalize=False)[0]
    else:
        raise ValueError( \
            'Type of stack must be lin, pow, or PWS')
    # Initialize indicators of cross correlation fit
    nt = len(EW_UD)
    ccmaxEW = np.zeros(nt)
    cc0EW = np.zeros(nt)
    timedelayEW = np.zeros(nt)
    rmsEW = np.zeros(nt)
    ccmaxNS = np.zeros(nt)
    cc0NS = np.zeros(nt)
    timedelayNS = np.zeros(nt)
    rmsNS = np.zeros(nt)
    # Windows of the cross correlation to look at
    i0 = int((len(EWstack) - 1) / 2)
    ibegin = i0 + int(tmin / EWstack.stats.delta)
    iend = i0 + int(tmax / EWstack.stats.delta) + 1
    rmsb = i0 + int(RMSmin / EWstack.stats.delta)
    rmse = i0 + int(RMSmax / EWstack.stats.delta) + 1
    for i in range(0, nt):
        rmsEW[i] = np.max(np.abs(EW_UD[i][ibegin : iend])) / \
            np.sqrt(np.mean(np.square(EW_UD[i][rmsb:rmse])))       
        rmsNS[i] = np.max(np.abs(NS_UD[i][ibegin : iend])) / \
            np.sqrt(np.mean(np.square(NS_UD[i][rmsb:rmse])))
        # Cross correlate cc for EW with stack       
        cc_EW = correlate(EW_UD[i][ibegin : iend], EWstack[ibegin : iend], \
            ncor)
        ccmaxEW[i] = np.max(cc_EW)
        cc0EW[i] = cc_EW[ncor]
        timedelayEW[i] = (np.argmax(cc_EW) - ncor) * EWstack.stats.delta
        # Cross correlate cc for NS with stack
        cc_NS = correlate(NS_UD[i][ibegin : iend], NSstack[ibegin : iend], \
            ncor)
        ccmaxNS[i] = np.max(cc_NS)
        cc0NS[i] = cc_NS[ncor]
        timedelayNS[i] = (np.argmax(cc_NS) - ncor) * NSstack.stats.delta
    # Sort cross correlations
    if (type_sort == 'ccmaxEW'):
        order = np.argsort(ccmaxEW)
    elif (type_sort == 'ccmaxNS'):
        order = np.argsort(ccmaxNS)
    elif (type_sort == 'cc0EW'):
        order = np.argsort(cc0EW)
    elif (type_sort == 'cc0NS'):
        order = np.argsort(cc0NS)
    elif (type_sort == 'timedelayEW'):
        order = np.flip(np.argsort(np.abs(timedelayEW)), axis=0)
    elif (type_sort == 'timedelayNS'):
        order = np.flip(np.argsort(np.abs(timedelayNS)), axis=0)
    elif (type_sort == 'rmsEW'):
        order = np.argsort(rmsEW)
    elif (type_sort == 'rmsNS'):
        order = np.argsort(rmsNS)
    else:
        raise ValueError( \
            'Type of ranking must be ccmaxEW, ccmaxNS, cc0EW, cc0NS, ' + \
            'timedelayEW, timedelayNS, rmsEW or rmsNS')
    # Plot cross correlations
    plt.figure(1, figsize=(20, 15))  
    ax1 = plt.subplot(121)
    for i in range(n1, n2):
        index = order[nt - i - 1]
        dt = EW_UD[index].stats.delta
        ncor = int((EW_UD[index].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW_UD[index].data, 'k-')
    plt.xlim(0, Tmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('East / Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Cross correlation', fontsize=24)
    ax1.set_yticklabels([])
    ax1.tick_params(labelsize=20)
    ax2 = plt.subplot(122)
    for i in range(n1, n2):
        index = order[nt - i - 1]
        dt = NS_UD[index].stats.delta
        ncor = int((NS_UD[index].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS_UD[index].data, 'k-')
    plt.xlim(0, Tmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('North / Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Cross correlation', fontsize=24)
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=20)
    plt.suptitle('{} at {} km, {} km ({} - {}) sorted by {}'.format( \
        arrayName, x0, y0, type_stack, cc_stack, type_sort), fontsize=24)
    plt.savefig('cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_sort_{}.eps'. \
        format(arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), \
        int(y0), type_stack, cc_stack, type_sort), format='eps')
    ax1.clear()
    ax2.clear()
    plt.close(1)
    # Plot autocorrelations
    plt.figure(2, figsize=(30, 15))  
    ax1 = plt.subplot(131)
    for i in range(n1, n2):
        index = order[nt - i - 1]
        dt = EW[index].stats.delta
        ncor = int((EW[index].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW[index].data, 'k-')
    plt.xlim(0, Tmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('East component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax1.set_yticklabels([])
    ax1.tick_params(labelsize=20)
    ax2 = plt.subplot(132)
    for i in range(n1, n2):
        index = order[nt - i - 1]
        dt = NS[index].stats.delta
        ncor = int((NS[index].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS[index].data, 'k-')
    plt.xlim(0, Tmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('North component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=20)
    ax3 = plt.subplot(133)
    for i in range(n1, n2):
        index = order[nt - i - 1]
        dt = UD[index].stats.delta
        ncor = int((UD[index].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * UD[index].data, 'k-')
    plt.xlim(0, Tmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax3.set_yticklabels([])
    ax3.tick_params(labelsize=20)
    plt.suptitle('{} at {} km, {} km ({} - {}) sorted by {}'.format( \
        arrayName, x0, y0, type_stack, cc_stack, type_sort), fontsize=24)
    plt.savefig('ac/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_sort_{}.eps'. \
        format(arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), \
        int(y0), type_stack, cc_stack, type_sort), format='eps')
    ax1.clear()
    ax2.clear()
    ax3.clear()
    plt.close(2)

if __name__ == '__main__':

    # Set the parameters
    arrayName = 'BS'
    x0 = 0.0
    y0 = 0.0
    w = 2.0
    Tmax = 15.0
    n1 = 0
    n2 = 82
    ncor = 40
    tmin = 4.0
    tmax = 6.0
    RMSmin = 12.0
    RMSmax = 14.0

    # Linear stack - Linear stack
    type_stack = 'lin'
    cc_stack = 'lin'
    amp = 10.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)

    # Linear stack - Power stack
    type_stack = 'lin'
    cc_stack = 'pow'
    amp = 10.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)

    # Linear stack - PWS stack
    type_stack = 'lin'
    cc_stack = 'PWS'
    amp = 10.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)

    # Power stack - Linear stack
    type_stack = 'pow'
    cc_stack = 'lin'
    amp = 2.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)

    # Power stack - Power stack
    type_stack = 'pow'
    cc_stack = 'pow'
    amp = 2.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)

    # Power stack - PWS stack
    type_stack = 'pow'
    cc_stack = 'PWS'
    amp = 2.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)

    # PWS stack - Linear stack
    type_stack = 'PWS'
    cc_stack = 'lin'
    amp = 50.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)

    # PWS stack - Power stack
    type_stack = 'PWS'
    cc_stack = 'pow'
    amp = 50.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)

    # PWS stack - PWS stack
    type_stack = 'PWS'
    cc_stack = 'PWS'
    amp = 50.0
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'ccmaxNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0EW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'cc0NS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayEW', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, \
        'timedelayNS', Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsEW', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
    plot_stack_sort(arrayName, x0, y0, type_stack, w, cc_stack, 'rmsNS', \
        Tmax, amp, n1, n2, ncor, tmin, tmax, RMSmin, RMSmax)
