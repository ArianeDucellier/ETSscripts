"""
This module contains a function to plot the stack of the autocorrelations
computed with stack_acorr_tremor
"""

import obspy

import matplotlib.pyplot as plt
import numpy as np
import pickle

from stacking import linstack, powstack, PWstack

def plot_stack_acorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
    amp_pow, amp_PWS, n1, n2):
    """
    This function stacks the autocorrelation over all the tremor windows
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
        type Tmax = float
        Tmax = Maximum time lag for autocorrelation plot
        type amp = float
        amp = Amplification factor of autocorrelation for plotting
        type amp_lin = float
        amp_lin = Amplification factor of linear stack for plotting
        type amp_pow = float
        amp_pow = Amplification factor of power stack for plotting
        type amp_PWS = float
        amp_PWS = Amplification factor of phase-weighted stack for plotting
        type n1 = integer
        n1 = Index of first tremor to be plotted
        type n2 = integer
        n2 = Index of last tremor to be plotted
    """
    # Read file containing data from stack_acorr_tremor
    filename = 'ac/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}.pkl'.format( \
        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), \
        type_stack)
    data = pickle.load(open(filename, 'rb'))
    EW = data[6]
    NS = data[7]
    UD = data[8]
    # Stack over all tremor windows
    EW_lin = linstack([EW], normalize=False)[0]
    EW_pow = powstack([EW], w, normalize=False)[0]
    EW_PWS = PWstack([EW], w, normalize=False)[0]
    NS_lin = linstack([NS], normalize=False)[0]
    NS_pow = powstack([NS], w, normalize=False)[0]
    NS_PWS = PWstack([NS], w, normalize=False)[0]
    UD_lin = linstack([UD], normalize=False)[0]
    UD_pow = powstack([UD], w, normalize=False)[0]
    UD_PWS = PWstack([UD], w, normalize=False)[0] 
    # Plot
    plt.figure(1, figsize=(30, 15))
    # EW autocorrelation
    ax1 = plt.subplot(131)
    for i in range(n1, n2):
        dt = EW[i].stats.delta
        ncor = int((EW[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW[i].data, 'k-')
    plt.plot(t, - 2.0 + amp_lin * EW_lin.data, 'r-')
    plt.plot(t, - 2.0 + amp_pow * EW_pow.data, 'b-')
    plt.plot(t, - 2.0 + amp_PWS * EW_PWS.data, 'g-')
    plt.xlim(0, Tmax)
    plt.ylim(- 5.0, 2.0 * (n2 - n1))
    plt.title('East component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax1.set_yticklabels([])
    ax1.tick_params(labelsize=20)
    # NS autocorrelation
    ax2 = plt.subplot(132)
    for i in range(n1, n2):
        dt = NS[i].stats.delta
        ncor = int((NS[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS[i].data, 'k-')
    plt.plot(t, - 2.0 + amp_lin * NS_lin.data, 'r-')
    plt.plot(t, - 2.0 + amp_pow * NS_pow.data, 'b-')
    plt.plot(t, - 2.0 + amp_PWS * NS_PWS.data, 'g-')
    plt.xlim(0, Tmax)
    plt.ylim(- 5.0, 2.0 * (n2 - n1))
    plt.title('North component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=20)
    # UD autocorrelation
    ax3 = plt.subplot(133)
    for i in range(n1, n2):
        dt = UD[i].stats.delta
        ncor = int((UD[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * UD[i].data, 'k-')
    plt.plot(t, - 2.0 + amp_lin * UD_lin.data, 'r-')
    plt.plot(t, - 2.0 + amp_pow * UD_pow.data, 'b-')
    plt.plot(t, - 2.0 + amp_PWS * UD_PWS.data, 'g-')
    plt.xlim(0, Tmax)
    plt.ylim(- 5.0, 2.0 * (n2 - n1))
    plt.title('Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax3.set_yticklabels([])
    ax3.tick_params(labelsize=20)
    # End figure and plot
    plt.suptitle('{} at {} km, {} km'.format(arrayName, x0, y0), fontsize=24)
    plt.savefig('ac/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}.eps'.format( \
        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), \
        type_stack), format='eps')
    ax1.clear()
    ax2.clear()
    ax3.clear()
    plt.close(1)

if __name__ == '__main__':

    # Set the parameters
    arrayName = 'BS'
    x0 = 0.0
    y0 = 0.0
    w = 2.0
    Tmax = 15.0
    n1 = 0
    n2 = 82

    # Linear stack
    type_stack = 'lin'
    amp = 10.0
    amp_lin = 100.0
    amp_pow = 2.0
    amp_PWS = 200.0
    plot_stack_acorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
    amp_pow, amp_PWS, n1, n2)

    # Power stack
    type_stack = 'pow'
    amp = 2.0
    amp_lin = 15.0
    amp_pow = 1.0
    amp_PWS = 100.0
    plot_stack_acorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
    amp_pow, amp_PWS, n1, n2)

    # Phase-weighted stack
    type_stack = 'PWS'
    amp = 20.0
    amp_lin = 200.0
    amp_pow = 10.0
    amp_PWS = 1000.0
    plot_stack_acorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
    amp_pow, amp_PWS, n1, n2)
