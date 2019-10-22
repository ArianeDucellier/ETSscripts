"""
This module contains a function to plot the stack of the cross correlations
computed with stack_ccorr_tremor
"""

import obspy

import matplotlib.pyplot as plt
import numpy as np
import pickle

from matplotlib.patches import Rectangle

from stacking import linstack, powstack, PWstack

def plot_stack_ccorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
    amp_pow, amp_PWS, n1, n2):
    """
    This function stacks the cross correlation over all the tremor windows
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
        Tmax = Maximum time lag for cross correlation plot
        type amp = float
        amp = Amplification factor of cross correlation for plotting
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
    # Read file containing data from stack_ccorr_tremor
    filename = 'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}.pkl'.format( \
        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), \
        type_stack)
    data = pickle.load(open(filename, 'rb'))
    EW_UD = data[6]
    NS_UD = data[7]
    # Stack over all tremor windows
    EW_lin = linstack([EW_UD], normalize=False)[0]
    EW_pow = powstack([EW_UD], w, normalize=False)[0]
    EW_PWS = PWstack([EW_UD], w, normalize=False)[0]
    NS_lin = linstack([NS_UD], normalize=False)[0]
    NS_pow = powstack([NS_UD], w, normalize=False)[0]
    NS_PWS = PWstack([NS_UD], w, normalize=False)[0]   
    # Plot
    plt.figure(1, figsize=(20, 15))
    # EW - UD cross correlation
    ax1 = plt.subplot(121)
    for i in range(n1, n2):
        dt = EW_UD[i].stats.delta
        ncor = int((EW_UD[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        if (i == 36):
            plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW_UD[i].data, 'r-')
        elif (i == 51):
            plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW_UD[i].data, 'b-')
        elif (i == 19):
            plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW_UD[i].data, 'g-')
        else:
            plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW_UD[i].data, 'k-')
    wave = Rectangle((4.0, 5), 1.4, 86, facecolor = 'c', fill=True, alpha=0.5)
    ax1.add_patch(wave)
    plt.plot(t, - 2.0 + amp_lin * EW_lin.data, 'r-')
    plt.plot(t, - 2.0 + amp_pow * EW_pow.data, 'b-')
    plt.plot(t, - 2.0 + amp_PWS * EW_PWS.data, 'g-')
    plt.xlim(0, Tmax)
    plt.ylim(- 5.0, 2.0 * (n2 - n1))
    plt.title('East / Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Cross correlation', fontsize=24)
    ax1.set_yticklabels([])
    ax1.tick_params(labelsize=20)
    # NS - UD cross correlation
    ax2 = plt.subplot(122)
    for i in range(n1, n2):
        dt = NS_UD[i].stats.delta
        ncor = int((NS_UD[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        if (i == 36):
            plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS_UD[i].data, 'r-')
        elif (i == 51):
            plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS_UD[i].data, 'b-')
        elif (i == 19):
            plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS_UD[i].data, 'g-')
        else:
            plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS_UD[i].data, 'k-')
    wave = Rectangle((4.0, 5), 1.4, 86, facecolor = 'c', fill=True, alpha=0.5)
    ax2.add_patch(wave)
    plt.plot(t, - 2.0 + amp_lin * NS_lin.data, 'r-')
    plt.plot(t, - 2.0 + amp_pow * NS_pow.data, 'b-')
    plt.plot(t, - 2.0 + amp_PWS * NS_PWS.data, 'g-')
    plt.xlim(0, Tmax)
    plt.ylim(- 5.0, 2.0 * (n2 - n1))
    plt.title('North / Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Cross correlation', fontsize=24)
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=20)
    # End figure and plot
    plt.suptitle('{} at {} km, {} km'.format(arrayName, x0, y0), fontsize=24)
    plt.savefig('cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}.eps'.format( \
        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), \
        type_stack), format='eps')
    ax1.clear()
    ax2.clear()
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
    plot_stack_ccorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
    amp_pow, amp_PWS, n1, n2)

    # Power stack
#    type_stack = 'pow'
#    amp = 2.0
#    amp_lin = 15.0
#    amp_pow = 1.0
#    amp_PWS = 100.0
#    plot_stack_ccorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
#    amp_pow, amp_PWS, n1, n2)

    # Phase-weighted stack
#    type_stack = 'PWS'
#    amp = 20.0
#    amp_lin = 200.0
#    amp_pow = 10.0
#    amp_PWS = 1000.0
#    plot_stack_ccorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
#    amp_pow, amp_PWS, n1, n2)
