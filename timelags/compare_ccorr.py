"""
This module contains a function to compare each individual cross correlation
with the stack and see whether there is some link with the depth or the
uncertainties on the location of the tremor source
"""

import obspy
from obspy.signal.cross_correlation import correlate

import matplotlib.pyplot as plt
import numpy as np
import pickle

from scipy.io import loadmat

from date import ymdhms2matlab
from stacking import linstack, powstack, PWstack

def compare_ccorr_location(arrayName, x0, y0, type_stack, w, cc_stack, ncor, \
        Tmin, Tmax):
    """
    This function compare each individual cross correlation  with the stack
    and compare the result with the depth and uncertainty on the location of
    the tremor source

    Input:
        type arrayName = string
        arrayName = Name of seismic array
        type x0 = float
        x0 = Distance of the center of the cell from the array (east)
        type y0 = float
        y0 = Distance of the center of the cell from the array (north)
        type type_stack = string
        type_stack = Type of stack ('lin', 'pow', 'PWS') over stations
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
    """
    # Read file containing data from stack_ccorr_tremor
    filename = 'cc/{}_{:03d}_{:03d}_{}.pkl'.format(arrayName, int(x0), \
        int(y0), type_stack)
    data = pickle.load(open(filename, 'rb'))
    Year = data[0]
    Month = data[1]
    Day = data[2]
    Hour = data[3]
    Minute = data[4]
    Second = data[5]
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
    # Open file containing tremor locations
    data = loadmat('../data/timelags/mbbp_cat_d_forHeidi')
    mbbp_cat_d = data['mbbp_cat_d']
    # Initialize indicators of cross correlation fit
    nt = len(Year)
    depth = np.zeros(nt)
    dx = np.zeros(nt)
    dy = np.zeros(nt)
    dz = np.zeros(nt)
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
        time = ymdhms2matlab(Year[i], Month[i], Day[i], Hour[i], Minute[i], \
            Second[i])
        find = np.where(mbbp_cat_d[:, 0] == time)
        tremor = mbbp_cat_d[find, :][0, 0, :]
        depth[i] = tremor[4]
        dx[i] = tremor[5]
        dy[i] = tremor[6]
        dz[i] = tremor[7]
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
    # Plots
    # Figure 1: Location uncertainty with cross correlation
    plt.figure(1, figsize=(30, 30))
    # Location uncertainty with max cross correlation
    ax1 = plt.subplot(331)
    plt.plot(dx, ccmaxEW, 'ro', label='East')
    plt.plot(dx, ccmaxNS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Max cross correlation')
    plt.title('East direction')
    plt.legend(loc=1)
    ax2 = plt.subplot(332)
    plt.plot(dy, ccmaxEW, 'ro', label='East')
    plt.plot(dy, ccmaxNS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Max cross correlation')
    plt.title('North direction')
    plt.legend(loc=1)
    ax3 = plt.subplot(333)
    plt.plot(dz, ccmaxEW, 'ro', label='East')
    plt.plot(dz, ccmaxNS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Max cross correlation')
    plt.title('Vertical direction')
    plt.legend(loc=1)
    # Location uncertainty with cross correlation at zero time lag
    ax4 = plt.subplot(334)
    plt.plot(dx, cc0EW, 'ro', label='East')
    plt.plot(dx, cc0NS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Cross correlation at 0')
    plt.title('East direction')
    plt.legend(loc=1)
    ax5 = plt.subplot(335)
    plt.plot(dy, cc0EW, 'ro', label='East')
    plt.plot(dy, cc0NS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Cross correlation at 0')
    plt.title('North direction')
    plt.legend(loc=1)
    ax6 = plt.subplot(336)
    plt.plot(dz, cc0EW, 'ro', label='East')
    plt.plot(dz, cc0NS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Cross correlation at 0')
    plt.title('Vertical direction')
    plt.legend(loc=1)
    # Location uncertainty with time delay
    ax7 = plt.subplot(337)
    plt.plot(dx, timedelayEW, 'ro', label='East')
    plt.plot(dx, timedelayNS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Time delay (s)')
    plt.title('East direction')
    plt.legend(loc=1)
    ax8 = plt.subplot(338)
    plt.plot(dy, timedelayEW, 'ro', label='East')
    plt.plot(dy, timedelayNS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Time delay (s)')
    plt.title('North direction')
    plt.legend(loc=1)
    ax9 = plt.subplot(339)
    plt.plot(dz, timedelayEW, 'ro', label='East')
    plt.plot(dz, timedelayNS, 'bo', label='North')
    plt.xlabel('Error (km)')
    plt.ylabel('Time delay (s)')
    plt.title('Vertical direction')
    plt.legend(loc=1)
    # End figure
    plt.suptitle('{} at {} km, {} km ({} - {})'.format(arrayName, x0, y0, \
        type_stack, cc_stack), fontsize=24)
    plt.savefig('cc/{}_{:03d}_{:03d}_{}_{}_loc.eps'.format(arrayName, int(x0), \
        int(y0), type_stack, cc_stack), format='eps')
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()
    ax5.clear()
    ax6.clear()
    ax7.clear()
    ax8.clear()
    ax9.clear()
    plt.close(1)
    # Figure 2: Depth with cross correlation
    plt.figure(2, figsize=(30, 10))
    ax1 = plt.subplot(131)
    # Depth with max cross correlation
    plt.plot(depth, ccmaxEW, 'ro', label='East')
    plt.plot(depth, ccmaxNS, 'bo', label='North')
    plt.xlabel('Depth (km)')
    plt.ylabel('Max cross correlation')
    plt.legend(loc=1)
    ax2 = plt.subplot(132)
    # Depth with cross correlation at zero time lag
    plt.plot(depth, cc0EW, 'ro', label='East')
    plt.plot(depth, cc0NS, 'bo', label='North')
    plt.xlabel('Depth (km)')
    plt.ylabel('Cross correlation at 0')
    plt.legend(loc=1)
    ax3 = plt.subplot(133)
    # Depth with time delay
    plt.plot(depth, timedelayEW, 'ro', label='East')
    plt.plot(depth, timedelayNS, 'bo', label='North')
    plt.xlabel('Depth (km)')
    plt.ylabel('Time delay (s)')
    plt.legend(loc=1)
    # End figure
    plt.suptitle('{} at {} km, {} km ({} - {})'.format(arrayName, x0, y0, \
        type_stack, cc_stack), fontsize=24)
    plt.savefig('cc/{}_{:03d}_{:03d}_{}_{}_depth.eps'.format(arrayName, \
        int(x0), int(y0), type_stack, cc_stack), format='eps')
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
    ncor = 120
    Tmin = 2.0
    Tmax = 8.0

    compare_ccorr_location(arrayName, x0, y0, 'lin', w, 'lin', ncor, \
        Tmin, Tmax)
    compare_ccorr_location(arrayName, x0, y0, 'lin', w, 'pow', ncor, \
        Tmin, Tmax)
    compare_ccorr_location(arrayName, x0, y0, 'lin', w, 'PWS', ncor, \
        Tmin, Tmax)
    compare_ccorr_location(arrayName, x0, y0, 'pow', w, 'lin', ncor, \
        Tmin, Tmax)
    compare_ccorr_location(arrayName, x0, y0, 'pow', w, 'pow', ncor, \
        Tmin, Tmax)
    compare_ccorr_location(arrayName, x0, y0, 'pow', w, 'PWS', ncor, \
        Tmin, Tmax)
    compare_ccorr_location(arrayName, x0, y0, 'PWS', w, 'lin', ncor, \
        Tmin, Tmax)
    compare_ccorr_location(arrayName, x0, y0, 'PWS', w, 'pow', ncor, \
        Tmin, Tmax)
    compare_ccorr_location(arrayName, x0, y0, 'PWS', w, 'PWS', ncor, \
        Tmin, Tmax)
