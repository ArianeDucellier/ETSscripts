"""
This module contains a function to plot the stack of the autocorrelation
using only selected tremor windows
"""

import obspy
import obspy.clients.earthworm.client as earthworm
import obspy.clients.fdsn.client as fdsn
from obspy import read_inventory
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.signal.cross_correlation import correlate

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from stacking import linstack, powstack, PWstack

def stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
        network, x0, y0, TDUR, filt, type_stack, w, ncor, cc_stack, \
        ncor_stack, Tmin, Tmax, xmax, ymax, Emax, Nmax, E0, N0, Et, Nt, \
        client):
    """
    This function compute the autocorrelation, stack over selected tremor
    windows and plot the stack

    Input:
        type arrayName = string
        arrayName = Name of seismic array
        type staName = list of strings
        staNames = Names of the seismic stations
        type staCodes = string
        staCodes = Names of the seismic stations
        type chaNames = list of strings
        chaNames = Names of the channels
        type chans = string
        chans = Names of the channels
        type network = string
        network = Name of the seismic network
        type x0 = float
        x0 = Distance of the center of the cell from the array (east)
        type y0 = float
        y0 = Distance of the center of the cell from the array (north)
        type TDUR = float
        TDUR = Duration of data downloaded before and after window of interest
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type type_stack = string
        type_stack = Type of stack ('lin', 'pow', 'PWS')
        type w = float
        w = Power of the stack (for 'pow' and 'PWS')
        type ncor = integer
        ncor = Number of points for the cross correlation
        type cc_stack = string
        cc_stack = Type of stack ('lin', 'pow', 'PWS') over tremor windows
        type ncor_stack = integer
        ncor_stack = Number of points for the cross correlation with the stack
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
        type client = string
        client = Server from which we download the data ('IRIS', 'Rainier')
    Output:
        None
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
        EW_UD_stack = linstack([EW_UD], normalize=False)[0]
        NS_UD_stack = linstack([NS_UD], normalize=False)[0]
    elif (cc_stack == 'pow'):
        EW_UD_stack = powstack([EW_UD], w, normalize=False)[0]
        NS_UD_stack = powstack([NS_UD], w, normalize=False)[0]
    elif (cc_stack == 'PWS'):
        EW_UD_stack = PWstack([EW_UD], w, normalize=False)[0]
        NS_UD_stack = PWstack([NS_UD], w, normalize=False)[0]
    else:
        raise ValueError( \
            'Type of stack must be lin, pow, or PWS')
    # Create client
    if client == 'IRIS':
        fdsn_client = fdsn.Client('IRIS')
    elif client == 'Rainier':
        earthworm_client = earthworm.Client('rainier.ess.washington.edu', \
            16017)
    else:
         raise ValueError('Data must be imported from IRIS or Rainier')       
    # Initialize streams to store cross correlations
    nt = len(EW_UD)
    EW_all = Stream()
    NS_all = Stream()
    UD_all = Stream()
    # Loop on tremor windows
    for i in range(0, nt):
        t1 = UTCDateTime(str(Year[i]) + '-' + str(Month[i]) + '-' + \
            str(Day[i]) + 'T' + str(Hour[i]) + ':' + str(Minute[i]) + ':' + \
            str(Second[i]))
        Tstart = t1 - TDUR
        t2 = t1 + 60.0
        Tend = t2 + TDUR
        # Get data from server
        try:
            if client == 'IRIS':
                Dtmp = fdsn_client.get_waveforms(network=network, \
                    station=staCodes, location='--', channel=chans, \
                    starttime=Tstart, endtime=Tend, attach_response=True)
            else:
                Dtmp = Stream()
                for ksta in range(0, len(staNames)):
                    for kchan in range(0, len(chaNames)):
                        trace = earthworm_client.get_waveforms( \
                            network=network, station=staNames[ksta], \
                            location='', channel=chaNames[kchan], \
                            starttime=Tstart, endtime=Tend)
                        if len(trace) > 0:
                            Dtmp.append(trace[0])
        except:
            message = 'Cannot open waveform file for tremor {} '. \
                format(i + 1) + \
                '({:04d}/{:02d}/{:02d} at {:02d}:{:02d}:{:02d})'. \
                format(Year[i], Month[i], Day[i], Hour[i], Minute[i], \
                Second[i])
            print(message)
        else:
            # Remove stations that have different amounts of data
            ntmp = []
            for ksta in range(0, len(Dtmp)):
                ntmp.append(len(Dtmp[ksta]))
            ntmp = max(set(ntmp), key=ntmp.count)
            D = Dtmp.select(npts=ntmp)
            # Detrend data
            D.detrend(type='linear')
            # Taper first and last 5 s of data
            D.taper(type='hann', max_percentage=None, max_length=5.0)
            # Remove instrument response
            if client == 'Rainier':
                filename = '../data/response/' + network + '_' + \
                    arrayName + '.xml'
                inventory = read_inventory(filename, format='STATIONXML')
                D.attach_response(inventory)
            D.remove_response(output='VEL', \
                pre_filt=(0.2, 0.5, 10.0, 15.0), water_level=80.0)
            # Filter
            D.filter('bandpass', freqmin=filt[0], freqmax=filt[1], \
                zerophase=True)
            # Resample data to .05 s
            D.interpolate(100.0, method='lanczos', a=10)
            D.decimate(5, no_filter=True)         
            # Cut data for cross correlation
            EW = D.select(component='E').slice(t1, t2)
            NS = D.select(component='N').slice(t1, t2)
            UD = D.select(component='Z').slice(t1, t2)
            # EW autocorrelation
            cc = Stream()
            for ksta in range(0, len(staNames)):
                if (D.select(station=staNames[ksta], component='E')):
                    cc.append(EW.select(station=staNames[ksta])[0].copy())
                    cc[-1].data = correlate( \
                        EW.select(station=staNames[ksta])[0], \
                        EW.select(station=staNames[ksta])[0], ncor)
                    cc[-1].stats['station'] = staNames[ksta]
            # Stack cross correlations within the array
            if (type_stack == 'lin'):
                stack = linstack([cc], normalize=False)[0]
            elif (type_stack == 'pow'):
                stack = powstack([cc], w, normalize=False)[0]
            elif (type_stack == 'PWS'):
                stack = PWstack([cc], w, normalize=False)[0]
            else:
                raise ValueError( \
                    'Type of stack must be lin, pow, or PWS')
            # Keep value of stack
            EW_all.append(stack)
            # NS autocorrelation
            cc = Stream()
            for ksta in range(0, len(staNames)):
                if (D.select(station=staNames[ksta], component='N')):
                    cc.append(NS.select(station=staNames[ksta])[0].copy())
                    cc[-1].data = correlate( \
                        NS.select(station=staNames[ksta])[0], \
                        NS.select(station=staNames[ksta])[0], ncor)
                    cc[-1].stats['station'] = staNames[ksta]
            # Stack cross correlations within the array
            if (type_stack == 'lin'):
                stack = linstack([cc], normalize=False)[0]
            elif (type_stack == 'pow'):
                stack = powstack([cc], w, normalize=False)[0]
            elif (type_stack == 'PWS'):
                stack = PWstack([cc], w, normalize=False)[0]
            else:
                raise ValueError( \
                    'Type of stack must be lin, pow, or PWS')
            # Keep value of stack
            NS_all.append(stack)
            # Vertical autocorrelation
            cc = Stream()
            for ksta in range(0, len(staNames)):
                if (D.select(station=staNames[ksta], component='Z')):
                    cc.append(UD.select(station=staNames[ksta])[0].copy())
                    cc[-1].data = correlate( \
                        UD.select(station=staNames[ksta])[0], \
                        UD.select(station=staNames[ksta])[0], ncor)
                    cc[-1].stats['station'] = staNames[ksta]
            # Stack cross correlations within the array
            if (type_stack == 'lin'):
                stack = linstack([cc], normalize=False)[0]
            elif (type_stack == 'pow'):
                stack = powstack([cc], w, normalize=False)[0]
            elif (type_stack == 'PWS'):
                stack = PWstack([cc], w, normalize=False)[0]
            else:
                raise ValueError( \
                    'Type of stack must be lin, pow, or PWS')
            # Keep value of stack
            UD_all.append(stack)
    # Stack over all tremor windows
    if (cc_stack == 'lin'):
        EW = linstack([EW_all], normalize=False)[0]
        NS = linstack([NS_all], normalize=False)[0]
        UD = linstack([UD_all], normalize=False)[0]
    elif (cc_stack == 'pow'):
        EW = powstack([EW_all], w, normalize=False)[0]
        NS = powstack([NS_all], w, normalize=False)[0]
        UD = powstack([UD_all], w, normalize=False)[0]
    elif (cc_stack == 'PWS'):
        EW = PWstack([EW_all], w, normalize=False)[0]
        NS = PWstack([NS_all], w, normalize=False)[0]
        UD = PWstack([UD_all], w, normalize=False)[0]
    else:
        raise ValueError( \
            'Type of stack must be lin, pow, or PWS')
    # Initialize indicators of cross correlation fit
    ccmaxEW = np.zeros(nt)
    cc0EW = np.zeros(nt)
    timedelayEW = np.zeros(nt)
    ccmaxNS = np.zeros(nt)
    cc0NS = np.zeros(nt)
    timedelayNS = np.zeros(nt)
    # Windows of the cross correlation to look at
    i0 = int((len(EW_UD_stack) - 1) / 2)
    ibegin = i0 + int(Tmin / EW_UD_stack.stats.delta)
    iend = i0 + int(Tmax / EW_UD_stack.stats.delta) + 1
    for i in range(0, nt):
        # Cross correlate cc for EW with stack       
        cc_EW = correlate(EW_UD[i][ibegin : iend], \
            EW_UD_stack[ibegin : iend], ncor_stack)
        ccmaxEW[i] = np.max(cc_EW)
        cc0EW[i] = cc_EW[ncor_stack]
        timedelayEW[i] = (np.argmax(cc_EW) - ncor_stack) * \
            EW_UD_stack.stats.delta
        # Cross correlate cc for NS with stack
        cc_NS = correlate(NS_UD[i][ibegin : iend], \
            NS_UD_stack[ibegin : iend], ncor_stack)
        ccmaxNS[i] = np.max(cc_NS)
        cc0NS[i] = cc_NS[ncor_stack]
        timedelayNS[i] = (np.argmax(cc_NS) - ncor_stack) * \
            NS_UD_stack.stats.delta
    # Plot
    plt.figure(1, figsize=(45, 30))
    npts = int((EW_UD_stack.stats.npts - 1) / 2)
    dt = EW_UD_stack.stats.delta
    t = dt * np.arange(- npts, npts + 1)
    # EW. Select with max cross correlation
    ax1 = plt.subplot(331)
    plt.plot(t, EW.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Emax)))
    for j in range(0, len(Emax)):
        EWselect = Stream()
        for i in range(0, nt):
            if ((ccmaxEW[i] >= Emax[j]) and (ccmaxNS[i] >= Nmax[j])):
                EWselect.append(EW_all[i])
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
    plt.title('EW (Max cross correlation)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # NS. Select with max cross correlation
    ax2 = plt.subplot(334)
    plt.plot(t, NS.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Nmax)))
    for j in range(0, len(Nmax)):
        NSselect = Stream()
        for i in range(0, nt):
            if ((ccmaxEW[i] >= Emax[j]) and (ccmaxNS[i] >= Nmax[j])):
                NSselect.append(NS_all[i])
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
    plt.title('NS (Max cross correlation)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # Vertical. Select with max cross correlation
    ax3 = plt.subplot(337)
    plt.plot(t, UD.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Nmax)))
    for j in range(0, len(Nmax)):
        UDselect = Stream()
        for i in range(0, nt):
            if ((ccmaxEW[i] >= Emax[j]) and (ccmaxNS[i] >= Nmax[j])):
                UDselect.append(UD_all[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            UDstack = linstack([UDselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            UDstack = powstack([UDselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            UDstack = PWstack([UDselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, UDstack.data, color = colors[j], \
            label='Emax = {:3.2f}, Nmax = {:3.2f}'.format(Emax[j], Nmax[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Vertical (Max cross correlation)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # EW. Select with cross correlation at zero time lag
    ax4 = plt.subplot(332)
    plt.plot(t, EW.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(E0)))
    for j in range(0, len(E0)):
        EWselect = Stream()
        for i in range(0, nt):
            if ((cc0EW[i] >= E0[j]) and (cc0NS[i] >= N0[j])):
                EWselect.append(EW_all[i])
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
    plt.title('EW (Cross correlation at 0)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # NS. Select with cross correlation at zero time lag
    ax5 = plt.subplot(335)
    plt.plot(t, NS.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(N0)))
    for j in range(0, len(N0)):
        NSselect = Stream()
        for i in range(0, nt):
            if ((cc0EW[i] >= E0[j]) and (cc0NS[i] >= N0[j])):
                NSselect.append(NS_all[i])
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
    plt.title('NS (Cross correlation at 0)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # Vertical. Select with cross correlation at zero time lag
    ax6 = plt.subplot(338)
    plt.plot(t, UD.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(N0)))
    for j in range(0, len(N0)):
        UDselect = Stream()
        for i in range(0, nt):
            if ((cc0EW[i] >= E0[j]) and (cc0NS[i] >= N0[j])):
                UDselect.append(UD_all[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            UDstack = linstack([UDselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            UDstack = powstack([UDselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            UDstack = PWstack([UDselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, UDstack.data, color = colors[j], \
            label='E0 = {:3.2f}, N0 = {:3.2f}'.format(E0[j], N0[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Vertical (Cross correlation at 0)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # EW. Select with time delay
    ax7 = plt.subplot(333)
    plt.plot(t, EW.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Et)))
    for j in range(0, len(Et)):
        EWselect = Stream()
        for i in range(0, nt):
            if ((timedelayEW[i] <= Et[j]) and (timedelayNS[i] <= Nt[j])):
                EWselect.append(EW_all[i])
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
    plt.title('EW (Time delay)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # NS. Select with time delay
    ax8 = plt.subplot(336)
    plt.plot(t, NS.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Nt)))
    for j in range(0, len(Nt)):
        NSselect = Stream()
        for i in range(0, nt):
            if ((timedelayEW[i] <= Et[j]) and (timedelayNS[i] <= Nt[j])):
                NSselect.append(NS_all[i])
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
    plt.title('NS (Time delay)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # Vertical. Select with time delay
    ax9 = plt.subplot(339)
    plt.plot(t, UD.data, 'k-', label='All')
    colors = cm.rainbow(np.linspace(0, 1, len(Nt)))
    for j in range(0, len(Nt)):
        UDselect = Stream()
        for i in range(0, nt):
            if ((timedelayEW[i] <= Et[j]) and (timedelayNS[i] <= Nt[j])):
                UDselect.append(UD_all[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            UDstack = linstack([UDselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            UDstack = powstack([UDselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            UDstack = PWstack([UDselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, UDstack.data, color = colors[j], \
            label='Et = {:3.2f}, Nt = {:3.2f}'.format(Et[j], Nt[j]))
    plt.xlim(0, xmax)
    plt.ylim(- ymax, ymax)
    plt.title('Vertical (Time delay)', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.legend(loc=1)
    # End figure
    plt.suptitle('{} at {} km, {} km ({} - {})'.format(arrayName, x0, y0, \
        type_stack, cc_stack), fontsize=24)
    plt.savefig('ac/{}_{:03d}_{:03d}_{}_{}_select.eps'.format(arrayName, \
        int(x0), int(y0), type_stack, cc_stack), format='eps')
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

if __name__ == '__main__':

    # Get the name of the client
    client = sys.argv[1]

    # Set the parameters

#    arrayName = 'BH'
#    staNames = ['BH01', 'BH02', 'BH03', 'BH04', 'BH05', 'BH06', 'BH07', 'BH08', 'BH09', 'BH10', 'BH11']
#    staCodes = 'BH01,BH02,BH03,BH04,BH05,BH06,BH07,BH08,BH09,BH10,BH11'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'

    arrayName = 'BS'
    staNames = ['BS01', 'BS02', 'BS03', 'BS04', 'BS05', 'BS06', 'BS11', 'BS20', 'BS21', 'BS22', 'BS23', 'BS24', 'BS25', 'BS26', 'BS27']
    staCodes = 'BS01,BS02,BS03,BS04,BS05,BS06,BS11,BS20,BS21,BS22,BS23,BS24,BS25,BS26,BS27'
    chaNames = ['SHE', 'SHN', 'SHZ']
    chans = 'SHE,SHN,SHZ'
    network = 'XU'

#    arrayName = 'CL'
#    staNames = ['CL01', 'CL02', 'CL03', 'CL04', 'CL05', 'CL06', 'CL07', 'CL08', 'CL09', 'CL10', 'CL11', 'CL12', 'CL13', 'CL14', 'CL15', 'CL16', 'CL17', 'CL18', 'CL19', 'CL20']
#    staCodes = 'CL01,CL02,CL03,CL04,CL05,CL06,CL07,CL08,CL09,CL10,CL11,CL12,CL13,CL14,CL15,CL16,CL17,CL18,CL19,CL20'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'

#    arrayName = 'DR'
#    staNames = ['DR01', 'DR02', 'DR03', 'DR04', 'DR05', 'DR06', 'DR07', 'DR08', 'DR09', 'DR10', 'DR12']
#    staCodes = 'DR01,DR02,DR03,DR04,DR05,DR06,DR07,DR08,DR09,DR10,DR12'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'

#    arrayName = 'GC'
#    staNames = ['GC01', 'GC02', 'GC03', 'GC04', 'GC05', 'GC06', 'GC07', 'GC08', 'GC09', 'GC10', 'GC11', 'GC12', 'GC13', 'GC14']
#    staCodes = 'GC01,GC02,GC03,GC04,GC05,GC06,GC07,GC08,GC09,GC10,GC11,GC12,GC13,GC14'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'

#    arrayName = 'LC'
#    staNames = ['LC01', 'LC02', 'LC03', 'LC04', 'LC05', 'LC06', 'LC07', 'LC08', 'LC09', 'LC10', 'LC11', 'LC12', 'LC13', 'LC14']
#    staCodes = 'LC01,LC02,LC03,LC04,LC05,LC06,LC07,LC08,LC09,LC10,LC11,LC12,LC13,LC14'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'

#    arrayName = 'PA'
#    staNames = ['PA01', 'PA02', 'PA03', 'PA04', 'PA05', 'PA06', 'PA07', 'PA08', 'PA09', 'PA10', 'PA11', 'PA12', 'PA13']
#    staCodes = 'PA01,PA02,PA03,PA04,PA05,PA06,PA07,PA08,PA09,PA10,PA11,PA12,PA13'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'

#    arrayName = 'TB'
#    staNames = ['TB01', 'TB02', 'TB03', 'TB04', 'TB05', 'TB06', 'TB07', 'TB08', 'TB09', 'TB10', 'TB11', 'TB12', 'TB13', 'TB14']
#    staCodes = 'TB01,TB02,TB03,TB04,TB05,TB06,TB07,TB08,TB09,TB10,TB11,TB12,TB13,TB14'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'

    x0 = 0.0
    y0 = 0.0
    TDUR = 10.0
    filt = (2, 8)
    w = 2.0
    ncor = 400
    ncor_stack = 120
    Tmin = 2.0
    Tmax = 8.0
    xmax = 15.0
    Emax = [0.3, 0.4, 0.5, 0.6]
    Nmax = [0.3, 0.4, 0.5, 0.6]
    E0 = [0.2, 0.3, 0.4, 0.5]
    N0 = [0.2, 0.3, 0.4, 0.5]
    Et = [0.0, 0.05, 0.1, 0.15, 0.2]
    Nt = [0.0, 0.05, 0.1, 0.15, 0.2]

#    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
#        network, x0, y0, TDUR, filt, 'lin', w, ncor, 'lin', ncor_stack, Tmin, \
#        Tmax, xmax, 0.1, Emax, Nmax, E0, N0, Et, Nt, client)
#    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
#        network, x0, y0, TDUR, filt, 'lin', w, ncor, 'pow', ncor_stack, Tmin, \
#        Tmax, xmax, 0.2, Emax, Nmax, E0, N0, Et, Nt, client)
#    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
#        network, x0, y0, TDUR, filt, 'lin', w, ncor, 'PWS', ncor_stack, Tmin, \
#        Tmax, xmax, 0.05, Emax, Nmax, E0, N0, Et, Nt, client)
    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
        network, x0, y0, TDUR, filt, 'pow', w, ncor, 'lin', ncor_stack, Tmin, \
        Tmax, xmax, 0.2, Emax, Nmax, E0, N0, Et, Nt, client)
    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
        network, x0, y0, TDUR, filt, 'pow', w, ncor, 'pow', ncor_stack, Tmin, \
        Tmax, xmax, 1.0, Emax, Nmax, E0, N0, Et, Nt, client)
    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
        network, x0, y0, TDUR, filt, 'pow', w, ncor, 'PWS', ncor_stack, Tmin, \
        Tmax, xmax, 0.15, Emax, Nmax, E0, N0, Et, Nt, client)
    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
        network, x0, y0, TDUR, filt, 'PWS', w, ncor, 'lin', ncor_stack, Tmin, \
        Tmax, xmax, 0.02, Emax, Nmax, E0, N0, Et, Nt, client)
    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
        network, x0, y0, TDUR, filt, 'PWS', w, ncor, 'pow', ncor_stack, Tmin, \
        Tmax, xmax, 0.2, Emax, Nmax, E0, N0, Et, Nt, client)
    stack_acorr_select(arrayName, staNames, staCodes, chaNames, chans, \
        network, x0, y0, TDUR, filt, 'PWS', w, ncor, 'PWS', ncor_stack, Tmin, \
        Tmax, xmax, 0.01, Emax, Nmax, E0, N0, Et, Nt, client)
