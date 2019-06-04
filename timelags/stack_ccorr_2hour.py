"""
This module contains a function to download 2 hours of data,
stack the signal over the stations, cross correlate 30-second-long
time windows, and plot
"""

import obspy
import obspy.clients.earthworm.client as earthworm
import obspy.clients.fdsn.client as fdsn
from obspy import read_inventory
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.signal.cross_correlation import correlate

import matplotlib.pyplot as plt
import numpy as np
import sys

from stacking import linstack, powstack, PWstack

def stack_ccorr_2hour(arrayName, staNames, staCodes, chaNames, chans, \
        network, myYear, myMonth, dStarts, hStarts, TDUR, tdur, filt, \
        type_stack, offset, w, ncor, Tmax, scal, client):
    """
    This function downloads 2 hours of data, stack the signal over the
    stations, cross correlate 30-second-long time windows, and plot
    
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
        type myYear = integer
        myYear = Year when we download the data
        type myMonth = integer
        myMonth = Month when we download the data
        type dStarts = numpy array of integers
        dStarts = Days when we download the data
        type hStarts = numpy array of integers
        hStarts = Hours when we download the data
        type TDUR = float
        TDUR = Duration of the signal we want to download
        type tdur = float
        tdur = Duration of the time windows we want to cross correlate
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type type_stack = string
        type_stack = Type of stack ('lin', 'pow', 'PWS')
        type offset = float
        offset = Vertical offset between time windows for plotting
        type w = float
        w = Power of the stack (for 'pow' and 'PWS')
        type ncor = integer
        ncor = Number of points for the cross correlation
        type Tmax = float
        Tmax = Maximum time lag for cross correlation plot
        type scal = float
        scale = Scale for envelope running along the right side of the plots
        type client = string
        client = Server from which we download the data ('IRIS', 'Rainier')
    Output:
        None
    """
    # Create client
    if client == 'IRIS':
        fdsn_client = fdsn.Client('IRIS')
    elif client == 'Rainier':
        earthworm_client = earthworm.Client('rainier.ess.washington.edu', \
            16017)
    else:
         raise ValueError('Data must be imported from IRIS or Rainier')       

    # Loop on days and hours of the day
    for dStart in dStarts:
        for hStart in hStarts:
            Tstart = UTCDateTime(year=myYear, month=myMonth, day=dStart, \
                hour=hStart)
            Tend = Tstart + TDUR
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
                print('Cannot open waveform file for day = {}, hour = {}'. \
                    format(dStart, hStart))
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
                # Stack data over all stations
                stack = linstack([D])
                # Compute the envelope of the stacked signal
                envelope = stack.copy()
                for kchan in range(0, len(stack)):
                    envelope[kchan].data = obspy.signal.filter.envelope( \
                        stack[kchan].data)
                envelope.decimate(10, no_filter=True)
                tstack = (1.0 / envelope[0].stats.sampling_rate) * \
                    np.arange(0, envelope[0].stats.npts)
                # Create figure
                plt.figure(1, figsize=(20, 15))
                # EW - UD cross correlation
                ax = plt.subplot(121)
                # Loop over tdur - second long time windows
                for kt in range(0, int(TDUR / tdur)):
                    EW = D.select(component='E').slice(Tstart + kt * tdur, \
                        Tstart + (kt + 1) * tdur)
                    UD = D.select(component='Z').slice(Tstart + kt * tdur, \
                        Tstart + (kt + 1) * tdur)
                    cc = Stream()
                    # Cross correlate the short time windows at each station
                    for ksta in range(0, len(staNames)):
                        if (D.select(station=staNames[ksta], \
                                     channel=chaNames[0]) and \
                            D.select(station=staNames[ksta], \
                                     channel=chaNames[1]) and \
                            D.select(station=staNames[ksta], \
                                     channel=chaNames[2])):
                            cc.append(EW.select(station=staNames[ksta]) \
                                [0].copy())
                            cc[-1].data = correlate( \
                                EW.select(station=staNames[ksta])[0], \
                                UD.select(station=staNames[ksta])[0], ncor)
                            cc[-1].stats['channel'] = 'CC'
                            cc[-1].stats['station'] = staNames[ksta]
                    # Time vector
                    t = (1.0 / EW[0].stats.sampling_rate) * \
                        np.arange(- ncor, ncor + 1)
                    # Stack cross correlations within the array and plot
                    if (type_stack == 'lin'):
                        plt.plot(t, kt * offset + linstack([cc], \
                            normalize=False)[0].data, 'k-')
                    elif (type_stack == 'pow'):
                        plt.plot(t, kt * offset + powstack([cc], w, \
                            normalize=False)[0].data, 'k-')
                    elif (type_stack == 'PWS'):
                        plt.plot(t, kt * offset + PWstack([cc], w, \
                            normalize=False)[0].data, 'k-')
                    else:
                        raise ValueError( \
                            'Type of stack must be lin, pow, or PWS')
                # Plot the envelope along the right side of the plot
                plt.plot(Tmax - scal * envelope.select(component='E') \
                    [0].data, tstack / tdur * offset, 'k-')
                # Finalize left-hand part of the figure
                plt.xlim(0, Tmax)
                plt.ylim(0, TDUR / tdur * offset)
                plt.title('{0}/{1}/{2} {3}-{4} {5} {6}'.format( \
                    Tstart.month, Tstart.day, Tstart.year, Tstart.hour, \
                    Tend.hour, arrayName, 'EW - UD'), fontsize=24)
                plt.xlabel('Lag time (s)', fontsize=24)
                ax.set_yticklabels([])
                ax.tick_params(labelsize=20)
                # NS - UD cross correlation
                ax = plt.subplot(122)
                # Loop over tdur - second long time windows
                for kt in range(0, int(TDUR / tdur)):
                    NS = D.select(component='N').slice(Tstart + kt * tdur, \
                        Tstart + (kt + 1) * tdur)
                    UD = D.select(component='Z').slice(Tstart + kt * tdur, \
                        Tstart + (kt + 1) * tdur)
                    cc = Stream()
                    # Cross correlate the short time windows at each station
                    for ksta in range(0, len(staNames)):
                        if (D.select(station=staNames[ksta], \
                                     channel=chaNames[0]) and \
                            D.select(station=staNames[ksta], \
                                     channel=chaNames[1]) and \
                            D.select(station=staNames[ksta], \
                                     channel=chaNames[2])):
                            cc.append(NS.select(station=staNames[ksta]) \
                                [0].copy())
                            cc[-1].data = correlate( \
                                NS.select(station=staNames[ksta])[0], \
                                UD.select(station=staNames[ksta])[0], ncor)
                            cc[-1].stats['channel'] = 'CC'
                            cc[-1].stats['station'] = staNames[ksta]
                    # Time vector
                    t = (1.0 / NS[0].stats.sampling_rate) * \
                        np.arange(- ncor, ncor + 1)
                    # Stack cross correlations within the array and plot
                    if (type_stack == 'lin'):
                        plt.plot(t, kt * offset + linstack([cc], \
                            normalize=False)[0].data, 'k-')
                    elif (type_stack == 'pow'):
                        plt.plot(t, kt * offset + powstack([cc], w, \
                            normalize=False)[0].data, 'k-')
                    elif (type_stack == 'PWS'):
                        plt.plot(t, kt * offset + PWstack([cc], w, \
                            normalize=False)[0].data, 'k-')
                    else:
                        raise ValueError( \
                            'Type of stack must be lin, pow, or PWS')
                # Plot the envelope along the right side of plot
                plt.plot(Tmax - scal * envelope.select(component='N') \
                    [0].data, tstack / tdur * offset, 'k-')
                # Finalize right-hand part of the figure
                plt.xlim(0, Tmax)
                plt.ylim(0, TDUR / tdur * offset)
                plt.title('{0}/{1}/{2} {3}-{4} {5} {6}'.format( \
                    Tstart.month, Tstart.day, Tstart.year, Tstart.hour, \
                    Tend.hour, arrayName, 'NS - UD'), fontsize=24)
                plt.xlabel('Lag time (s)', fontsize=24)
                ax.set_yticklabels([])
                ax.tick_params(labelsize=20)
                # Save figure
                plt.savefig('plot_cc/{0}_{1}_{2}_{3}_{4}_{5}.eps'.format( \
                    arrayName, Tstart.year, Tstart.month,\
                    Tstart.day, Tstart.hour, type_stack), format='eps')
                ax.clear()
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

    myYear = 2010
    myMonth = 8
    dStarts = np.arange(17, 18, 1)
    hStarts = np.arange(6, 8, 2)
    TDUR = 2 * 3600.0
    tdur = 30.0
    filt = (2, 8)
    w = 2.0
    ncor = 400
    Tmax = 15.0
    scal = 0.5

    # Linear stack
    type_stack = 'lin'
    offset = 0.1
    stack_ccorr_2hour(arrayName, staNames, staCodes, chaNames, chans, \
        network, myYear, myMonth, dStarts, hStarts, TDUR, tdur, filt, \
        type_stack, offset, w, ncor, Tmax, scal, client)
        
    # Power stack
    type_stack = 'pow'
    offset = 0.5
    stack_ccorr_2hour(arrayName, staNames, staCodes, chaNames, chans, \
        network, myYear, myMonth, dStarts, hStarts, TDUR, tdur, filt, \
        type_stack, offset, w, ncor, Tmax, scal, client)

    # Phase-weighted stack
    type_stack = 'PWS'
    offset = 0.05
    stack_ccorr_2hour(arrayName, staNames, staCodes, chaNames, chans, \
        network, myYear, myMonth, dStarts, hStarts, TDUR, tdur, filt, \
        type_stack, offset, w, ncor, Tmax, scal, client)
