"""
This module contains a function to download every one-minute time window
where there is a tremor at a given location, stack the signal over the
stations, cross correlate, and plot
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
import os
import pickle
import sys
import time

from math import pi, cos, sin, sqrt
from scipy.io import loadmat

from date import matlab2ymdhms
from stacking import linstack, powstack, PWstack

def stack_ccorr_tremor(arrayName, staNames, staCodes, chaNames, chans, \
    network, lat0, lon0, ds, x0, y0, TDUR, filt, type_stack, w, ncor, Tmax, \
    amp, amp_stack, draw_plot, client, nattempts, waittime):
    """
    This function download every one-minute time window where there is a
    tremor at a given location, stack the signal over the stations, cross
    correlate, and plot
    
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
        type lat0 = float
        lat0 = Latitude of the center of the array
        type lon0 = float
        lon0 = Longitude of the center of the array
        type ds = float
        ds = Size of the cell where we look for tremor
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
        type Tmax = float
        Tmax = Maximum time lag for cross correlation plot
        type amp = float
        amp = Amplification factor of cross correlation for plotting
        type amp_stack = float
        amp_stack = Amplification factor of stack for plotting
        type draw_plot = boolean
        draw_plot = Do we draw the plot for every tremor window?
        type client = string
        client = Server from which we download the data ('IRIS', 'Rainier')
        type nattempts = integer
        nattempts = Number of times we try to download data
        type waittime = positive float
        waittime = Type to wait between two attempts at downloading
    Output:
        None
    """
    # Earth's radius and ellipticity
    a = 6378.136
    e = 0.006694470

    # Create client
    if client == 'IRIS':
        fdsn_client = fdsn.Client('IRIS')
    elif client == 'Rainier':
        earthworm_client = earthworm.Client('rainier.ess.washington.edu', \
            16017)
    else:
         raise ValueError('Data must be imported from IRIS or Rainier')       

    # Find tremors located in the cell (2009 - 2010)
    data = loadmat('../data/timelags/mbbp_cat_d_forHeidi')
    mbbp = data['mbbp_cat_d']
    lat = mbbp[:, 2]
    lon = mbbp[:, 3]
    dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
        sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
        pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
    lonmin = lon0 + (x0 - 0.5 * ds) / dx
    lonmax = lon0 + (x0 + 0.5 * ds) / dx 
    latmin = lat0 + (y0 - 0.5 * ds) / dy
    latmax = lat0 + (y0 + 0.5 * ds) / dy
    find = np.where((lat >= latmin) & (lat <= latmax) & \
                    (lon >= lonmin) & (lon <= lonmax))
    tremor1 = mbbp[find, :][0, :, :]

    # Find tremors located in the cell (August - September 2011)
    data = loadmat('../data/timelags/mbbp_ets12')
    mbbp = data['mbbp_ets12']
    lat = mbbp[:, 2]
    lon = mbbp[:, 3]
    dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
        sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
        pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)
    lonmin = lon0 + (x0 - 0.5 * ds) / dx
    lonmax = lon0 + (x0 + 0.5 * ds) / dx 
    latmin = lat0 + (y0 - 0.5 * ds) / dy
    latmax = lat0 + (y0 + 0.5 * ds) / dy
    find = np.where((lat >= latmin) & (lat <= latmax) & \
                    (lon >= lonmin) & (lon <= lonmax))
    tremor2 = mbbp[find, :][0, :, :]
    tremor = np.concatenate((tremor1[:, 0 : 8], tremor2[:, 0 : 8]), axis=0)
    nt = np.shape(tremor)[0]

    # Initialize streams to store cross correlations
    EW_UD = Stream()
    NS_UD = Stream()
    Year = []
    Month = []
    Day = []
    Hour = []
    Minute = []
    Second = []

    # File to write error messages
    namedir = 'error/cc/{}/'.format(arrayName)
    if not os.path.exists(namedir):
        os.makedirs(namedir)
    errorfile = namedir + 'error_{}_{:03d}_{:03d}.txt'.format( \
        arrayName, int(x0), int(y0))

    # Loop on tremor windows
    for i in range(0, nt):
        (YY1, MM1, DD1, HH1, mm1, ss1) = matlab2ymdhms(tremor[i, 0])
        t1 = UTCDateTime(str(YY1) + '-' + str(MM1) + '-' + str(DD1) + 'T' + \
            str(HH1) + ':' + str(mm1) + ':' + str(ss1))
        Tstart = t1 - TDUR
        (YY2, MM2, DD2, HH2, mm2, ss2) = matlab2ymdhms(tremor[i, 1])
        t2 = UTCDateTime(str(YY2) + '-' + str(MM2) + '-' + str(DD2) + 'T' + \
            str(HH2) + ':' + str(mm2) + ':' + str(ss2))
        Tend = t2 + TDUR
        # Loop to try downloading several times
        success = False
        attempts = 0
        while attempts < nattempts and not success:
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
                success = True
            except:
                message = 'Cannot open waveform file for tremor {} '. \
                    format(i + 1) + \
                    '({:04d}/{:02d}/{:02d} at {:02d}:{:02d}:{:02d})\n'. \
                    format(YY1, MM1, DD1, HH1, mm1, ss1)
                with open(errorfile, 'a') as file:
                    file.write(message)
                attempts += 1
                time.sleep(waittime)
                if attempts == nattempts:
                    with open(errorfile, 'a') as file:
                        file.write( \
                            'Failed to download data after {} attempts\n'. \
                            format(nattempts))
            else:
                # Cut data for cross correlation
                EW = D.select(component='E').slice(t1, t2)
                NS = D.select(component='N').slice(t1, t2)
                UD = D.select(component='Z').slice(t1, t2)
                # Time vector
                t = (1.0 / EW[0].stats.sampling_rate) * np.arange(- ncor, \
                    ncor + 1)
                # Create figure
                if (draw_plot == True):
                    plt.figure(1, figsize=(30, 15))
                # EW - UD cross correlation
                if (draw_plot == True):
                    ax = plt.subplot(211)
                cc = Stream()
                for ksta in range(0, len(staNames)):
                    if (D.select(station=staNames[ksta], \
                        channel=chaNames[0]) and \
                        D.select(station=staNames[ksta], \
                        channel=chaNames[1]) and \
                        D.select(station=staNames[ksta], \
                        channel=chaNames[2])):
                        cc.append(EW.select(station=staNames[ksta])[0].copy())
                        cc[-1].data = correlate( \
                            EW.select(station=staNames[ksta])[0], \
                            UD.select(station=staNames[ksta])[0], ncor)
                        cc[-1].stats['channel'] = 'CC'
                        cc[-1].stats['station'] = staNames[ksta]
                        if (draw_plot == True):
                            plt.plot(t, (2.0 * ksta + 1) + amp * cc[-1].data, \
                                'k-')
                # Stack cross correlations within the array and plot
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
                EW_UD.append(stack)
                if (draw_plot == True):
                    plt.plot(t, - 2.0 + amp_stack * stack.data, 'r-')
                    plt.xlim(0, Tmax)
                    plt.ylim(- 4.0, 2.0 * len(staNames) + 1.0)
                    plt.title('East / Vertical component', loc='right', \
                        fontsize=24)
                    plt.xlabel('Lag time (s)', fontsize=24)
                    plt.ylabel('Cross correlation', fontsize=24)
                    ax.set_yticklabels([])
                    ax.tick_params(labelsize=20)
                # NS - UD cross correlation
                if (draw_plot == True):
                    ax = plt.subplot(212)
                cc = Stream()
                for ksta in range(0, len(staNames)):
                    if (D.select(station=staNames[ksta], \
                        channel=chaNames[0]) and \
                        D.select(station=staNames[ksta], \
                        channel=chaNames[1]) and \
                        D.select(station=staNames[ksta], \
                        channel=chaNames[2])):
                        cc.append(NS.select(station=staNames[ksta])[0].copy())
                        cc[-1].data = correlate( \
                            NS.select(station=staNames[ksta])[0], \
                            UD.select(station=staNames[ksta])[0], ncor)
                        cc[-1].stats['channel'] = 'CC'
                        cc[-1].stats['station'] = staNames[ksta]
                        if (draw_plot == True):
                            plt.plot(t, (2.0 * ksta + 1) + amp * cc[-1].data, \
                                'k-')
                # Stack cross correlations within the array and plot
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
                NS_UD.append(stack)
                if (draw_plot == True):
                    plt.plot(t, - 2.0 + amp_stack * stack.data, 'r-')
                    plt.xlim(0, Tmax)
                    plt.ylim(- 4.0, 2.0 * len(staNames) + 1.0)
                    plt.title('North / Vertical component', loc='right', \
                        fontsize=24)
                    plt.xlabel('Lag time (s)', fontsize=24)
                    plt.ylabel('Cross correlation', fontsize=24)
                    ax.set_yticklabels([])
                    ax.tick_params(labelsize=20)
                    title = \
                        '{} on {:04d}/{:02d}/{:02d} at {:02d}:{:02d}:{:02d}'. \
                        format(arrayName, YY1, MM1, DD1, HH1, mm1, ss1)
                    plt.suptitle(title, fontsize=24)
                    filename = \
                        'cc/{}_{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}.eps'. \
                        format(arrayName, YY1, MM1, DD1, HH1, mm1, ss1)
                    plt.savefig(filename, format='eps')
                    ax.clear()
                    plt.close(1)
                # Keep values of tremor time
                Year.append(YY1)
                Month.append(MM1)
                Day.append(DD1)
                Hour.append(HH1)
                Minute.append(mm1)
                Second.append(ss1)

    # Save stacked cross correlations into file
    if nt > 0:
        namedir = 'cc/{}/{}_{:03d}_{:03d}/'.format(arrayName, arrayName, \
            int(x0), int(y0))
        if not os.path.exists(namedir):
            os.makedirs(namedir)
        filename = namedir + '{}_{:03d}_{:03d}_{}.pkl'.format(arrayName, \
            int(x0), int(y0), type_stack)
        pickle.dump([Year, Month, Day, Hour, Minute, Second, EW_UD, NS_UD], \
            open(filename, 'wb'))

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
#    lat0 = 48.0056818181818
#    lon0 = -123.084354545455

#    arrayName = 'BS'
#    staNames = ['BS01', 'BS02', 'BS03', 'BS04', 'BS05', 'BS06', 'BS11', 'BS20', 'BS21', 'BS22', 'BS23', 'BS24', 'BS25', 'BS26', 'BS27']
#    staCodes = 'BS01,BS02,BS03,BS04,BS05,BS06,BS11,BS20,BS21,BS22,BS23,BS24,BS25,BS26,BS27'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XU'
#    lat0 = 47.95728
#    lon0 = -122.92866

#    arrayName = 'CL'
#    staNames = ['CL01', 'CL02', 'CL03', 'CL04', 'CL05', 'CL06', 'CL07', 'CL08', 'CL09', 'CL10', 'CL11', 'CL12', 'CL13', 'CL14', 'CL15', 'CL16', 'CL17', 'CL18', 'CL19', 'CL20']
#    staCodes = 'CL01,CL02,CL03,CL04,CL05,CL06,CL07,CL08,CL09,CL10,CL11,CL12,CL13,CL14,CL15,CL16,CL17,CL18,CL19,CL20'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'
#    lat0 = 48.068735
#    lon0 = -122.969935

#    arrayName = 'DR'
#    staNames = ['DR01', 'DR02', 'DR03', 'DR04', 'DR05', 'DR06', 'DR07', 'DR08', 'DR09', 'DR10', 'DR12']
#    staCodes = 'DR01,DR02,DR03,DR04,DR05,DR06,DR07,DR08,DR09,DR10,DR12'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'
#    lat0 = 48.0059272727273
#    lon0 = -123.313118181818

#    arrayName = 'GC'
#    staNames = ['GC01', 'GC02', 'GC03', 'GC04', 'GC05', 'GC06', 'GC07', 'GC08', 'GC09', 'GC10', 'GC11', 'GC12', 'GC13', 'GC14']
#    staCodes = 'GC01,GC02,GC03,GC04,GC05,GC06,GC07,GC08,GC09,GC10,GC11,GC12,GC13,GC14'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'
#    lat0 = 47.9321857142857
#    lon0 = -123.045528571429

#    arrayName = 'LC'
#    staNames = ['LC01', 'LC02', 'LC03', 'LC04', 'LC05', 'LC06', 'LC07', 'LC08', 'LC09', 'LC10', 'LC11', 'LC12', 'LC13', 'LC14']
#    staCodes = 'LC01,LC02,LC03,LC04,LC05,LC06,LC07,LC08,LC09,LC10,LC11,LC12,LC13,LC14'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'
#    lat0 = 48.0554071428571
#    lon0 = -123.210035714286

#    arrayName = 'PA'
#    staNames = ['PA01', 'PA02', 'PA03', 'PA04', 'PA05', 'PA06', 'PA07', 'PA08', 'PA09', 'PA10', 'PA11', 'PA12', 'PA13']
#    staCodes = 'PA01,PA02,PA03,PA04,PA05,PA06,PA07,PA08,PA09,PA10,PA11,PA12,PA13'
#    chaNames = ['SHE', 'SHN', 'SHZ']
#    chans = 'SHE,SHN,SHZ'
#    network = 'XG'
#    lat0 = 48.0549384615385
#    lon0 = -123.464415384615

    arrayName = 'TB'
    staNames = ['TB01', 'TB02', 'TB03', 'TB04', 'TB05', 'TB06', 'TB07', 'TB08', 'TB09', 'TB10', 'TB11', 'TB12', 'TB13', 'TB14']
    staCodes = 'TB01,TB02,TB03,TB04,TB05,TB06,TB07,TB08,TB09,TB10,TB11,TB12,TB13,TB14'
    chaNames = ['SHE', 'SHN', 'SHZ']
    chans = 'SHE,SHN,SHZ'
    network = 'XG'
    lat0 = 47.9730357142857
    lon0 = -123.138492857143

    ds = 5.0
    TDUR = 10.0
    filt = (2, 8)
    w = 2.0
    ncor = 400
    Tmax = 15.0
    draw_plot = False
    nattempts = 10
    waittime = 10

    for i in range(2, 3):
        x0 = i * 5.0
        for j in [-1, 1, 2, 3, 4, 5]: #range(1, 3):
            y0 = j * 5.0

            # Linear stack
            type_stack = 'lin'
            amp = 3.0
            amp_stack = 10.0
            stack_ccorr_tremor(arrayName, staNames, staCodes, chaNames, \
                chans, network, lat0, lon0, ds, x0, y0, TDUR, filt, \
                type_stack, w, ncor, Tmax, amp, amp_stack, draw_plot, \
                client, nattempts, waittime)
        
            # Power stack
            type_stack = 'pow'
            amp = 3.0
            amp_stack = 2.0
            stack_ccorr_tremor(arrayName, staNames, staCodes, chaNames, \
                chans, network, lat0, lon0, ds, x0, y0, TDUR, filt, \
                type_stack, w, ncor, Tmax, amp, amp_stack, draw_plot, \
                client, nattempts, waittime)

            # Phase-weighted stack
            type_stack = 'PWS'
            amp = 3.0
            amp_stack = 30.0
            stack_ccorr_tremor(arrayName, staNames, staCodes, chaNames, \
                chans, network, lat0, lon0, ds, x0, y0, TDUR, filt, \
                type_stack, w, ncor, Tmax, amp, amp_stack, draw_plot, \
                client, nattempts, waittime)
