"""
This module contains a function to download every one-minute time window
where there is an LFE recorded, stack the signal over all the LFEs, cross
correlate each window with the stack, sort the LFEs and keep only the best
"""

import obspy
import obspy.clients.fdsn.client as fdsn
from obspy import read
from obspy import read_inventory
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.signal.cross_correlation import correlate

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from fractions import Fraction
from math import cos, pi, sin, sqrt

from stacking import linstack

def get_from_IRIS(station, Tstart, Tend, filt, dt):
    """
    Function to get the waveform from IRIS for a given station and LFE

    Input:
        type station = string
        station = Name of the station
        type Tstart = obspy UTCDateTime
        Tstart = Time when to begin downloading
        type Tend = obspy UTCDateTime
        Tend = Time when to end downloading
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type dt = float
        dt = Time step for resampling
    Output:
        type D = obspy Stream
        D = Stream with data detrended, tapered, instrument response
        deconvolved, filtered, and resampled
    """
    # Create client
    fdsn_client = fdsn.Client('IRIS')
    # Download data
    try:
        D = fdsn_client.get_waveforms(network='XQ', station=station, \
            location='01', channel='BHE,BHN,BHZ', starttime=Tstart, \
            endtime=Tend, attach_response=True)
    except:
        message = 'Could not download data for station {} '.format(station) + \
            'at time {}/{}/{} - {}:{}:{}'.format(Tstart.year, Tstart.month, \
            Tstart.day, Tstart.hour, Tstart.minute, Tstart.second)
        print(message)
        return(0)
    else:
        # Detrend data
        D.detrend(type='linear')
        # Taper first and last 5 s of data
        D.taper(type='hann', max_percentage=None, max_length=5.0)
        # Remove instrument response
        D.remove_response(output='VEL', \
            pre_filt=(0.2, 0.5, 10.0, 15.0), water_level=80.0)
        D.filter('bandpass', freqmin=filt[0], freqmax=filt[1], \
            zerophase=True)
        freq = D[0].stats.sampling_rate
        ratio = Fraction(int(freq), int(1.0 / dt))
        D.interpolate(ratio.denominator * freq, method='lanczos', a=10)
        D.decimate(ratio.numerator, no_filter=True)
        return(D)

def get_from_NCEDC(station, Tstart, Tend, filt, dt):
    """
    Function to get the waveform from NCEDC for a given station and LFE

    Input:
        type station = string
        station = Name of the station
        type Tstart = obspy UTCDateTime
        Tstart = Time when to begin downloading
        type Tend = obspy UTCDateTime
        Tend = Time when to end downloading
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type dt = float
        dt = Time step for resampling
    Output:
        type D = obspy Stream
        D = Stream with data detrended, tapered, instrument response
        deconvolved, filtered, and resampled
    """
    # Define network and channels
    if (station == 'B039'):
        network = 'PB'
        channels = 'EH1,EH2,EHZ'
    elif (station == 'WDC' or station == 'YBH'):
        network = 'BK'
        channels = 'BHE,BHN,BHZ'
    else:
        network = 'NC'
        channels = 'HHE,HHN,HHZ'
    # Write waveform request
    file = open('waveform.request', 'w')
    message = '{} {} -- {} '.format(network, station, channels) + \
        '{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d} '.format(Tstart.year, \
        Tstart.month, Tstart.day, Tstart.hour, Tstart.minute, \
        Tstart.second) + \
        '{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}\n'.format(Tend.year, \
        Tend.month, Tend.day, Tend.hour, Tend.minute, Tend.second)
    file.write(message)
    file.close()
    # Send waveform request
    request = 'curl --data-binary @waveform.request -o station.miniseed ' + \
         'http://service.ncedc.org/fdsnws/dataselect/1/query'
    try:
        os.system(request)
        D = read('station.miniseed')
    except:
        message = 'Could not download data for station {} '.format(station) + \
            'at time {}/{}/{} - {}:{}:{}'.format(Tstart.year, Tstart.month, \
            Tstart.day, Tstart.hour, Tstart.minute, Tstart.second)
        print(message)
        return(0)
    else:
        # Detrend data
        D.detrend(type='linear')
        # Taper first and last 5 s of data
        D.taper(type='hann', max_percentage=None, max_length=5.0)
        # Remove instrument response
        filename = '../data/response/' + network + '_' + station + '.xml'
        inventory = read_inventory(filename, format='STATIONXML')
        D.attach_response(inventory)
        D.remove_response(output='VEL', \
            pre_filt=(0.2, 0.5, 10.0, 15.0), water_level=80.0)
        D.filter('bandpass', freqmin=filt[0], freqmax=filt[1], \
            zerophase=True)
        freq = D[0].stats.sampling_rate
        ratio = Fraction(int(freq), int(1.0 / dt))
        D.interpolate(ratio.denominator * freq, method='lanczos', a=10)
        D.decimate(ratio.numerator, no_filter=True)
        return(D)

def compute_templates(filename, TDUR, filt, ratios, dt, ncor, window,
        method='RMS'):
    """
    This function computes the waveform for each template, cross correlate
    them with the stack, and keep only the best to get the final template
    that will be used to find LFEs

    Input:
        type filename = string
        filename = Name of the template
        type TDUR = float
        TDUR = Time to add before and after the time window for tapering
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type ratios = list of floats
        ratios = Percentage of LFEs to be kept for the final template
        type dt = float
        dt = Time step for resampling
        type ncor = integer
        ncor = Number of points for the cross correlation
        type window = float
        window = Length of the window to do the cross correlation
        type method = string
        method = Normalization method for linear stack (RMS or Max)
    Output:
        None
    """
    # To transform latitude and longitude into kilometers
    a = 6378.136
    e = 0.006694470
    lat0 = 41.0
    lon0 = -123.0
    dx = (pi / 180.0) * a * cos(lat0 * pi / 180.0) / sqrt(1.0 - e * e * \
        sin(lat0 * pi / 180.0) * sin(lat0 * pi / 180.0))
    dy = (3.6 * pi / 648.0) * a * (1.0 - e * e) / ((1.0 - e * e * sin(lat0 * \
        pi / 180.0) * sin(lat0 * pi / 180.0)) ** 1.5)

    # Get the names of the stations which have a waveform for this LFE family
    file = open('../data/Plourde_2015/detections/' + filename + \
        '_detect5_cull.txt')
    first_line = file.readline().strip()
    staNames = first_line.split()
    file.close()

    # Get the locations of the stations
    staloc = np.loadtxt('../data/Plourde_2015/station_locations.txt', \
        dtype={'names': ('name', 'lat', 'lon'), \
             'formats': ('|S5', np.float, np.float)})

    # Get the time of LFE detections
    LFEtime = np.loadtxt('../data/Plourde_2015/detections/' + filename + \
        '_detect5_cull.txt', \
        dtype={'names': ('unknown', 'day', 'hour', 'second', 'threshold'), \
             'formats': (np.float, '|S6', np.int, np.float, np.float)}, \
        skiprows=2)

    # Get the location of the source of the LFE
    LFEloc = np.loadtxt('../data/Plourde_2015/templates_list.txt', \
        dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
        'eZ', 'nb'), \
             'formats': ('S13', 'S3', np.float, np.float, np.float, \
        np.float, np.float, np.int)}, \
        skiprows=1)
    for ie in range(0, len(LFEloc)):
        if (filename == LFEloc[ie][0].decode('utf-8')):
            lats = LFEloc[ie][2]
            lons = LFEloc[ie][3]
            xs = dx * (lons - lon0)
            ys = dy * (lats - lat0)

    # Create directory to store the waveforms
    namedir = 'templates/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    # Read origin time and station slowness files
    origintime = pickle.load(open('timearrival/origintime.pkl', 'rb'))
    slowness = pickle.load(open('timearrival/slowness.pkl', 'rb'))
    
    # Loop over stations
    for station in staNames:
        # Compute source-receiver distance
        for ir in range(0, len(staloc)):
            if (station == staloc[ir][0].decode('utf-8')):
                latr = staloc[ir][1]
                lonr = staloc[ir][2]
                xr = dx * (lonr - lon0)
                yr = dy * (latr - lat0)
                distance = sqrt((xr - xs) ** 2.0 + (yr - ys) ** 2.0)
        # Create streams
        EW = Stream()
        NS = Stream()
        UD = Stream()
        # Loop on LFEs
        for i in range(0, np.shape(LFEtime)[0]):
            YMD = LFEtime[i][1]
            myYear = 2000 + int(YMD[0 : 2])
            myMonth = int(YMD[2 : 4])
            myDay = int(YMD[4 : 6])
            myHour = LFEtime[i][2] - 1
            myMinute = int(LFEtime[i][3] / 60.0)
            mySecond = int(LFEtime[i][3] - 60.0 * myMinute)
            myMicrosecond = int(1000000.0 * \
                (LFEtime[i][3] - 60.0 * myMinute - mySecond))
            Tori = UTCDateTime(year=myYear, month=myMonth, day=myDay, \
                hour=myHour, minute=myMinute, second=mySecond, \
                microsecond=myMicrosecond)
            Tstart = Tori - TDUR
            Tend = Tori + 60.0 + TDUR
            # First case: we can get the data from IRIS
            if (station[0 : 2] == 'ME'):
                D = get_from_IRIS(station, Tstart, Tend, filt, dt)
            # Second case: we get the data from NCEDC
            else:
                D = get_from_NCEDC(station, Tstart, Tend, filt, dt)
            if (type(D) == obspy.core.stream.Stream):
                # Add to stream
                if (station == 'B039'):
                    EW.append(D.select(channel='EH1').slice(Tori, \
                        Tori + 60.0)[0])
                    NS.append(D.select(channel='EH2').slice(Tori, \
                        Tori + 60.0)[0])
                    UD.append(D.select(channel='EHZ').slice(Tori, \
                        Tori + 60.0)[0])
                else:
                    EW.append(D.select(component='E').slice(Tori, \
                        Tori + 60.0)[0])
                    NS.append(D.select(component='N').slice(Tori, \
                        Tori + 60.0)[0])
                    UD.append(D.select(component='Z').slice(Tori, \
                        Tori + 60.0)[0])
            else:
                print('Failed at downloading data')
        # Stack
        if (len(EW) > 0):
            # Stack waveforms
            EWstack = linstack([EW], normalize=True, method=method) 
            NSstack = linstack([NS], normalize=True, method=method)
            UDstack = linstack([UD], normalize=True, method=method)
            # Get time arrival
            arrivaltime = origintime[filename] + slowness[station] * distance
            Tmin = arrivaltime - window/ 2.0
            Tmax = arrivaltime + window / 2.0
            ibegin = int(Tmin / EWstack[0].stats.delta)
            iend = int(Tmax / EWstack[0].stats.delta) + 1
            # Cross correlation
            maxCC = np.zeros(len(EW))
            for i in range(0, len(EW)):
                ccEW = correlate(EWstack[0].data[ibegin : iend], \
                    EW[i].data[ibegin : iend], ncor)
                ccNS = correlate(NSstack[0].data[ibegin : iend], \
                    NS[i].data[ibegin : iend], ncor)
                ccUD = correlate(UDstack[0].data[ibegin : iend], \
                    UD[i].data[ibegin : iend], ncor)
                maxCC[i] = np.max(ccEW) + np.max(ccNS) + np.max(ccUD)
            # Sort cross correlations
            index = np.flip(np.argsort(maxCC), axis=0)
            EWbest = Stream()
            NSbest = Stream()
            UDbest = Stream()
            # Compute stack of best LFEs
            for j in range(0, len(ratios)):
                nLFE = int(ratios[j] * len(EW) / 100.0)
                EWselect = Stream()
                NSselect = Stream()
                UDselect = Stream()
                for i in range(0, nLFE):
                    EWselect.append(EW[index[i]])
                    NSselect.append(NS[index[i]])
                    UDselect.append(UD[index[i]])
                # Stack best LFEs
                EWbest.append(linstack([EWselect], normalize=True, \
                    method=method)[0])
                NSbest.append(linstack([NSselect], normalize=True, \
                    method=method)[0])
                UDbest.append(linstack([UDselect], normalize=True, \
                    method=method)[0])
            # Plot figure
            plt.figure(1, figsize=(20, 15))
            colors = cm.rainbow(np.linspace(0, 1, len(ratios)))
            # East - West component
            ax1 = plt.subplot(311)
            dt = EWstack[0].stats.delta
            nt = EWstack[0].stats.npts
            t = dt * np.arange(0, nt)
            for j in range(0, len(ratios)):
                if (method == 'RMS'):
                    norm = EWbest[j].data / np.sqrt(np.mean(np.square( \
                        EWbest[j].data)))
                elif (method == 'MAD'):
                    norm = EWbest[j].data / np.median(np.abs(EWbest[j].data - \
                        np.median(EWbest[j].data)))
                else:
                    raise ValueError('Method must be RMS or MAD')
                norm = np.nan_to_num(norm)
                plt.plot(t, norm, color = colors[j], \
                    label = str(int(ratios[j])) + '%')
            if (method == 'RMS'):
                norm = EWstack[0].data / np.sqrt(np.mean(np.square( \
                    EWstack[0].data)))
            elif (method == 'MAD'):
                norm = EWstack[0].data / np.median(np.abs(EWstack[0].data - \
                    np.median(EWstack[0].data)))
            else:
                raise ValueError('Method must be RMS or MAD')
            norm = np.nan_to_num(norm)
            plt.plot(t, norm, 'k', label='All')
            plt.axvline(Tmin, linewidth=2, color='grey')
            plt.axvline(Tmax, linewidth=2, color='grey')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('East - West component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.legend(loc=1)
            # North - South component
            ax2 = plt.subplot(312)
            dt = NSstack[0].stats.delta
            nt = NSstack[0].stats.npts
            t = dt * np.arange(0, nt)
            for j in range(0, len(ratios)):
                if (method == 'RMS'):
                    norm = NSbest[j].data / np.sqrt(np.mean(np.square( \
                        NSbest[j].data)))
                elif (method == 'MAD'):
                    norm = NSbest[j].data / np.median(np.abs(NSbest[j].data - \
                        np.median(NSbest[j].data)))
                else:
                    raise ValueError('Method must be RMS or MAD')
                norm = np.nan_to_num(norm)
                plt.plot(t, norm, color = colors[j], \
                    label = str(int(ratios[j])) + '%')
            if (method == 'RMS'):
                norm = NSstack[0].data / np.sqrt(np.mean(np.square( \
                    NSstack[0].data)))
            elif (method == 'MAD'):
                norm = NSstack[0].data / np.median(np.abs(NSstack[0].data - \
                    np.median(NSstack[0].data)))
            else:
                raise ValueError('Method must be RMS or MAD')
            norm = np.nan_to_num(norm)
            plt.plot(t, norm, 'k', label='All')
            plt.axvline(Tmin, linewidth=2, color='grey')
            plt.axvline(Tmax, linewidth=2, color='grey')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('North - South component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.legend(loc=1)
            # Vertical component
            ax3 = plt.subplot(313)
            dt = UDstack[0].stats.delta
            nt = UDstack[0].stats.npts
            t = dt * np.arange(0, nt)
            for j in range(0, len(ratios)):
                if (method == 'RMS'):
                    norm = UDbest[j].data / np.sqrt(np.mean(np.square( \
                        UDbest[j].data)))
                elif (method == 'MAD'):
                    norm = UDbest[j].data / np.median(np.abs(UDbest[j].data - \
                        np.median(UDbest[j].data)))
                else:
                    raise ValueError('Method must be RMS or MAD')
                norm = np.nan_to_num(norm)
                plt.plot(t, norm, color = colors[j], \
                    label = str(int(ratios[j])) + '%')
            if (method == 'RMS'):
                norm = UDstack[0].data / np.sqrt(np.mean(np.square( \
                    UDstack[0].data)))
            elif (method == 'MAD'):
                norm = UDstack[0].data / np.median(np.abs(UDstack[0].data - \
                    np.median(UDstack[0].data)))
            else:
                raise ValueError('Method must be RMS or MAD')
            norm = np.nan_to_num(norm)
            plt.plot(t, norm, 'k', label='All')
            plt.axvline(Tmin, linewidth=2, color='grey')
            plt.axvline(Tmax, linewidth=2, color='grey')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('Vertical component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.legend(loc=1)
            # End figure
            plt.suptitle(station, fontsize=24)
            plt.savefig(namedir + '/' + station + '.eps', format='eps')
            ax1.clear()
            ax2.clear()
            ax3.clear()
            plt.close(1)
            # Save stacks into file
            for j in range(0, len(ratios)):
                savename = namedir + '/' + station + '_' + \
                    str(int(ratios[j])) + '.pkl'
                pickle.dump([EWbest[j], NSbest[j], UDbest[j]], \
                    open(savename, 'wb'))

if __name__ == '__main__':

    # Set the parameters
    filename = '080408.08.007'
    TDUR = 10.0
    filt = (1.5, 9.0)
    ratios = [50.0, 60.0, 70.0, 80.0, 90.0]
    dt = 0.05
    ncor = 400
    window = 10.0
    method = 'RMS'

    compute_templates(filename, TDUR, filt, ratios, dt, ncor, window, method)
