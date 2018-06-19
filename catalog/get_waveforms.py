"""
This module contains a function to download every one-minute time window
where there is an LFE recorded, stack the signal over all the LFEs, and
compare the waveform with the one from Plourde et al. (2015)
"""

import obspy
import obspy.clients.fdsn.client as fdsn
from obspy import read
from obspy import read_inventory
from obspy import UTCDateTime
from obspy.core.stream import Stream

import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.io import loadmat

from stacking import linstack

def get_from_IRIS(station, Tstart, Tend, filt):
    """
    Function to get the waveform from IRIS

    Input:
        type station = string
        station = Name of the station
        type Tstart = obspy UTCDateTime
        Tstart = Time when to begin downloading
        type Tend = obspy UTCDateTime
        Tend = Time when to end downloading
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
    Output:
        type D = obspy Stream
        D = Stream with data detrended, tapered, instrument response
        deconvolved, and filtered
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
            'at time {}/{}/{} - {}:{}:{}'.format(Tstart.year, Tstart.month,\
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
        return(D)

def get_from_NCEDC(station, Tstart, Tend, filt):
    """
    Function to get the waveform from NCEDC

    Input:
        type station = string
        station = Name of the station
        type Tstart = obspy UTCDateTime
        Tstart = Time when to begin downloading
        type Tend = obspy UTCDateTime
        Tend = Time when to end downloading
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
    Output:
        type D = obspy Stream
        D = Stream with data detrended, tapered, instrument response
        deconvolved, and filtered
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
            'at time {}/{}/{} - {}:{}:{}'.format(Tstart.year, Tstart.month,\
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
        return(D)

def get_waveform(filename, TDUR, filt):
    """
    This function computes the waveform for each template and compare it to
    the waveform from Plourde et al. (2015)

    Input:
        type filename = string
        filename = Name of the template
        type TDUR = float
        TDUR = Time to add before and after the time window for tapering
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
    Output:
        None
    """
    # Get the names of the stations which have a waveform for this LFE family
    file = open('../data/LFEcatalog/detections/' + filename + \
        '_detect5_cull.txt')
    first_line = file.readline().strip()
    staNames = first_line.split()
    file.close()

    # Get the time of LFE detections
    LFEtime = np.loadtxt('../data/LFEcatalog/detections/' + filename + \
        '_detect5_cull.txt', \
        dtype={'names': ('unknown', 'day', 'hour', 'second', 'threshold'), \
             'formats': (np.float, '|S6', np.int, np.float, np.float)}, \
        skiprows=2)

    # Get the waveforms from the catalog of Plourde et al. (2015)
    data = loadmat('../data/LFEcatalog/waveforms/' + filename + '.mat')
    ndt = ndt = data['ndt'][0][0]
    ordlst = data['ordlst']
    uk = data['uk']
    ns = len(ordlst)

    # Create directory to store the waveforms
    namedir = 'waveforms/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    # Loop over stations
    for station in staNames:
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
                D = get_from_IRIS(station, Tstart, Tend, filt)
            # Second case: we get the data from NCEDC
            else:
                D = get_from_NCEDC(station, Tstart, Tend, filt)
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
        # Stack and plot
        if (len(EW) > 0 and len(NS) > 0 and len(UD) > 0):
            # Stack waveforms
            EWstack = linstack([EW], normalize=True) 
            NSstack = linstack([NS], normalize=True)
            UDstack = linstack([UD], normalize=True)
            # First figure
            # Comparison with the waveforms from Plourde et al. (2015)
            plt.figure(1, figsize=(10, 15))
            station4 = station
            if (len(station4) < 4):
                for j in range(len(station), 4):
                    station4 = station4 + ' '
            index = np.argwhere(ordlst == station4)[0][0]
            # EW component
            ax1 = plt.subplot(311)
            dt = EWstack[0].stats.delta
            nt = EWstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, EWstack[0].data, 'r', label='Stack')
            t0 = ndt * np.arange(0, np.shape(uk)[1])
            plt.plot(t0, uk[ns + index, :], 'k', label='Waveform')
            plt.xlim(0.0, 60.0)
            plt.title('East component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Velocity (m/s)', fontsize=16)
            plt.legend(loc=1)
            # NS component
            ax2 = plt.subplot(312)
            dt = NSstack[0].stats.delta
            nt = NSstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, NSstack[0].data, 'r', label='Stack')
            t0 = ndt * np.arange(0, np.shape(uk)[1])
            plt.plot(t0, uk[index, :], 'k', label='Waveform')
            plt.xlim(0.0, 60.0)
            plt.title('North component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Velocity (m/s)', fontsize=16)
            plt.legend(loc=1)
            # UD component
            ax3 = plt.subplot(313)
            dt = UDstack[0].stats.delta
            nt = UDstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, UDstack[0].data, 'r', label='Stack')
            t0 = ndt * np.arange(0, np.shape(uk)[1])
            plt.plot(t0, uk[2 * ns + index, :], 'k', label='Waveform')
            plt.xlim(0.0, 60.0)
            plt.title('Vertical component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Velocity (m/s)', fontsize=16)
            plt.legend(loc=1)
            # End and save figure
            plt.suptitle(station, fontsize=24)
            plt.savefig(namedir + '/' + station + '_compare.eps', format='eps')
            ax1.clear()
            ax2.clear()
            ax3.clear()
            plt.close(1)
            # Second figure
            # Look at all the waveforms and the stack
            plt.figure(2, figsize=(30, 15))
            # EW component
            ax1 = plt.subplot(131)
            for i in range(0, len(EW)):
                dt = EW[i].stats.delta
                nt = EW[i].stats.npts
                t = dt * np.arange(0, nt)
                datanorm = EW[i].data / np.max(np.abs(EW[i].data))
                plt.plot(t, (2.0 * i + 1) + datanorm, 'k-')
            datanorm = EWstack[0].data / np.max(np.abs(EWstack[0].data))
            plt.plot(t, - 2.0 + datanorm, 'r-')
            plt.xlim(0.0, 60.0)
            plt.ylim(- 3.0, 2.0 * len(EW))
            plt.title('East component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.ylabel('Velocity (m/s)', fontsize=24)
            ax1.set_yticklabels([])
            ax1.tick_params(labelsize=20)
            # NS component
            ax2 = plt.subplot(132)
            for i in range(0, len(NS)):
                dt = NS[i].stats.delta
                nt = NS[i].stats.npts
                t = dt * np.arange(0, nt)
                datanorm = NS[i].data / np.max(np.abs(NS[i].data))
                plt.plot(t, (2.0 * i + 1) + datanorm, 'k-')
            datanorm = NSstack[0].data / np.max(np.abs(NSstack[0].data))
            plt.plot(t, - 2.0 + datanorm, 'r-')
            plt.xlim(0.0, 60.0)
            plt.ylim(- 3.0, 2.0 * len(NS))
            plt.title('North component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.ylabel('Velocity (m/s)', fontsize=24)
            ax2.set_yticklabels([])
            ax2.tick_params(labelsize=20)
            # UD component
            ax3 = plt.subplot(133)
            for i in range(0, len(UD)):
                dt = UD[i].stats.delta
                nt = UD[i].stats.npts
                t = dt * np.arange(0, nt)
                datanorm = UD[i].data / np.max(np.abs(UD[i].data))
                plt.plot(t, (2.0 * i + 1) + datanorm, 'k-')  
            datanorm = UDstack[0].data / np.max(np.abs(UDstack[0].data))
            plt.plot(t, - 2.0 + datanorm, 'r-')
            plt.xlim(0.0, 60.0)
            plt.ylim(- 3.0, 2.0 * len(UD))
            plt.title('Vertical component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.ylabel('Velocity (m/s)', fontsize=24)
            ax3.set_yticklabels([])
            ax3.tick_params(labelsize=20)
            # End and save figure
            plt.suptitle(station, fontsize=24)
            plt.savefig(namedir + '/' + station + '_stack.eps', format='eps')
            ax1.clear()
            ax2.clear()
            ax3.clear()
            plt.close(2)               

if __name__ == '__main__':

    # Set the parameters
    filename = '080326.07.004'
    TDUR = 10.0
    filt = (1.5, 9.0)

    get_waveform(filename, TDUR, filt)
