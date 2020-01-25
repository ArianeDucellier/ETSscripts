"""
This module contains a function to download every one-minute time window
where there is an LFE recorded, stack the signal over all the LFEs, and
compare the template with the one from Plourde et al. (2015)
"""

import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from scipy.io import loadmat

from get_data import get_from_IRIS, get_from_NCEDC
from stacking import linstack

def get_waveform(filename, TDUR, filt, nattempts, waittime, method='RMS'):
    """
    This function computes the waveforms for a given template and compare
    them to the waveforms from Plourde et al. (2015)

    Input:
        type filename = string
        filename = Name of the template
        type TDUR = float
        TDUR = Time to add before and after the time window for tapering
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type nattempts = integer
        nattempts = Number of times we try to download data
        type waittime = positive float
        waittime = Type to wait between two attempts at downloading
        type method = string
        method = Normalization method for linear stack (RMS or Max)
    Output:
        None
    """
    # Get the names of the stations which have a waveform for this LFE family
    file = open('../data/Plourde_2015/detections/' + filename + \
        '_detect5_cull.txt')
    first_line = file.readline().strip()
    staNames = first_line.split()
    file.close()

    # Get the time of LFE detections
    LFEtime = np.loadtxt('../data/Plourde_2015/detections/' + filename + \
        '_detect5_cull.txt', \
        dtype={'names': ('unknown', 'day', 'hour', 'second', 'threshold'), \
             'formats': (np.float, '|S6', np.int, np.float, np.float)}, \
        skiprows=2)

    # Get the waveforms from the catalog of Plourde et al. (2015)
    data = loadmat('../data/Plourde_2015/waveforms/' + filename + '.mat')
    ndt = data['ndt'][0][0]
    ordlst = data['ordlst']
    uk = data['uk']
    ns = len(ordlst)

    # Get the network, channels, and location of the stations
    staloc = pd.read_csv('../data/Plourde_2015/station_locations.txt', \
        sep=r'\s{1,}', header=None)
    staloc.columns = ['station', 'network', 'channels', 'location', \
        'server', 'latitude', 'longitude']

    # Create directory to store the waveforms
    namedir = 'waveforms/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    # File to write error messages
    errorfile = 'error/' + filename + '.txt'

    # Loop over stations
    for station in staNames:
        # Create streams
        EW = Stream()
        NS = Stream()
        UD = Stream()
        # Get station metadata for downloading
        for ir in range(0, len(staloc)):
            if (station == staloc['station'][ir]):
                network = staloc['network'][ir]
                channels = staloc['channels'][ir]
                location = staloc['location'][ir]
                server = staloc['server'][ir]
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
            if (server == 'IRIS'):
                (D, orientation) = get_from_IRIS(station, network, channels, \
                    location, Tstart, Tend, filt, ndt, nattempts, waittime, \
                    errorfile)
            # Second case: we get the data from NCEDC
            elif (server == 'NCEDC'):
                (D, orientation) = get_from_NCEDC(station, network, channels, \
                    location, Tstart, Tend, filt, ndt, nattempts, waittime, \
                    errorfile)
            else:
                raise ValueError( \
                    'You can only download data from IRIS and NCEDC')
            if (type(D) == obspy.core.stream.Stream):
                # Add to stream
                if (channels == 'EH1,EH2,EHZ'):
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
            EWstack = linstack([EW], normalize=True, method=method) 
            NSstack = linstack([NS], normalize=True, method=method)
            UDstack = linstack([UD], normalize=True, method=method)
            # First figure
            # Comparison with the waveforms from Plourde et al. (2015)
            plt.figure(1, figsize=(20, 15))
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
            norm = np.max(np.abs(EWstack[0].data))
            plt.plot(t, EWstack[0].data / norm, 'r', label='Stack')
            t0 = ndt * np.arange(0, np.shape(uk)[1])
            norm = np.max(np.abs(uk[ns + index, :]))
            plt.plot(t0, uk[ns + index, :] / norm, 'k', label='Waveform')
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
            norm = np.max(np.abs(NSstack[0].data))
            plt.plot(t, NSstack[0].data / norm, 'r', label='Stack')
            t0 = ndt * np.arange(0, np.shape(uk)[1])
            norm = np.max(np.abs(uk[index, :]))
            plt.plot(t0, uk[index, :] / norm, 'k', label='Waveform')
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
            norm = np.max(np.abs(UDstack[0].data))
            plt.plot(t, UDstack[0].data / norm, 'r', label='Stack')
            t0 = ndt * np.arange(0, np.shape(uk)[1])
            norm = np.max(np.abs(uk[2 * ns + index, :]))
            plt.plot(t0, uk[2 * ns + index, :] / norm, 'k', label='Waveform')
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
    TDUR = 10.0
    filt = (1.5, 9.0)
    nattempts = 10
    waittime = 10.0
    method = 'RMS'

    LFEloc = np.loadtxt('../data/Plourde_2015/templates_list.txt', \
        dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
        'eZ', 'nb'), \
             'formats': ('S13', 'S3', np.float, np.float, np.float, \
        np.float, np.float, np.int)}, \
        skiprows=1)
    for ie in range(0, len(LFEloc)):
        filename = LFEloc[ie][0].decode('utf-8')
        get_waveform(filename, TDUR, filt, nattempts, waittime, method)
