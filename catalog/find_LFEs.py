"""
This module contains functions to find LFEs with the
temporary stations or with the permanent stations
using the templates from Plourde et al. (2015)
"""
import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.signal.cross_correlation import correlate

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from datetime import timedelta
from math import ceil, floor

import correlate
from get_data import get_from_IRIS, get_from_NCEDC

def clean_LFEs(index, times, meancc, dt, freq0):
    """
    This function takes all times where the
    cross-correlation is higher than a threshold
    and groups those that belongs to the same LFE

    Input:
        type index = 1D numpy array
        index = Indices where cc is higher than threshold
        type times = 1D numpy array
        times = Times where cc is computed
        type meancc = 1D numpy array
        meancc = Average cc across all channels
        type dt = float
        dt = Time step of the seismograms
        type freq0 = float
        freq0 = Maximum frequency rate of LFE occurrence
    Output:
        type time = 1D numpy array
        time = Timing of LFEs
        type cc = 1D numpy array
        cc = Maximum cc during LFE
    """
    # Initializations
    maxdiff = int(floor(1.0 / (dt * freq0)))
    list_index = [[index[0][0]]]
    list_times = [[times[index[0][0]]]]
    list_cc = [[meancc[index[0][0]]]]
    # Group LFE times that are close to each other
    for i in range(1, np.shape(index)[1]):
        if (index[0][i] - list_index[-1][-1] <= maxdiff):
            list_index[-1].append(index[0][i])
            list_times[-1].append(times[index[0][i]])
            list_cc[-1].append(meancc[index[0][i]])
        else:
            list_index.append([index[0][i]])
            list_times.append([times[index[0][i]]])
            list_cc.append([meancc[index[0][i]]])
    # Number of LFEs identified
    N = len(list_index)
    time = np.zeros(N)
    cc = np.zeros(N)
    # Timing of LFE is where cc is maximum
    for i in range(0, N):
        maxcc =  np.amax(np.array(list_cc[i]))
        imax = np.argmax(np.array(list_cc[i]))
        cc[i] = maxcc
        time[i] = list_times[i][imax]
    return(time, cc)

def find_LFEs(filename, stations, tbegin, tend, TDUR, filt, \
        freq0, nattempts, waittime, draw=False, type_threshold='MAD', threshold=0.0075):
    """
    Find LFEs with the temporary stations from FAME
    using the templates from Plourde et al. (2015)

    Input:
        type filename = string
        filename = Name of the template
        type stations = list of strings
        stations = name of the stations used for the matched-filter algorithm
        type tebgin = tuplet of 6 integers
        tbegin = Time when we begin looking for LFEs
        type tend = tuplet of 6 integers
        tend = Time we stop looking for LFEs
        type TDUR = float
        TDUR = Time to add before and after the time window for tapering
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type freq0 = float
        freq0 = Maximum frequency rate of LFE occurrence
        type nattempts = integer
        nattempts = Number of times we try to download data
        type waittime = positive float
        waittime = Type to wait between two attempts at downloading
        type draw = boolean
        draw = Do we draw a figure of the cross-correlation?
        type type_threshold = string
        type_threshold = 'MAD' or 'Threshold'
        type threshold = float
        threshold = Cross correlation value must be higher than that
    Output:
        None
    """

    # Get the network, channels, and location of the stations
    staloc = pd.read_csv('../data/Plourde_2015/station_locations.txt', \
        sep=r'\s{1,}', header=None)
    staloc.columns = ['station', 'network', 'channels', 'location', \
        'server', 'latitude', 'longitude']

    # Create directory to store the LFEs times
    namedir = 'LFEs/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    # File to write error messages
    errorfile = 'error/' + filename + '.txt'

    # Read the templates
    templates = Stream()
    for station in stations:
        data = pickle.load(open('templates/' + filename + \
            '/' + station + '.pkl', 'rb'))
        EW = data[0]
        NS = data[1]
        UD = data[2]
        EW.stats.station = station
        NS.stats.station = station
        UD.stats.station = station
        EW.stats.channel = 'E'
        NS.stats.channel = 'N'
        UD.stats.channel = 'Z'
        templates.append(EW)
        templates.append(NS)
        templates.append(UD)

    # Begin and end time of analysis
    t1 = UTCDateTime(year=tbegin[0], month=tbegin[1], \
        day=tbegin[2], hour=tbegin[3], minute=tbegin[4], \
        second=tbegin[5])
    t2 = UTCDateTime(year=tend[0], month=tend[1], \
        day=tend[2], hour=tend[3], minute=tend[4], \
        second=tend[5])

    # Read the data
    data = Stream()
    for station in stations:
        # Get station metadata for downloading
        for ir in range(0, len(staloc)):
            if (station == staloc['station'][ir]):
                network = staloc['network'][ir]
                channels = staloc['channels'][ir]
                location = staloc['location'][ir]
                server = staloc['server'][ir]

        # Duration of template
        template = templates.select(station=station, component='E')[0]
        dt = template.stats.delta
        nt = template.stats.npts
        duration = (nt - 1) * dt   
        Tstart = t1 - TDUR
        Tend = t2 + duration + TDUR
        delta = t2 + duration - t1
        ndata = int(delta / dt) + 1

        # First case: we can get the data from IRIS
        if (server == 'IRIS'):
            D = get_from_IRIS(station, network, channels, location, \
                Tstart, Tend, filt, dt, nattempts, waittime, errorfile)
        # Second case: we get the data from NCEDC
        elif (server == 'NCEDC'):
            D = get_from_NCEDC(station, network, channels, location, \
                Tstart, Tend, filt, dt, nattempts, waittime, errorfile)
        else:
            raise ValueError('You can only download data from IRIS and NCEDC')

        # Append data to stream
        if (type(D) == obspy.core.stream.Stream):
            # East-West channel
            if (channels == 'EH1,EH2,EHZ'):
                if (len(D.select(channel='EH1').slice(t1, \
                    t2 + duration)) == 1):
                    EW = D.select(channel='EH1').slice(t1, t2 + duration)[0]
            else:
                if (len(D.select(component='E').slice(t1, \
                    t2 + duration)) == 1):
                    EW = D.select(component='E').slice(t1, t2 + duration)[0]
            if (len(EW.data) == ndata):
                EW.stats.station = station
                EW.stats.channel = 'E'
                data.append(EW)
            # North-South channel
            if (channels == 'EH1,EH2,EHZ'):
                if (len(D.select(channel='EH2').slice(t1, \
                    t2 + duration)) == 1):
                    NS = D.select(channel='EH2').slice(t1, t2 + duration)[0]
            else:
                if (len(D.select(component='N').slice(t1, \
                    t2 + duration)) == 1):
                    NS = D.select(component='N').slice(t1, t2 + duration)[0]
            if (len(NS.data) == ndata):
                NS.stats.station = station
                NS.stats.channel = 'N'
                data.append(NS)
            # Vertical channel
            if (channels == 'EH1,EH2,EHZ'):
                if (len(D.select(channel='EHZ').slice(t1, \
                    t2 + duration)) == 1):
                    UD = D.select(channel='EHZ').slice(t1, t2 + duration)[0]
            else:
                if (len(D.select(component='Z').slice(t1, \
                    t2 + duration)) == 1):
                    UD = D.select(component='Z').slice(t1, t2 + duration)[0]
            if (len(UD.data) == ndata):
                UD.stats.station = station
                UD.stats.channel = 'Z'
                data.append(UD)

    # Number of hours of data to analyze
    nhour = int(ceil((t2 - t1) / 3600.0))

    # Create dataframe to store LFE times
    df = pd.DataFrame(columns=['year', 'month', 'day', 'hour', \
        'minute', 'second', 'cc', 'nchannel'])

    # Loop on hours of data
    for hour in range(0, nhour):
        nchannel = 0
        Tstart = t1 + hour * 3600.0
        Tend = t1 + (hour + 1) * 3600.0

        # Loop on channels
        for channel in range(0, len(data)):
            # Cut the data
            subdata = data[channel]
            if (Tend > subdata.stats.endtime):
                tend = subdata.stats.endtime
            subdata = subdata.slice(Tstart, Tend)
            # Get the template
            station = subdata.stats.station
            component = subdata.stats.channel
            template = templates.select(station=station, component=component)[0]
            dt = template.stats.delta
            # Cross correlation
            cctemp = correlate.optimized(template, subdata)
            if (nchannel > 0):
                cc = np.vstack((cc, cctemp))
            else:
                cc = cctemp
            nchannel = nchannel + 1
    
        # Compute average cross-correlation across channels
        meancc = np.mean(cc, axis=0)
        if (type_threshold == 'MAD'):
            MAD = np.median(np.abs(meancc - np.mean(meancc)))
            index = np.where(meancc >= threshold * MAD)
        elif (type_threshold == 'Threshold'):
            index = np.where(meancc >= threshold)
        else:
            raise ValueError('Type of threshold must be MAD or Threshold')
        times = np.arange(0.0, np.shape(meancc)[0] * dt, dt)

        # Get LFE times
        if np.shape(index)[1] > 0:
            (time, cc) = clean_LFEs(index, times, meancc, dt, freq0)

            # Add LFE times to dataframe
            i0 = len(df.index)
            for i in range(0, len(time)):
                timeLFE = Tstart + time[i]
                df.loc[i0 + i] = [int(timeLFE.year), int(timeLFE.month), \
                    int(timeLFE.day), int(timeLFE.hour), int(timeLFE.minute), \
                    timeLFE.second + timeLFE.microsecond / 1000000.0, cc[i], \
                    nchannel]

        # Draw figure
        if (draw == True):
            params = {'xtick.labelsize':16,
                      'ytick.labelsize':16}
            pylab.rcParams.update(params) 
            plt.figure(1, figsize=(20, 8))
            if np.shape(index)[1] > 0:
                for i in range(0, len(time)):
                    plt.axvline(time[i], linewidth=2, color='grey')
            plt.plot(np.arange(0.0, np.shape(meancc)[0] * dt, \
                dt), meancc, color='black')
            if (type_threshold == 'MAD'):
                plt.axhline(threshold * MAD, linewidth=2, color='red', \
                    label = '{:6.2f} * MAD'.format(threshold))
            elif (type_threshold == 'Threshold'):
                plt.axhline(threshold, linewidth=2, color='red', \
                    label = 'Threshold = {:8.4f}'.format(threshold))
            else:
                raise ValueError('Type of threshold must be MAD or Threshold')
            plt.xlim(0.0, (np.shape(meancc)[0] - 1) * dt)
            plt.xlabel('Time (s)', fontsize=24)
            plt.ylabel('Cross-correlation', fontsize=24)
            plt.title('Average cross-correlation across stations', \
                fontsize=30)
            plt.legend(loc=2, fontsize=24)
            plt.savefig('LFEs/' + filename + '/' + \
                '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format( \
                Tstart.year, Tstart.month, Tstart.day, Tstart.hour, Tstart.minute, \
                Tstart.second) + '.png', format='png')
            plt.close(1)

    # Add to pandas dataframe and save
    namefile = 'LFEs/' + filename + '/catalog.pkl'
    if os.path.exists(namefile):
        df_all = pickle.load(open(namefile, 'rb'))
        df_all = pd.concat([df_all, df], ignore_index=True)
    else:
        df_all = df    
    df_all = df_all.astype(dtype={'year':'int32', 'month':'int32', \
        'day':'int32', 'hour':'int32', 'minute':'int32', \
        'second':'float', 'cc':'float', 'nchannel':'int32'})
    pickle.dump(df_all, open(namefile, 'wb'))

if __name__ == '__main__':

    # Set the parameters
    filename = '080422.12.039'
    TDUR = 10.0
    filt = (1.5, 9.0)
    freq0 = 1.0
    nattempts = 10
    waittime = 10.0
    draw = False
    type_threshold = 'MAD'
    
    # Get the names of the stations which have a waveform for this LFE family
    file = open('../data/Plourde_2015/detections/' + filename + \
        '_detect5_cull.txt')
    first_line = file.readline().strip()
    stations = first_line.split()
    second_line = file.readline().strip()
    if (type_threshold == 'MAD'):
        threshold = float(second_line.split()[1])
    file.close()

    # For FAME network (known LFEs)    
    year = 2008
    month = 4
    for day in range(21, 31):
        tbegin = (year, month, day, 0, 0, 0)
        if day == 30:
            tend = (year, month + 1, 1, 0, 0, 0)
        else:
            tend = (year, month, day + 1, 0, 0, 0)

        find_LFEs(filename, stations, tbegin, tend, TDUR, filt, \
            freq0, nattempts, waittime, draw, type_threshold, threshold)

    # For FAME network (unknown LFEs)    
#    year = 2009
#    month = 1
#    for day in range(21, 32):
#        for hour in range(1, 24):
#            tbegin = (year, month, day, hour, 0, 0)
#            if (hour == 23):
#                if (day == 31):
#                    if (month == 12):
#                        tend = (year + 1, 1, 1, 0, 0, 0)
#                    else:
#                        tend = (year, month + 1, 1, 0, 0, 0)
#                else:
#                    tend = (year, month, day + 1, 0, 0, 0)
#            else:
#                tend = (year, month, day, hour + 1, 0, 0)
#
#            find_LFEs_FAME(filename, tbegin, tend, TDUR, filt, \
#                freq0, draw, use_threshold, threshold)
