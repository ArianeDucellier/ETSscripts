"""
This module contains functions to find LFEs with the
temporary stations or with the permanent stations
using the templates from Plourde et al. (2015)
"""
import obspy
from obspy import UTCDateTime
from obspy.core.trace import Trace
from obspy.signal.cross_correlation import correlate

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from math import floor

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

def find_LFEs_FAME(filename, tbegin, tend, TDUR, filt, \
        freq0, draw=False, use_threshold=False, threshold=0.0075):
    """
    Find LFEs with the temporary stations from FAME
    using the templates from Plourde et al. (2015)

    Input:
        type filename = string
        filename = Name of the template
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
        type draw = boolean
        draw = Do we draw a figure of the cross-correlation?
        type use_threshold = boolean
        use_threshold = Do we give the cross-correlation threshold as input?
        type threshold = float
        threshold = Cross-correlation vlaue must be higher than that
    Output:
        None
    """
    # Get the names of the stations which have a waveform for this LFE family
    file = open('../data/Plourde_2015/detections/' + filename + \
        '_detect5_cull.txt')
    first_line = file.readline().strip()
    staNames = first_line.split()
    second_line = file.readline().strip()
    if (use_threshold == False):
        threshold = float(second_line.split()[1])
    file.close()

    # Get the network, channels, and location of the stations
    staloc = pd.read_csv('../data/Plourde_2015/station_locations.txt', \
        sep=r'\s{1,}', header=None)
    staloc.columns = ['station', 'network', 'channels', 'location', \
        'server', 'latitude', 'longitude']

    # Create directory to store the LFEs times
    namedir = 'LFEs/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    nchannel = 0

    for station in staNames:
        # Open file containing template
        data = pickle.load(open('templates/' + filename + \
            '/' + station + '.pkl', 'rb'))
        EW0 = data[0]
        NS0 = data[1]
        UD0 = data[2]
        dt = EW0.stats.delta
        nt = EW0.stats.npts
        duration = (nt - 1) * dt

        # Get station metadata for downloading
        for ir in range(0, len(staloc)):
            if (station == staloc['station'][ir]):
                network = staloc['network'][ir]
                channels = staloc['channels'][ir]
                location = staloc['location'][ir]
                server = staloc['server'][ir]

        # Download data from server
        t1 = UTCDateTime(year=tbegin[0], month=tbegin[1], \
            day=tbegin[2], hour=tbegin[3], minute=tbegin[4], \
            second=tbegin[5])
        t2 = UTCDateTime(year=tend[0], month=tend[1], \
            day=tend[2], hour=tend[3], minute=tend[4], \
            second=tend[5])
        Tstart = t1 - TDUR
        Tend = t2 + duration + TDUR
        delta = t2 + duration - t1
        ndata = int(delta / dt) + 1

        # First case: we can get the data from IRIS
        if (server == 'IRIS'):
            D = get_from_IRIS(station, network, channels, location, \
                Tstart, Tend, filt, dt)
        # Second case: we get the data from NCEDC
        elif (server == 'NCEDC'):
            D = get_from_NCEDC(station, network, channels, location, \
                Tstart, Tend, filt, dt)
        else:
            raise ValueError('You can only download data from IRIS and NCEDC')
        # Compute cross-correlation
        if (type(D) == obspy.core.stream.Stream):
            if (channels == 'EH1,EH2,EHZ'):
                if (len(D.select(channel='EH1').slice(t1, \
                    t2 + duration)) == 1):
                    EW = D.select(channel='EH1').slice(t1, t2 + duration)[0]
                else:
                    EW = Trace()
                if (len(D.select(channel='EH2').slice(t1, \
                    t2 + duration)) == 1):
                    NS = D.select(channel='EH2').slice(t1, t2 + duration)[0]
                else:
                    NS = Trace()
                if (len(D.select(channel='EHZ').slice(t1, \
                    t2 + duration)) == 1):
                    UD = D.select(channel='EHZ').slice(t1, t2 + duration)[0]
                else:
                    UD = Trace()
            else:
                if (len(D.select(component='E').slice(t1, \
                    t2 + duration)) == 1):
                    EW = D.select(component='E').slice(t1, t2 + duration)[0]
                else:
                    EW = Trace()
                if (len(D.select(component='N').slice(t1, \
                    t2 + duration)) == 1):
                    NS = D.select(component='N').slice(t1, t2 + duration)[0]
                else:
                    NS = Trace()
                if (len(D.select(component='Z').slice(t1, \
                    t2 + duration)) == 1):
                    UD = D.select(component='Z').slice(t1, t2 + duration)[0]
                else:
                    UD = Trace()
            for channel in ['EW', 'NS', 'UD']:
                if (channel == 'EW'):
                    template = EW0
                    data = EW
                elif (channel == 'NS'):
                    template = NS0
                    data = NS
                else:
                    template = UD0
                    data = UD
                if (len(data.data) == ndata):
                    cctemp = correlate(template, data, \
                        int((len(data) - len(template)) / 2))
                    if (nchannel > 0):
                        cc = np.vstack((cc, cctemp))
                    else:
                        cc = cctemp
                    nchannel = nchannel + 1
    
    # Compute average cross-correlation across channels
    meancc = np.flipud(np.mean(cc, axis=0))
    if (use_threshold == False):
        MAD = np.median(np.abs(meancc - np.mean(meancc)))
        index = np.where(meancc >= threshold * MAD)
    else:
        index = np.where(meancc >= threshold)
    times = np.arange(0.0, np.shape(meancc)[0] * dt, dt)

    # Get LFE times
    if np.shape(index)[1] > 0:
        (time, cc) = clean_LFEs(index, times, meancc, dt, freq0)

        # Add to pandas dataframe and save
        namefile = 'LFEs/' + filename + '/catalog.pkl'
        if os.path.exists(namefile):
            df = pickle.load(open(namefile, 'rb'))
        else:
            df = pd.DataFrame(columns=['year', 'month', 'day', 'hour', \
                'minute', 'second', 'cc', 'nchannel'])
        i0 = len(df.index)
        for i in range(0, len(time)):
            timeLFE = t1 + time[i]
            df.loc[i0 + i] = [int(timeLFE.year), int(timeLFE.month), \
                int(timeLFE.day), int(timeLFE.hour), int(timeLFE.minute), \
                timeLFE.second + timeLFE.microsecond / 1000000.0, cc[i], \
                nchannel]
        df = df.astype(dtype={'year':'int32', 'month':'int32', \
            'day':'int32', 'hour':'int32', 'minute':'int32', \
            'second':'float', 'cc':'float', 'nchannel':'int32'})
        pickle.dump(df, open(namefile, 'wb'))

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
        if (use_threshold == False):
            plt.axhline(threshold * MAD, linewidth=2, color='red', \
                label = '{:6.2f} * MAD'.format(threshold))
        else:
            plt.axhline(threshold, linewidth=2, color='red', \
                label = 'Threshold = {:8.4f}'.format(threshold))
        plt.xlim(0.0, (np.shape(meancc)[0] - 1) * dt)
        plt.xlabel('Time (s)', fontsize=24)
        plt.ylabel('Cross-correlation', fontsize=24)
        plt.title('Average cross-correlation across stations', \
            fontsize=30)
        plt.legend(loc=2, fontsize=24)
        plt.savefig('LFEs/' + filename + '/' + \
            '{:04d}{:02d}{:02d}_{:02d}{:02d}{:02d}'.format( \
            tbegin[0], tbegin[1], tbegin[2], tbegin[3], tbegin[4], \
            tbegin[5]) + '.png', format='png')
        plt.close(1)

def find_LFEs_permanent(filename, stations, tbegin, tend,
    TDUR, filt, dt):
    """
    """
    for station in stations:
        # Open file containing template
        data = pickle.load(open('data/raw/' + filename \
            + '/' + station + '.pkl', 'rb'))
        channels = data[0]
        templates = data[1]
        # Download data from server
        t1 = UTCDateTime(year=tbegin[0], month=tbegin[1], \
            day=tbegin[2], hour=tbegin[3], minute=tbegin[4], \
            second=tbegin[5])
        t2 = UTCDateTime(year=tend[0], month=tend[1], \
            day=tend[2], hour=tend[3], minute=tend[4], \
            second=tend[5])
        Tstart = t1 - TDUR
        Tend = t2 + TDUR
        # First case: we can get the data from IRIS
        if (station[0 : 2] == 'ME' or station == 'B039'):
            D = get_from_IRIS(station, Tstart, Tend, filt, dt)
        # Second case: we get the data from NCEDC
        else:
            D = get_from_NCEDC(station, Tstart, Tend, filt, dt)
            print(D)
        for channel, template in zip(channels, templates):
            if (channel == 'EW'):
                if (station == 'B039'):
                    data = D.select(channel='EH1').slice(t1, t2)[0]
                else:
                    data = D.select(component='E').slice(t1, t2)[0]
            elif (channel == 'NS'):
                if (station == 'B039'):
                    data = D.select(channel='EH2').slice(t1, t2)[0]
                else:
                    data = D.select(component='N').slice(t1, t2)[0]
            else:
                if (station == 'B039'):
                    data = D.select(channel='EHZ').slice(t1, t2)[0]
                else:
                    data = D.select(component='Z').slice(t1, t2)[0]
            cctemp = correlate(template, data, int((len(data) - len(template)) / 2))
            print(np.shape(data), np.shape(template), np.shape(cctemp))
  
if __name__ == '__main__':

    # Set the parameters
    filename = '080421.14.048'
    TDUR = 10.0
    filt = (1.5, 9.0)
    freq0 = 1.0
    draw = False
    use_threshold = True
    threshold = 0.01

    # For FAME network (known LFEs)    
#    year = 2008
#    month = 4
#    for day in range(21, 29):
#        for hour in range(0, 24):
#            tbegin = (year, month, day, hour, 0, 0)
#            if (hour == 23):
#                if (day == 31):
#                    tend = (year, month + 1, 1, 0, 0, 0)
#                else:
#                    tend = (year, month, day + 1, 0, 0, 0)
#            else:
#                tend = (year, month, day, hour + 1, 0, 0)
#
#            find_LFEs_FAME(filename, tbegin, tend, TDUR, filt, \
#                freq0, draw, use_threshold, threshold)

    # For FAME network (unknown LFEs)    
    year = 2009
    month = 1
    for day in range(21, 32):
        for hour in range(1, 24):
            tbegin = (year, month, day, hour, 0, 0)
            if (hour == 23):
                if (day == 31):
                    if (month == 12):
                        tend = (year + 1, 1, 1, 0, 0, 0)
                    else:
                        tend = (year, month + 1, 1, 0, 0, 0)
                else:
                    tend = (year, month, day + 1, 0, 0, 0)
            else:
                tend = (year, month, day, hour + 1, 0, 0)

            find_LFEs_FAME(filename, tbegin, tend, TDUR, filt, \
                freq0, draw, use_threshold, threshold)
