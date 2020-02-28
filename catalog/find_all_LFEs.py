"""
This module contains functions to find LFEs with the
temporary stations or with the permanent stations
using the templates from Plourde et al. (2015)
"""
import obspy
from obspy import read
from obspy import read_inventory
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from obspy.signal.cross_correlation import correlate

import numpy as np
import os
import pandas as pd
import pickle

from datetime import timedelta
from math import ceil, cos, floor, pi

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

def fill_data(D, orientation, station, channels, reference):
    """
    Return the data that must be cross correlated with the template

    Input:
        type D = obspy Stream
        D = Data downloaded
        type orientation = list of dictionnaries
        orientation = azimuth, dip for 3 channels (for data)
        type station = string
        station = Name of station
        type channels = string
        channels = Names of channels
        type reference = list of dictionnaries
        reference = azimuth, dip for 3 channels (for template)
    Output:
        type data = list of obspy Stream
        data = Data to be analyzed with correct azimuth
    """
    # East-West channel
    EW = Stream()
    if (channels == 'EH1,EH2,EHZ'):
        if (len(D.select(channel='EH1')) > 0):
            EW = D.select(channel='EH1')
    else:
        if (len(D.select(component='E')) > 0):
            EW = D.select(component='E')
    # North-South channel
    NS = Stream()
    if (channels == 'EH1,EH2,EHZ'):
         if (len(D.select(channel='EH2')) > 0):
            NS = D.select(channel='EH2')
    else:
         if (len(D.select(component='N')) > 0):
            NS = D.select(component='N')
    # Vertical channel
    UD = Stream()
    if (channels == 'EH1,EH2,EHZ'):
        if (len(D.select(channel='EHZ')) > 0):
             UD = D.select(channel='EHZ')
    else:
         if (len(D.select(component='Z')) > 0):
            UD = D.select(component='Z')
    # Rotation of the data
    data = []
    if ((len(EW) > 0) and (len(NS) > 0) and (len(EW) == len(NS))):
        # Orientation of the data
        dE = orientation[0]['azimuth'] * pi / 180.0
        dN = orientation[1]['azimuth'] * pi / 180.0
        # Orientation of the template
        tE = reference[0]['azimuth'] * pi / 180.0
        tN = reference[1]['azimuth'] * pi / 180.0   
        EWrot = Stream()
        NSrot = Stream()
        for i in range(0, len(EW)):
            if (len(EW[i].data) == len(NS[i].data)):
                EWrot0 = EW[i].copy()
                NSrot0 = NS[i].copy()
                EWrot0.data = cos(dE - tE) * EW[i].data + \
                              cos(dN - tE) * NS[i].data
                NSrot0.data = cos(dE - tN) * EW[i].data + \
                              cos(dN - tN) * NS[i].data
                EWrot0.stats.station = station
                EWrot0.stats.channel = 'E'
                NSrot0.stats.station = station
                NSrot0.stats.channel = 'N'
                EWrot.append(EWrot0)
                NSrot.append(NSrot0)
        data.append(EWrot)
        data.append(NSrot)
    if (len(UD) > 0):
        for i in range(0, len(UD)):
            UD[i].stats.station = station
            UD[i].stats.channel = 'Z'
        data.append(UD)
    return(data)

def find_LFEs(family_file, station_file, template_dir, tbegin, tend, \
    TDUR, duration, filt, freq0, dt, nattempts, waittime, type_threshold='MAD', \
    threshold=0.0075):
    """
    
    """
    # Get the network, channels, and location of the stations
    staloc = pd.read_csv(station_file, sep=r'\s{1,}', header=None, engine='python')
    staloc.columns = ['station', 'network', 'channels', 'location', \
        'server', 'latitude', 'longitude', 'time_on', 'time_off']

    # Begin and end time of analysis
    t1 = UTCDateTime(year=tbegin[0], month=tbegin[1], \
        day=tbegin[2], hour=tbegin[3], minute=tbegin[4], \
        second=tbegin[5])
    t2 = UTCDateTime(year=tend[0], month=tend[1], \
        day=tend[2], hour=tend[3], minute=tend[4], \
        second=tend[5])
    
    # Number of hours of data to analyze
    nhour = int(ceil((t2 - t1) / 3600.0))

    # Begin and end time of downloading
    Tstart = t1 - TDUR
    Tend = t2 + duration + TDUR

    # Temporary directory to store the data
    namedir = 'tmp'
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    # Download the data from the stations
    for ir in range(0, len(staloc)):
        station = staloc['station'][ir]
        network = staloc['network'][ir]
        channels = staloc['channels'][ir]
        location = staloc['location'][ir]
        server = staloc['server'][ir]
        time_on = staloc['time_on'][ir]
        time_off = staloc['time_off'][ir]

        # File to write error messages
        namedir = 'error'
        if not os.path.exists(namedir):
            os.makedirs(namedir)
        errorfile = 'error/' + station + '.txt'

        # Check whether there are data for this period of time
        year_on = int(time_on[0:4])
        month_on = int(time_on[5:7])
        day_on = int(time_on[8:10])
        year_off = int(time_off[0:4])
        month_off = int(time_off[5:7])
        day_off = int(time_off[8:10])
        if ((Tstart > UTCDateTime(year=year_on, month=month_on, day=day_on)) \
           and (Tend < UTCDateTime(year=year_off, month=month_off, day=day_off))):

            # First case: we can get the data from IRIS
            if (server == 'IRIS'):
                (D, orientation) = get_from_IRIS(station, network, channels, \
                    location, Tstart, Tend, filt, dt, nattempts, waittime, \
                    errorfile)
            # Second case: we get the data from NCEDC
            elif (server == 'NCEDC'):
                (D, orientation) = get_from_NCEDC(station, network, channels, \
                    location, Tstart, Tend, filt, dt, nattempts, waittime, \
                    errorfile)
            else:
                raise ValueError('You can only download data from IRIS and NCEDC')

            # Store the data into temporary files
            if (type(D) == obspy.core.stream.Stream):
                D.write('tmp/' + station + '.mseed', format='MSEED')
                namefile = 'tmp/' + station + '.pkl'
                pickle.dump(orientation, open(namefile, 'wb'))

    # Loop on families
    families = pd.read_csv(family_file, sep=r'\s{1,}', header=None, engine='python')
    families.columns = ['family', 'stations']
    for i in range(0, len(families)):

        # Create directory to store the LFEs times
        namedir = 'LFEs/' + families['family'].iloc[i]
        if not os.path.exists(namedir):
             os.makedirs(namedir)

        # File to write error messages
        namedir = 'error'
        if not os.path.exists(namedir):
            os.makedirs(namedir)
            errorfile = 'error/' + families['family'].iloc[i] + '.txt'

        # Create dataframe to store LFE times
        df = pd.DataFrame(columns=['year', 'month', 'day', 'hour', \
        'minute', 'second', 'cc', 'nchannel'])

        # Read the templates
        stations = families['stations'].iloc[i].split(',')
        templates = Stream()
        for station in stations:
            data = pickle.load(open(template_dir + '/' + families['family'].iloc[i] + \
            '/' + station + '.pkl', 'rb'))
            if (len(data) == 3):
                EW = data[0]
                NS = data[1]
                UD = data[2]
                EW.stats.station = station
                NS.stats.station = station
                EW.stats.channel = 'E'
                NS.stats.channel = 'N'
                templates.append(EW)
                templates.append(NS)
            else:
                UD = data[0]
            UD.stats.station = station
            UD.stats.channel = 'Z'
            templates.append(UD)                       

        # Loop on hours of data
        for hour in range(0, nhour):
            nchannel = 0
            Tstart = t1 + hour * 3600.0
            Tend = t1 + (hour + 1) * 3600.0 + duration
            delta = Tend - Tstart
            ndata = int(delta / dt) + 1

            # Get the data
            data = []
            for station in stations:
                try:
                    D = read('tmp/' + station + '.mseed')
                    D = D.slice(Tstart, Tend)
                    namefile = 'tmp/' + station + '.pkl'
                    orientation = pickle.load(open(namefile, 'rb'))

                    # Get station metadata for reading response file
                    for ir in range(0, len(staloc)):
                        if (station == staloc['station'][ir]):
                            network = staloc['network'][ir]
                            channels = staloc['channels'][ir]
                            location = staloc['location'][ir]
                            server = staloc['server'][ir]

                    # Orientation of template
                    # Date chosen: April 1st 2008
                    mychannels = channels.split(',')
                    mylocation = location
                    if (mylocation == '--'):
                        mylocation = ''
                    response = '../data/response/' + network + '_' + station + '.xml'
                    inventory = read_inventory(response, format='STATIONXML')
                    reference = []
                    for channel in mychannels:
                        angle = inventory.get_orientation(network + '.' + \
                            station + '.' + mylocation + '.' + channel, \
                            UTCDateTime(2008, 4, 1, 0, 0, 0))
                        reference.append(angle)

                    # Append data to stream
                    if (type(D) == obspy.core.stream.Stream):
                        stationdata = fill_data(D, orientation, station, channels, reference)
                        if (len(stationdata) > 0):
                            for stream in stationdata:
                                data.append(stream)
                except:
                    message = 'No data available for station {} '.format( \
                        station) + 'at time {}/{}/{} - {}:{}:{}\n'.format( \
                        Tstart.year, Tstart.month, Tstart.day, Tstart.hour, \
                        Tstart.minute, Tstart.second)

            # Loop on channels
            for channel in range(0, len(data)):
                subdata = data[channel]
                # Check whether we have a complete one-hour-long recording
                if (len(subdata) == 1):
                    if (len(subdata[0].data) == ndata):
                        # Get the template
                        station = subdata[0].stats.station
                        component = subdata[0].stats.channel
                        template = templates.select(station=station, \
                            component=component)[0]
                        # Cross correlation
                        cctemp = correlate.optimized(template, subdata[0])
                        if (nchannel > 0):
                            cc = np.vstack((cc, cctemp))
                        else:
                            cc = cctemp
                        nchannel = nchannel + 1
    
            if (nchannel > 0):   
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
                    for j in range(0, len(time)):
                        timeLFE = Tstart + time[j]
                        df.loc[i0 + j] = [int(timeLFE.year), int(timeLFE.month), \
                            int(timeLFE.day), int(timeLFE.hour), \
                            int(timeLFE.minute), timeLFE.second + \
                            timeLFE.microsecond / 1000000.0, cc[j], nchannel]

        # Add to pandas dataframe and save
        namefile = 'LFEs/' + families['family'].iloc[i] + '/catalog.pkl'
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
    family_file = '../data/Ducellier/families_temporary.txt'
    station_file = '../data/Ducellier/stations_temporary.txt'
    template_dir = 'templates'
    tbegin = (2008, 4, 1, 0, 0, 0)
    tend = (2008, 4, 2, 0, 0, 0)
    TDUR = 10.0
    duration = 60.0
    filt = (1.5, 9.0)
    freq0 = 1.0
    dt = 0.05
    nattempts = 10
    waittime = 10.0
    type_threshold = 'MAD'
    threshold = 8.0
    
    find_LFEs(family_file, station_file, template_dir, tbegin, tend, \
        TDUR, duration, filt, freq0, dt, nattempts, waittime, type_threshold, \
        threshold)
