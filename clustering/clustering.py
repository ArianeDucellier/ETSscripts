""" Module to try prediction strength
on recordings of LFEs """

import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream

import numpy as np

from get_data import get_from_IRIS, get_from_NCEDC

def get_dataset(stations, tbegin, tend, TDUR, filt, dt, nattempts, waittime, errorfile):
    """
    """
    # Get the network, channels, and location of the stations
    staloc = pd.read_csv('../station_locations.txt', \
        sep=r'\s{1,}', header=None)
    staloc.columns = ['station', 'network', 'channels', 'location', \
        'server', 'latitude', 'longitude']

    # File to write error messages
    errorfile = 'error/' + filename + '.txt'

    # Begin and end time of analysis
    t1 = UTCDateTime(year=tbegin[0], month=tbegin[1], \
        day=tbegin[2], hour=tbegin[3], minute=tbegin[4], \
        second=tbegin[5])
    t2 = UTCDateTime(year=tend[0], month=tend[1], \
        day=tend[2], hour=tend[3], minute=tend[4], \
        second=tend[5])
    Tstart = t1 - TDUR
    Tend = t2 + TDUR

    # Read the data
    data = Stream()

    # Loop on stations
    for station in stations:
        # Get station metadata for downloading
        for ir in range(0, len(staloc)):
            if (station == staloc['station'][ir]):
                network = staloc['network'][ir]
                channels = staloc['channels'][ir]
                location = staloc['location'][ir]
                server = staloc['server'][ir]

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
            for trace in D:
                data.append(trace)

    # Output
    return(data)

def select_peaks(data, threshold):
    """
    """
    mean_data = np.zeros(data[0].stats.npts)
    for trace in data:
        RMS = np.sqrt(np.mean(np.square(trace.data)))
        trace.data = trace.data / RMS
        mean_data = mean_data + trace.data
    index = np.where(mean_data >= threshold)
    return(index)

def divide_data(index):
    """
    """
    N = np.shape(index)[0]
    selection = np.random.uniform(0.0, 1.0, N)
    train = np.where(selection <= 0.5)
    test = np.where(selection > 0.5)
    index_tr = index[train]
    index_te = index[test]
    return(index_tr, index_te)

def dist_obs(index1, index2, duration):
    """
    """
    # Cut data
    Tstart
    Tend
    data1 = subdata.slice(Tstart, Tend)
    data2 = subdata.slice(Tstart, Tend)

    # Loop on channels
        for channel in range(0, len(data)):
            # Cut the data
            subdata = data[channel]
            subdata = subdata.slice(Tstart, Tend)
            # Check whether we have a complete one-hour-long recording
            if (len(subdata) == 1):
                if (len(subdata[0].data) == ndata):
                    # Get the template
                    station = subdata[0].stats.station
                    component = subdata[0].stats.channel
                    template = templates.select(station=station, component=component)[0]
                    # Cross correlation
                    cctemp = correlate.optimized(template, subdata[0])
                    if (nchannel > 0):
                        cc = np.vstack((cc, cctemp))
                    else:
                        cc = cctemp
                    nchannel = nchannel + 1
    