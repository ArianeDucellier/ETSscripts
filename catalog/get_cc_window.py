"""
This module contains functions to find the time arrival of each template
waveform for each station
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
import pickle

from fractions import Fraction
from math import cos, pi, sin, sqrt
from sklearn import linear_model
from sklearn.metrics import r2_score

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

def get_cc_window(filename, TDUR, filt, dt, method='RMS', envelope=True):
    """
    This function finds the time arrival of each template waveform
    for each station

    Input:
        type filename = string
        filename = Name of the template
        type TDUR = float
        TDUR = Time to add before and after the time window for tapering
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type dt = float
        dt = Time step for resampling
        type method = string
        method = Normalization method for linear stack (RMS or Max)
        type envelope = boolean
        envelope = Do we compute the max CC on the signal or the envelope?
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

    # Initialize lists
    maxEW = []
    maxNS = []
    maxUD = []
    timeEW = []
    timeNS = []
    timeUD = []
    stations = []

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
            if (envelope == True):
                EWstack[0].data = obspy.signal.filter.envelope( \
                    EWstack[0].data)
                NSstack[0].data = obspy.signal.filter.envelope( \
                    NSstack[0].data)
                UDstack[0].data = obspy.signal.filter.envelope( \
                    UDstack[0].data)
            maxEW.append(np.max(np.abs(EWstack[0].data) / np.sqrt(np.mean( \
                np.square(EWstack[0].data)))))
            maxNS.append(np.max(np.abs(NSstack[0].data) / np.sqrt(np.mean( \
                np.square(NSstack[0].data)))))
            maxUD.append(np.max(np.abs(UDstack[0].data) / np.sqrt(np.mean( \
                np.square(UDstack[0].data)))))
            timeEW.append(np.argmax(np.abs(EWstack[0].data)) * \
                EWstack[0].stats.delta)
            timeNS.append(np.argmax(np.abs(NSstack[0].data)) * \
                NSstack[0].stats.delta)
            timeUD.append(np.argmax(np.abs(UDstack[0].data)) * \
                UDstack[0].stats.delta)
            stations.append(station)

    # Save time arrivals into file
    output = 'timearrival/' + filename + '.pkl'
    pickle.dump([stations, maxEW, maxNS, maxUD, timeEW, timeNS, timeUD], \
        open(output, 'wb'))

def get_timeLFE(filename):
    """
    This function finds the origin time of the LFE for a given template

    Input:
        type filename = string
        filename = Name of the template
    Output:
        type tori = float
        tori = Origin time of the LFE
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

    # Get the locations of the stations
    staloc = np.loadtxt('../data/Plourde_2015/station_locations.txt', \
        dtype={'names': ('name', 'lat', 'lon'), \
             'formats': ('|S5', np.float, np.float)})

    # Open time arrival files
    data = pickle.load(open('timearrival/' + filename +'.pkl', 'rb'))
    stations = data[0]
    maxEW = data[1]
    maxNS = data[2]
    maxUD = data[3]
    timeEW = data[4]
    timeNS = data[5]
    timeUD = data[6]
    
    # Compute source-receiver distances
    distance = []
    for i in range(0, len(stations)):
        for ir in range(0, len(staloc)):
            if (stations[i] == staloc[ir][0].decode('utf-8')):
                latr = staloc[ir][1]
                lonr = staloc[ir][2]
                xr = dx * (lonr - lon0)
                yr = dy * (latr - lat0)
                distance.append(sqrt((xr - xs) ** 2.0 + (yr - ys) ** 2.0))

    # Linear regression
    x = np.reshape(np.array(distance + distance + distance), \
        (3 * len(stations), 1))
    y = np.reshape(np.array(timeEW + timeNS + timeUD), \
        (3 * len(stations), 1))
    w = list(map(lambda x : pow(x, 3.0), maxEW)) + \
        list(map(lambda x : pow(x, 3.0), maxNS)) + \
        list(map(lambda x : pow(x, 3.0), maxUD))
    w = np.array(w)
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y, w)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    s = regr.coef_[0][0]
    tori = regr.intercept_[0]
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(x, y, 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Distance (km)', fontsize=24)
    plt.ylabel('Arrival time (s)', fontsize=24)
    plt.title('{} - R2 = {:4.2f} - slowness = {:4.3f} s/km'.format( \
        filename, R2, s), fontsize=24)
    plt.savefig('timearrival/' + filename + '.eps', format='eps')
    plt.close(1)
    return tori    

def get_time_station():
    """
    This function looks at the travel time from the LFE source to the station
    location for all the templates

    Input:
        None
    Output:
        type slowness = dictionary
        slowness = Slowness associated to each of the stations to compute
                   the arrival time
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

    # Get the locations of the sources of the LFEs
    LFEloc = np.loadtxt('../data/Plourde_2015/templates_list.txt', \
        dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
        'eZ', 'nb'), \
             'formats': ('S13', 'S3', np.float, np.float, np.float, \
        np.float, np.float, np.int)}, \
        skiprows=1)
    lats = np.zeros(len(LFEloc))
    lons = np.zeros(len(LFEloc))
    for ie in range(0, len(LFEloc)):
        lats[ie] = LFEloc[ie][2]
        lons[ie] = LFEloc[ie][3]
    xs = dx * (lons - lon0)
    ys = dy * (lats - lat0)

    # Get the locations of the stations
    staloc = np.loadtxt('../data/Plourde_2015/station_locations.txt', \
        dtype={'names': ('name', 'lat', 'lon'), \
             'formats': ('|S5', np.float, np.float)})

    # Get the origin time for each of the templates
    origintime = pickle.load(open('timearrival/origintime.pkl', 'rb'))

    slowness = {}
    # Loop on the stations
    for ir in range(0, len(staloc)):
        # Compute source-receiver distances
        distance = []
        maxEWlist = []
        maxNSlist = []
        maxUDlist = []
        timeEWlist = []
        timeNSlist = []
        timeUDlist = []
        # Loop on the templates
        for ie in range(0, len(LFEloc)):
            filename = LFEloc[ie][0].decode('utf-8')
            # Open time arrival files
            data = pickle.load(open('timearrival/' + filename +'.pkl', 'rb'))
            stations = data[0]
            maxEW = data[1]
            maxNS = data[2]
            maxUD = data[3]
            timeEW = data[4]
            timeNS = data[5]
            timeUD = data[6]
            # If the station was used for this template
            for i in range(0, len(stations)):
                if (stations[i] == staloc[ir][0].decode('utf-8')):
                    latr = staloc[ir][1]
                    lonr = staloc[ir][2]
                    xr = dx * (lonr - lon0)
                    yr = dy * (latr - lat0)
                    distance.append(sqrt((xr - xs[ie]) ** 2.0 + \
                                         (yr - ys[ie]) ** 2.0))
                    maxEWlist.append(maxEW[i])
                    maxNSlist.append(maxNS[i])
                    maxUDlist.append(maxUD[i])
                    timeEWlist.append(timeEW[i] - origintime[filename])
                    timeNSlist.append(timeNS[i] - origintime[filename])
                    timeUDlist.append(timeUD[i] - origintime[filename])
        # Linear regression
        if (len(distance) > 0):
            x = np.reshape(np.array(distance + distance + distance), \
                (3 * len(distance), 1))
            y = np.reshape(np.array(timeEWlist + timeNSlist + timeUDlist), \
                (3 * len(distance), 1))
            w = list(map(lambda x : pow(x, 3.0), maxEWlist)) + \
                list(map(lambda x : pow(x, 3.0), maxNSlist)) + \
                list(map(lambda x : pow(x, 3.0), maxUDlist))
            w = np.array(w)
            regr = linear_model.LinearRegression(fit_intercept=False)
            regr.fit(x, y, w)
            y_pred = regr.predict(x)
            R2 = r2_score(y, y_pred)
            s = regr.coef_[0][0]
                # Plot
            plt.figure(1, figsize=(10, 10))
            plt.plot(x, y, 'ko')
            plt.plot(x, y_pred, 'r-')
            plt.xlabel('Distance (km)', fontsize=24)
            plt.ylabel('Travel time (s)', fontsize=24)
            plt.title('{} - R2 = {:4.2f} - slowness = {:4.3f} s/km'.format( \
                staloc[ir][0].decode('utf-8'), R2, s), fontsize=24)
            plt.savefig('timearrival/' + staloc[ir][0].decode('utf-8') + \
                '.eps', format='eps')
            plt.close(1)
            slowness[staloc[ir][0].decode('utf-8')] = s
    return slowness
    
if __name__ == '__main__':

#    # Set the parameters
    TDUR = 10.0
    filt = (1.5, 9.0)
    dt = 0.05
    method = 'RMS'
    envelope = True

#    LFEloc = np.loadtxt('../data/Plourde_2015/templates_list.txt', \
#        dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
#        'eZ', 'nb'), \
#             'formats': ('S13', 'S3', np.float, np.float, np.float, \
#        np.float, np.float, np.int)}, \
#        skiprows=1)
#    for ie in range(47, len(LFEloc)):
#        filename = LFEloc[ie][0].decode('utf-8')
#        get_cc_window(filename, TDUR, filt, dt, method, envelope)

#    LFEloc = np.loadtxt('../data/Plourde_2015/templates_list.txt', \
#        dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
#        'eZ', 'nb'), \
#             'formats': ('S13', 'S3', np.float, np.float, np.float, \
#        np.float, np.float, np.int)}, \
#        skiprows=1)
#    origintime = {}
#    for ie in range(0, len(LFEloc)):
#        filename = LFEloc[ie][0].decode('utf-8')
#        tori = get_timeLFE(filename)
#        origintime[filename] = tori
#    # Save origin times into file
#    output = 'timearrival/origintime.pkl'
#    pickle.dump(origintime, open(output, 'wb'))

    slowness = get_time_station()
    # Save slownesses into file
    output = 'timearrival/slowness.pkl'
    pickle.dump(slowness, open(output, 'wb'))
