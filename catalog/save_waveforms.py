"""
This module contains a function to download every one-minute time window
where there is an LFE recorded and save the waveforms into a file
"""

import obspy
from obspy import read
from obspy import read_inventory
from obspy import UTCDateTime
from obspy.core.stream import Stream

import numpy as np
import pickle
import os

from fractions import Fraction

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
    elif (station == 'GASB' or station == 'WDC' or station == 'YBH'):
        network = 'BK'
        channels = 'BHE,BHN,BHZ'
    elif (station == 'KCPB' or station == 'KHBB' or station == 'KRMB' \
       or station == 'KSXB'):
        network = 'NC'
        channels = 'HHE,HHN,HHZ'
    else:
        network = 'NC'
        channels = 'EHZ'
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

def save_waveform(filename, TDUR, filt, dt):
    """
    This function download the waveform for each template and save it

    Input:
        type filename = string
        filename = Name of the template
        type TDUR = float
        TDUR = Time to add before and after the time window for tapering
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type dt = float
        dt = Time step for resampling
    Output:
        None
    """
    # Get the names of the stations for which we save the waveform
    stations = pickle.load(open('stations/' + filename + '.pkl', 'rb'))

    # Get the time of LFE detections
    LFEtime = np.loadtxt('../data/Plourde_2015/detections/' + filename + \
        '_detect5_cull.txt', \
        dtype={'names': ('unknown', 'day', 'hour', 'second', 'threshold'), \
             'formats': (np.float, '|S6', np.int, np.float, np.float)}, \
        skiprows=2)

    # Create list of streams
    EWall = []
    NSall = []
    UDall = []

    # Loop over stations
    for station in stations:
        print(station)
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
                    if (len(D.select(component='E').slice(Tori, \
                        Tori + 60.0)) > 0):
                        EW.append(D.select(component='E').slice(Tori, \
                            Tori + 60.0)[0])
                    if (len(D.select(component='N').slice(Tori, \
                        Tori + 60.0)) > 0):
                        NS.append(D.select(component='N').slice(Tori, \
                            Tori + 60.0)[0])
                    if (len(D.select(component='Z').slice(Tori, \
                        Tori + 60.0)) > 0):
                        UD.append(D.select(component='Z').slice(Tori, \
                            Tori + 60.0)[0])
            else:
                print('Failed at downloading data')
        # Add stream to list of streams
        EWall.append(EW)
        NSall.append(NS)
        UDall.append(UD)

    # Save in output file
    filename = 'data/' + filename + '.pkl'
    pickle.dump([stations, EWall, NSall, UDall], open(filename, 'wb'))

if __name__ == '__main__':

    # Set the parameters
    filename = '080326.07.004'
    TDUR = 10.0
    filt = (1.5, 9.0)
    dt = 0.025

    save_waveform(filename, TDUR, filt, dt)
