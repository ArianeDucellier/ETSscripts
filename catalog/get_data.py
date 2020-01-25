"""
This module contains functions to download seismic data
from the IRIS DMC or the NCEDC website
"""

import obspy
import obspy.clients.fdsn.client as fdsn
from obspy import read
from obspy import read_inventory
from obspy import UTCDateTime

import os
import time

from fractions import Fraction

def get_from_IRIS(station, network, channels, location, Tstart, Tend, \
    filt, dt, nattempts, waittime, errorfile):
    """
    Function to get the waveform from IRIS for a given station

    Input:
        type station = string
        station = Name of the station
        type network = string
        network = Name of network
        type channels = string
        channels = Names of channels
        type location = string
        location = Name of location
        type Tstart = obspy UTCDateTime
        Tstart = Time when to begin downloading
        type Tend = obspy UTCDateTime
        Tend = Time when to end downloading
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type dt = float
        dt = Time step for resampling
        type nattempts = integer
        nattempts = Number of times we try to download data
        type waittime = positive float
        waittime = Type to wait between two attempts at downloading
        type errorfile = string
        errorfile = Name of the file where we write error messages
    Output:
        type D = obspy Stream
        D = Stream with data detrended, tapered, instrument response
        deconvolved, filtered, and resampled
        type orientation = list of dictionnaries
        orientation = azimuth, dip for 3 channels
    """
    # Create client
    fdsn_client = fdsn.Client('IRIS')
    # Loop to try downloading several times
    success = False
    attempts = 0
    while attempts < nattempts and not success:
        try:
            # Get data from server
            D = fdsn_client.get_waveforms(network=network, station=station, \
                location=location, channel=channels, starttime=Tstart, \
                endtime=Tend, attach_response=True)
            # Detrend data
            D.detrend(type='linear')
            # Taper first and last 5 s of data
            D.taper(type='hann', max_percentage=None, max_length=5.0)
            # Remove instrument response
            D.remove_response(output='VEL', \
                pre_filt=(0.2, 0.5, 10.0, 15.0), water_level=80.0)
            # Filter
            D.filter('bandpass', freqmin=filt[0], freqmax=filt[1], \
                zerophase=True)
            # Resample
            freq = D[0].stats.sampling_rate
            ratio = Fraction(int(freq), int(1.0 / dt))
            D.interpolate(ratio.denominator * freq, method='lanczos', a=10)
            D.decimate(ratio.numerator, no_filter=True)
            # Get station orientation
            filename = '../data/response/' + network + '_' + station + '.xml'
            inventory = read_inventory(filename, format='STATIONXML')
            orientation = []
            for channel in range(0, len(D)):
                angle = inventory.get_orientation(D[channel].stats.network + \
                    '.' + D[channel].stats.station + '.' + \
                    D[channel].stats.location + '.' + \
                    D[channel].stats.channel, Tstart + D[channel].stats.delta \
                    * D[channel].stats.npts * 0.5)
                orientation.append(angle)
            success = True
            return(D, orientation)
        except:
            message = 'Could not download data for station {} '.format( \
                station) + 'at time {}/{}/{} - {}:{}:{}\n'.format( \
                Tstart.year, Tstart.month, Tstart.day, Tstart.hour, \
                Tstart.minute, Tstart.second)
            with open(errorfile, 'a') as file:
                file.write(message)
            attempts += 1
            time.sleep(waittime)
            if attempts == nattempts:
                with open(errorfile, 'a') as file:
                    file.write('Failed to download data after {} attempts\n'. \
                        format(nattempts))
                return((0, 0))

def get_from_NCEDC(station, network, channels, location, Tstart, Tend, \
    filt, dt, nattempts, waittime, errorfile):
    """
    Function to get the waveform from NCEDC for a given station

    Input:
        type station = string
        station = Name of the station
        type network = string
        network = Name of network
        type channels = string
        channels = Names of channels
        type location = string
        location = Name of location
        type Tstart = obspy UTCDateTime
        Tstart = Time when to begin downloading
        type Tend = obspy UTCDateTime
        Tend = Time when to end downloading
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type dt = float
        dt = Time step for resampling
        type nattempts = integer
        nattempts = Number of times we try to download data
        type waittime = positive float
        waittime = Type to wait between two attempts at downloading
        type errorfile = string
        errorfile = Name of the file where we write error messages
    Output:
        type D = obspy Stream
        D = Stream with data detrended, tapered, instrument response
        deconvolved, filtered, and resampled
        type orientation = list of dictionnaries
        orientation = azimuth, dip for 3 channels
    """
    # Write waveform request
    file = open('waveform.request', 'w')
    message = '{} {} {} {} '.format(network, station, location, channels) + \
        '{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d} '.format(Tstart.year, \
        Tstart.month, Tstart.day, Tstart.hour, Tstart.minute, \
        Tstart.second) + \
        '{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}\n'.format(Tend.year, \
        Tend.month, Tend.day, Tend.hour, Tend.minute, Tend.second)
    file.write(message)
    file.close()
    # Send waveform request
    request = 'curl -s --data-binary @waveform.request -o station.miniseed ' + \
         'http://service.ncedc.org/fdsnws/dataselect/1/query'
    # Loop to try downloading several times
    success = False
    attempts = 0
    while attempts < nattempts and not success:
        try:
            # Get data from server
            os.system(request)
            D = read('station.miniseed')
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
            # Filter
            D.filter('bandpass', freqmin=filt[0], freqmax=filt[1], \
                zerophase=True)
            # Resample
            freq = D[0].stats.sampling_rate
            ratio = Fraction(int(freq), int(1.0 / dt))
            D.interpolate(ratio.denominator * freq, method='lanczos', a=10)
            D.decimate(ratio.numerator, no_filter=True)
            # Get station orientation
            orientation = []
            for channel in range(0, len(D)):
                angle = inventory.get_orientation(D[channel].stats.network + \
                    '.' + D[channel].stats.station + '.' + \
                    D[channel].stats.location + '.' + \
                    D[channel].stats.channel, Tstart + D[channel].stats.delta \
                    * D[channel].stats.npts * 0.5)
                orientation.append(angle)
            success = True
            return(D, orientation)
        except:
            message = 'Could not download data for station {} '.format( \
                station) + 'at time {}/{}/{} - {}:{}:{}\n'.format( \
                Tstart.year, Tstart.month, Tstart.day, Tstart.hour, \
                Tstart.minute, Tstart.second)
            with open(errorfile, 'a') as file:
                file.write(message)
            attempts += 1
            time.sleep(waittime)
            if attempts == nattempts:
                with open(errorfile, 'a') as file:
                    file.write('Failed to download data after {} attempts\n'. \
                        format(nattempts))
                return((0, 0))
