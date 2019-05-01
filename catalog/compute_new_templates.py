"""
This module contains a function to download every one-minute time window
where there is an LFE recorded, stack the signal over all the LFEs, cross
correlate each window with the stack, sort the LFEs and keep only the best
"""

import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.signal.cross_correlation import correlate

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from math import cos, floor, pi, sin, sqrt

from get_data import get_from_IRIS, get_from_NCEDC
from stacking import linstack

def compute_new_templates(filename, catalog, threshold, TDUR, filt, dt, method):
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
        type window = boolean
        window = Do we do the cross correlation on the whole seismogram
                 or a selected time window?
        type winlength = float
        winlength = Length of the window to do the cross correlation
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
    namefile = 'LFEs/' + filename + '/' + catalog
    LFEtime = pickle.load(open(namefile, 'rb'))
    best = LFEtime['cc'] > threshold
    LFEtime = LFEtime[best]

    # Get the network, channels, and location of the stations
    staloc = pd.read_csv('../data/Plourde_2015/station_locations.txt', \
        sep=r'\s{1,}', header=None)
    staloc.columns = ['station', 'network', 'channels', 'location', \
        'server', 'latitude', 'longitude']

    # Create directory to store the waveforms
    namedir = 'new_templates/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

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
        for i in range(0, len(LFEtime)):
            mySecond = int(floor(LFEtime['second'].iloc[i]))
            myMicrosecond = int(1000000.0 * \
                (LFEtime['second'].iloc[i] - floor(LFEtime['second'].iloc[i])))
            Tori = UTCDateTime(year=LFEtime['year'].iloc[i], \
                month=LFEtime['month'].iloc[i], day=LFEtime['day'].iloc[i], \
                hour=LFEtime['hour'].iloc[i], minute=LFEtime['minute'].iloc[i], \
                second=mySecond, microsecond=myMicrosecond)
            Tstart = Tori - TDUR
            Tend = Tori + 60.0 + TDUR
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
        # Stack
        if (len(EW) > 0 and len(NS) > 0 and len(UD) > 0):
            # Stack waveforms
            EWstack = linstack([EW], normalize=True, method=method) 
            NSstack = linstack([NS], normalize=True, method=method)
            UDstack = linstack([UD], normalize=True, method=method)
            # Plot figure
            plt.figure(1, figsize=(20, 15))
            # East - West component
            ax1 = plt.subplot(311)
            dt = EWstack[0].stats.delta
            nt = EWstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, EWstack[0].data, 'k')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('East - West component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            # North - South component
            ax2 = plt.subplot(312)
            dt = NSstack[0].stats.delta
            nt = NSstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, NSstack[0].data, 'k')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('North - South component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            # Vertical component
            ax3 = plt.subplot(313)
            dt = UDstack[0].stats.delta
            nt = UDstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, UDstack[0].data, 'k')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('Vertical component', fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            # End figure
            plt.suptitle(station, fontsize=24)
            plt.savefig(namedir + '/' + station + '.eps', format='eps')
            ax1.clear()
            ax2.clear()
            ax3.clear()
            plt.close(1)
            # Save stack into file
            savename = namedir + '/' + station +'.pkl'
            pickle.dump([EWstack[0], NSstack[0], UDstack[0]], \
                open(savename, 'wb'))

if __name__ == '__main__':

    # Set the parameters
    filename = '080421.14.048'
    catalog = 'catalog_200708_200901.pkl'
    threshold = 0.025
    TDUR = 10.0
    filt = (1.5, 9.0)
    dt = 0.05
    method = 'RMS'

    compute_new_templates(filename, catalog, threshold, TDUR, filt, dt, method)
