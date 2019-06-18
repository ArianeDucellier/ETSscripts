"""
This module contains a function to take only the best LFEs from an LFE catalog,
download every one-minute time window where there is an LFE recorded,
and stack the signal over all the LFEs to get the template
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

def compute_new_templates(filename, catalog, threshold, stations, TDUR, \
    filt, dt, nattempts, waittime, method='RMS'):
    """
    This function take only the best LFEs from an LFE catalog,
    downloads every one-minute time window where there is an LFE recorded,
    and stacks the signal over all the LFEs to get the template

    Input:
        type filename = string
        filename = Name of the template
        type catalog = string
        catalog = Name of the catalog containing the LFEs
        type threshold = float
        threshold = Minimun value of cross correlation to keep LFE
        type stations = list of strings
        stations = Name of the stations where we want a template
        type TDUR = float
        TDUR = Time to add before and after the time window for tapering
        type filt = tuple of floats
        filt = Lower and upper frequencies of the filter
        type dt = float
        dt = Time step for resampling
        type nattempts = integer
        nattempts = Number of times we try to download data
        type waittime = positive float
        waittime = Type to wait between two attempts at downloading
        type method = string
        method = Normalization method for linear stack (RMS or Max)
    Output:
        None
    """
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

    # File to write error messages
    errorfile = 'error/' + filename + '.txt'

    # Loop over stations
    for station in stations:
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
                    Tstart, Tend, filt, dt, nattempts, waittime, errorfile)
            # Second case: we get the data from NCEDC
            elif (server == 'NCEDC'):
                D = get_from_NCEDC(station, network, channels, location, \
                    Tstart, Tend, filt, dt, nattempts, waittime, errorfile)
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
                elif (channels == 'EHZ'):
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
        if (len(EW) > 0 or len(NS) > 0 or len(UD) > 0):
            # Plot figure
            plt.figure(1, figsize=(20, 15))# Stack waveforms
            # East - West component
            if (len(EW) > 0):
                EWstack = linstack([EW], normalize=True, method=method) 
                ax1 = plt.subplot(311)
                dt = EWstack[0].stats.delta
                nt = EWstack[0].stats.npts
                t = dt * np.arange(0, nt)
                plt.plot(t, EWstack[0].data, 'k')
                plt.xlim([np.min(t), np.max(t)])
                plt.title('East - West component', fontsize=16)
                plt.xlabel('Time (s)', fontsize=16)
            # North - South component
            if (len(NS) > 0):
                NSstack = linstack([NS], normalize=True, method=method)
                ax2 = plt.subplot(312)
                dt = NSstack[0].stats.delta
                nt = NSstack[0].stats.npts
                t = dt * np.arange(0, nt)
                plt.plot(t, NSstack[0].data, 'k')
                plt.xlim([np.min(t), np.max(t)])
                plt.title('North - South component', fontsize=16)
                plt.xlabel('Time (s)', fontsize=16)
            # Vertical component
            if (len(UD) > 0):
                UDstack = linstack([UD], normalize=True, method=method)
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
            if (len(EW) > 0):
                ax1.clear()
            if (len(NS) > 0):
                ax2.clear()
            if (len(UD) > 0):
                ax3.clear()
            plt.close(1)
            # Save stack into file
            savename = namedir + '/' + station +'.pkl'
            data = []
            if (len(EW) > 0):
                data.append(EWstack[0])
            if (len(NS) > 0):
                data.append(NSstack[0])
            if (len(UD) > 0):
                data.append(UDstack[0])
            pickle.dump(data, open(savename, 'wb'))

if __name__ == '__main__':

    # Set the parameters
    filename = '080421.14.048'
    catalog = 'catalog_200707-200810.pkl'
    threshold = 0.08
    stations = ['KCS', 'KOM', 'KTR', 'LAM', 'LBK', 'LBO', 'LCSB', 'LGB', 'LGP', 'LHE', 'LMP', 'LPK', 'LSF', 'LSR']
    TDUR = 10.0
    filt = (1.5, 9.0)
    dt = 0.05
    nattempts = 10
    waittime = 10.0    
    method = 'RMS'

    compute_new_templates(filename, catalog, threshold, stations, TDUR, \
        filt, dt, nattempts, waittime, method)
