"""
This module contains a function to download every one-minute time window
where there is an LFE recorded, stack the signal over all the LFEs, cross
correlate each window with the stack, sort the LFEs and keep only the best
We also save the value of the maximum cross correlation for each LFE
"""

import obspy
from obspy import UTCDateTime
from obspy.core.stream import Stream
from obspy.signal.cross_correlation import correlate

import matplotlib.cm as cm
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from math import cos, pi, sin, sqrt

from get_data import get_from_IRIS, get_from_NCEDC
from stacking import linstack

def compute_templates(filename, TDUR, filt, ratios, dt, ncor, window, \
        winlength, nattempts, waittime, method='RMS'):
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
        type nattempts = integer
        nattempts = Number of times we try to download data
        type waittime = positive float
        waittime = Type to wait between two attempts at downloading
        type method = string
        method = Normalization method for linear stack (RMS or Max)
    Output:
        None
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

    # Get the network, channels, and location of the stations
    staloc = pd.read_csv('../data/Plourde_2015/station_locations.txt', \
        sep=r'\s{1,}', header=None)
    staloc.columns = ['station', 'network', 'channels', 'location', \
        'server', 'latitude', 'longitude']

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

    # Create directory to store the waveforms
    namedir = 'templates/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    # Read origin time and station slowness files
    origintime = pickle.load(open('timearrival/origintime.pkl', 'rb'))
    slowness = pickle.load(open('timearrival/slowness.pkl', 'rb'))

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
                # Compute source-receiver distance
                latitude = staloc['latitude'][ir]
                longitude = staloc['longitude'][ir]
                xr = dx * (longitude - lon0)
                yr = dy * (latitude - lat0)
                distance = sqrt((xr - xs) ** 2.0 + (yr - ys) ** 2.0)
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
                    location, Tstart, Tend, filt, dt, nattempts, waittime, \
                    errorfile)
            # Second case: we get the data from NCEDC
            elif (server == 'NCEDC'):
                (D, orientation) = get_from_NCEDC(station, network, channels, \
                    location, Tstart, Tend, filt, dt, nattempts, waittime, \
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
        # Stack
        if (len(EW) > 0 and len(NS) > 0 and len(UD) > 0):
            # Stack waveforms
            EWstack = linstack([EW], normalize=True, method=method) 
            NSstack = linstack([NS], normalize=True, method=method)
            UDstack = linstack([UD], normalize=True, method=method)
            # Initializations
            maxCC = np.zeros(len(EW))
            cc0EW = np.zeros(len(EW))
            cc0NS = np.zeros(len(EW))
            cc0UD = np.zeros(len(EW))
            if (window == True):
                # Get time arrival
                arrivaltime = origintime[filename] + \
                    slowness[station] * distance
                Tmin = arrivaltime - winlength / 2.0
                Tmax = arrivaltime + winlength / 2.0
                if Tmin < 0.0:
                    Tmin = 0.0
                if Tmax > EWstack[0].stats.delta * (EWstack[0].stats.npts - 1):
                    Tmax = EWstack[0].stats.delta * (EWstack[0].stats.npts - 1)
                ibegin = int(Tmin / EWstack[0].stats.delta)
                iend = int(Tmax / EWstack[0].stats.delta) + 1
                # Cross correlation
                for i in range(0, len(EW)):
                    ccEW = correlate(EWstack[0].data[ibegin : iend], \
                        EW[i].data[ibegin : iend], ncor)
                    ccNS = correlate(NSstack[0].data[ibegin : iend], \
                        NS[i].data[ibegin : iend], ncor)
                    ccUD = correlate(UDstack[0].data[ibegin : iend], \
                        UD[i].data[ibegin : iend], ncor)
                    maxCC[i] = np.max(ccEW) + np.max(ccNS) + np.max(ccUD)
                    cc0EW[i] = ccEW[ncor]
                    cc0NS[i] = ccNS[ncor]
                    cc0UD[i] = ccUD[ncor]
            else:
                # Cross correlation
                for i in range(0, len(EW)):
                    ccEW = correlate(EWstack[0].data, EW[i].data, ncor)
                    ccNS = correlate(NSstack[0].data, NS[i].data, ncor)
                    ccUD = correlate(UDstack[0].data, UD[i].data, ncor)
                    maxCC[i] = np.max(ccEW) + np.max(ccNS) + np.max(ccUD)
                    cc0EW[i] = ccEW[ncor]
                    cc0NS[i] = ccNS[ncor]
                    cc0UD[i] = ccUD[ncor]
            # Sort cross correlations
            index = np.flip(np.argsort(maxCC), axis=0)
            EWbest = Stream()
            NSbest = Stream()
            UDbest = Stream()
            # Compute stack of best LFEs
            for j in range(0, len(ratios)):
                nLFE = int(ratios[j] * len(EW) / 100.0)
                EWselect = Stream()
                NSselect = Stream()
                UDselect = Stream()
                for i in range(0, nLFE):
                    EWselect.append(EW[index[i]])
                    NSselect.append(NS[index[i]])
                    UDselect.append(UD[index[i]])
                # Stack best LFEs
                EWbest.append(linstack([EWselect], normalize=True, \
                    method=method)[0])
                NSbest.append(linstack([NSselect], normalize=True, \
                    method=method)[0])
                UDbest.append(linstack([UDselect], normalize=True, \
                    method=method)[0])
            # Plot figure
            plt.figure(1, figsize=(20, 15))
            params = {'xtick.labelsize':16,
                      'ytick.labelsize':16}
            pylab.rcParams.update(params) 
            colors = cm.rainbow(np.linspace(0, 1, len(ratios)))
            # East - West component
            ax1 = plt.subplot(311)
            dt = EWstack[0].stats.delta
            nt = EWstack[0].stats.npts
            t = dt * np.arange(0, nt)
            for j in range(0, len(ratios)):
                if (method == 'RMS'):
                    norm = EWbest[j].data / np.sqrt(np.mean(np.square( \
                        EWbest[j].data)))
                elif (method == 'MAD'):
                    norm = EWbest[j].data / np.median(np.abs(EWbest[j].data - \
                        np.median(EWbest[j].data)))
                else:
                    raise ValueError('Method must be RMS or MAD')
                norm = np.nan_to_num(norm)
                plt.plot(t, norm, color = colors[j], \
                    label = str(int(ratios[j])) + '%')
            if (method == 'RMS'):
                norm = EWstack[0].data / np.sqrt(np.mean(np.square( \
                    EWstack[0].data)))
            elif (method == 'MAD'):
                norm = EWstack[0].data / np.median(np.abs(EWstack[0].data - \
                    np.median(EWstack[0].data)))
            else:
                raise ValueError('Method must be RMS or MAD')
            norm = np.nan_to_num(norm)
            plt.plot(t, norm, 'k', label='All')
            if (window == True):
                plt.axvline(Tmin, linewidth=2, color='grey')
                plt.axvline(Tmax, linewidth=2, color='grey')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('East - West component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.legend(loc=1)
            # North - South component
            ax2 = plt.subplot(312)
            dt = NSstack[0].stats.delta
            nt = NSstack[0].stats.npts
            t = dt * np.arange(0, nt)
            for j in range(0, len(ratios)):
                if (method == 'RMS'):
                    norm = NSbest[j].data / np.sqrt(np.mean(np.square( \
                        NSbest[j].data)))
                elif (method == 'MAD'):
                    norm = NSbest[j].data / np.median(np.abs(NSbest[j].data - \
                        np.median(NSbest[j].data)))
                else:
                    raise ValueError('Method must be RMS or MAD')
                norm = np.nan_to_num(norm)
                plt.plot(t, norm, color = colors[j], \
                    label = str(int(ratios[j])) + '%')
            if (method == 'RMS'):
                norm = NSstack[0].data / np.sqrt(np.mean(np.square( \
                    NSstack[0].data)))
            elif (method == 'MAD'):
                norm = NSstack[0].data / np.median(np.abs(NSstack[0].data - \
                    np.median(NSstack[0].data)))
            else:
                raise ValueError('Method must be RMS or MAD')
            norm = np.nan_to_num(norm)
            plt.plot(t, norm, 'k', label='All')
            if (window == True):
                plt.axvline(Tmin, linewidth=2, color='grey')
                plt.axvline(Tmax, linewidth=2, color='grey')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('North - South component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.legend(loc=1)
            # Vertical component
            ax3 = plt.subplot(313)
            dt = UDstack[0].stats.delta
            nt = UDstack[0].stats.npts
            t = dt * np.arange(0, nt)
            for j in range(0, len(ratios)):
                if (method == 'RMS'):
                    norm = UDbest[j].data / np.sqrt(np.mean(np.square( \
                        UDbest[j].data)))
                elif (method == 'MAD'):
                    norm = UDbest[j].data / np.median(np.abs(UDbest[j].data - \
                        np.median(UDbest[j].data)))
                else:
                    raise ValueError('Method must be RMS or MAD')
                norm = np.nan_to_num(norm)
                plt.plot(t, norm, color = colors[j], \
                    label = str(int(ratios[j])) + '%')
            if (method == 'RMS'):
                norm = UDstack[0].data / np.sqrt(np.mean(np.square( \
                    UDstack[0].data)))
            elif (method == 'MAD'):
                norm = UDstack[0].data / np.median(np.abs(UDstack[0].data - \
                    np.median(UDstack[0].data)))
            else:
                raise ValueError('Method must be RMS or MAD')
            norm = np.nan_to_num(norm)
            plt.plot(t, norm, 'k', label='All')
            if (window == True):
                plt.axvline(Tmin, linewidth=2, color='grey')
                plt.axvline(Tmax, linewidth=2, color='grey')
            plt.xlim([np.min(t), np.max(t)])
            plt.title('Vertical component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.legend(loc=1)
            # End figure
            plt.suptitle(station, fontsize=24)
            plt.savefig(namedir + '/' + station + '.eps', format='eps')
            ax1.clear()
            ax2.clear()
            ax3.clear()
            plt.close(1)
            # Save stacks into files
            savename = namedir + '/' + station +'.pkl'
            pickle.dump([EWstack[0], NSstack[0], UDstack[0]], \
                open(savename, 'wb'))
            for j in range(0, len(ratios)):
                savename = namedir + '/' + station + '_' + \
                    str(int(ratios[j])) + '.pkl'
                pickle.dump([EWbest[j], NSbest[j], UDbest[j]], \
                    open(savename, 'wb'))
            # Save cross correlations into files
            savename = namedir + '/' + station + '_cc.pkl'
            pickle.dump([cc0EW, cc0NS, cc0UD], \
                open(savename, 'wb'))

if __name__ == '__main__':

    # Set the parameters
    TDUR = 10.0
    filt = (1.5, 9.0)
    ratios = [50.0, 60.0, 70.0, 80.0, 90.0]
    dt = 0.05
    ncor = 400
    window = False
    winlength = 10.0
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
        compute_templates(filename, TDUR, filt, ratios, dt, ncor, window, \
            winlength, nattempts, waittime, method)
