#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from obspy import UTCDateTime
import obspy.clients.fdsn.client as fdsn

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Function to compute error
def error(d1, d2):
    res = np.abs(d1 - d2) / sqrt(np.mean(np.power(d1, 2.0)))
    return res

# Parameters
arrayName = 'DR'
staCodes = 'DR01,DR02,DR03,DR04,DR05,DR06,DR07,DR08,DR09,DR10,DR12'
chans = 'SHE'
network = 'XG'
TDUR = 10.0
filt = (2, 8)

YY = '2010'
MM = '08'
DD = '16'
HH = '12'
mm = '30'
ss = '00'

T = UTCDateTime(YY + '-' + MM + '-' + DD + 'T' + HH + ':' + mm + ':' + ss)
Tstart = T - TDUR
Tend = T + 60.0 + TDUR

fdsn_client = fdsn.Client('IRIS')

# Preprocessing
D1 = fdsn_client.get_waveforms(network=network, station=staCodes, \
         location='--', channel=chans, starttime=Tstart, endtime=Tend, \
         attach_response=True)
D2 = D1.copy()
D2.detrend(type='linear')
D3 = D2.copy()
D3.taper(type='hann', max_percentage=0.0625)
D4 = D3.copy()
D4.remove_response(output='VEL', pre_filt=(0.2, 0.5, 10.0, 15.0), \
    water_level=80.0)
D5 = D4.copy()
D5.filter('bandpass', freqmin=filt[0], freqmax=filt[1], zerophase=True)
D6 = D5.copy()
D6.interpolate(100.0, method='lanczos', a=10)
D6.decimate(5, no_filter=True)

# Plot seismograms
for ksta in range(0, len(D1)):
    plt.figure(1, figsize=(24, 20))

    plt.subplot(321)
    dt = 1.0 / D1[ksta].stats.sampling_rate
    nt = D1[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename1 = 'matlab/' + str(D1[ksta].stats.station) + '_1.mat'
    data = loadmat(filename1)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, data_m, 'r-', label='Matlab')
    plt.plot(time, D1[ksta].data, 'k-', label='Python')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.clients.fdsn.client.Client.get_waveforms')
    plt.legend()

    plt.subplot(322)
    dt = 1.0 / D2[ksta].stats.sampling_rate
    nt = D2[ksta].stats.npts
    filename2 = 'matlab/' + str(D2[ksta].stats.station) + '_2.mat'
    data = loadmat(filename2)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, data_m, 'r-', label='Matlab')
    plt.plot(time, D2[ksta].data, 'k-', label='Python')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.detrend')
    plt.legend()

    plt.subplot(323)
    dt = 1.0 / D3[ksta].stats.sampling_rate
    nt = D3[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename3 = 'matlab/' + str(D3[ksta].stats.station) + '_3.mat'
    data = loadmat(filename3)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, data_m, 'r-', label='Matlab')
    plt.plot(time, D3[ksta].data, 'k-', label='Python')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.taper')
    plt.legend()

    plt.subplot(324)
    dt = 1.0 / D4[ksta].stats.sampling_rate
    nt = D4[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename4 = 'matlab/' + str(D4[ksta].stats.station) + '_4.mat'
    data = loadmat(filename4)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, data_m, 'r-', label='Matlab')
    plt.plot(time, D4[ksta].data, 'k-', label='Python')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.remove_response')
    plt.legend()

    plt.subplot(325)
    dt = 1.0 / D5[ksta].stats.sampling_rate
    nt = D5[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename5 = 'matlab/' + str(D5[ksta].stats.station) + '_5.mat'
    data = loadmat(filename5)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, data_m, 'r-', label='Matlab')
    plt.plot(time, D5[ksta].data, 'k-', label='Python')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.filter')
    plt.legend()

    plt.subplot(326)
    dt = 1.0 / D6[ksta].stats.sampling_rate
    nt = D6[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename6 = 'matlab/' + str(D6[ksta].stats.station) + '_6.mat'
    data = loadmat(filename6)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, data_m, 'r-', label='Matlab')
    plt.plot(time, D6[ksta].data, 'k-', label='Python')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.resample')
    plt.legend()

    plt.suptitle('Station {}'.format(str(D1[ksta].stats.station)))
    plt.savefig('python_fdsn/' + str(D1[ksta].stats.station) + '.eps')
    plt.close()

# Plot differences
for ksta in range(0, len(D1)):
    plt.figure(1, figsize=(24, 20))

    plt.subplot(321)
    dt = 1.0 / D1[ksta].stats.sampling_rate
    nt = D1[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename1 = 'matlab/' + str(D1[ksta].stats.station) + '_1.mat'
    data = loadmat(filename1)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, error(data_m, D1[ksta].data.reshape(nt, 1)), 'k-')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.clients.fdsn.client.Client.get_waveforms')

    plt.subplot(322)
    dt = 1.0 / D2[ksta].stats.sampling_rate
    nt = D2[ksta].stats.npts
    filename2 = 'matlab/' + str(D2[ksta].stats.station) + '_2.mat'
    data = loadmat(filename2)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, error(data_m, D2[ksta].data.reshape(nt, 1)), 'k-')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.detrend')

    plt.subplot(323)
    dt = 1.0 / D3[ksta].stats.sampling_rate
    nt = D3[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename3 = 'matlab/' + str(D3[ksta].stats.station) + '_3.mat'
    data = loadmat(filename3)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, error(data_m, D3[ksta].data.reshape(nt, 1)), 'k-')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.taper')

    plt.subplot(324)
    dt = 1.0 / D4[ksta].stats.sampling_rate
    nt = D4[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename4 = 'matlab/' + str(D4[ksta].stats.station) + '_4.mat'
    data = loadmat(filename4)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, error(data_m, D4[ksta].data.reshape(nt, 1)), 'k-')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.remove_response')

    plt.subplot(325)
    dt = 1.0 / D5[ksta].stats.sampling_rate
    nt = D5[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename5 = 'matlab/' + str(D5[ksta].stats.station) + '_5.mat'
    data = loadmat(filename5)
    time_m = data['time']
    data_m = data['data']
    maxv = np.max(np.abs(data_m))
    plt.plot(time_m, error(data_m, D5[ksta].data.reshape(nt, 1)), 'k-')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.filter')

    plt.subplot(326)
    dt = 1.0 / D6[ksta].stats.sampling_rate
    nt = D6[ksta].stats.npts
    time = np.linspace(- TDUR, -TDUR + nt * dt, nt, endpoint=False)
    filename6 = 'matlab/' + str(D6[ksta].stats.station) + '_6.mat'
    data = loadmat(filename6)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m, error(data_m, D6[ksta].data.reshape(nt, 1)), 'k-')
    plt.xlim(- TDUR, 60.0 + TDUR)
    plt.title('obspy.core.stream.Stream.resample')

    plt.suptitle('Station {}'.format(str(D1[ksta].stats.station)))
    plt.savefig('python_fdsn/' + str(D1[ksta].stats.station) + '_diff.eps')
    plt.close()
