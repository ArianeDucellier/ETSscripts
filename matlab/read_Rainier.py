"""
Script to compare preprocessing done with Matlab or Python
Waveforms are downloaded from Rainier
"""

from obspy import UTCDateTime
import obspy.clients.earthworm.client as earthworm
from obspy import read_inventory
from obspy.core.stream import Stream

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Function to compute error
def error(d1, d2):
    res = np.abs(d1 - d2) / np.max(np.abs(d1))
    return res

# Parameters
arrayName = 'DR'
staCodes = ['DR01', 'DR02', 'DR03', 'DR04', 'DR05', 'DR06', 'DR07', 'DR08', \
            'DR09', 'DR10', 'DR12']
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

earthworm_client = earthworm.Client('rainier.ess.washington.edu', 16017)

# Preprocessing
trace = []
for ksta in range(0, len(staCodes)):
    D = earthworm_client.get_waveforms(network=network, \
            station=staCodes[ksta], location='', channel=chans, \
            starttime=Tstart, endtime=Tend)
    if len(D) > 0:
        trace.append(D[0])
D1 = Stream(traces=trace)
D2 = D1.copy()
D2.detrend(type='linear')
D3 = D2.copy()
D3.taper(type='hann', max_percentage=0.0625)
D4 = D3.copy()
inventory = read_inventory('../data/response/' + network + '_' + arrayName \
    + '.xml', format='STATIONXML')
D4.attach_response(inventory)
D4.remove_response(output="VEL", pre_filt=(0.2, 0.5, 10.0, 15.0), \
    water_level=80.0)
D5 = D4.copy()
D5.filter('bandpass', freqmin=filt[0], freqmax=filt[1], zerophase=True)
D6 = D5.copy()
D6.resample(20.0, no_filter=True)
D7 = D5.copy()
D7.interpolate(100.0, method='lanczos', a=10)
D8 = D7.copy()
D8.decimate(5, no_filter=True)

# Plot seismograms
for ksta in range(0, len(D1)):
    plt.figure(1, figsize=(24, 20))

    plt.subplot(321)
    dt = 1.0 / D1[ksta].stats.sampling_rate
    nt = D1[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename1 = 'matlab/' + str(D1[ksta].stats.station) + '_1.mat'
    data = loadmat(filename1)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], data_m[i1 : i2], 'r-', label='Matlab')
    plt.plot(time, D1[ksta].data[i1 : i2], 'k-', label='Python')
    plt.xlim(0, 60.0)
    plt.title('obspy.clients.earthworm.client.Client.get_waveforms')
    plt.legend()

    plt.subplot(322)
    dt = 1.0 / D2[ksta].stats.sampling_rate
    nt = D2[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename2 = 'matlab/' + str(D2[ksta].stats.station) + '_2.mat'
    data = loadmat(filename2)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], data_m[i1 : i2], 'r-', label='Matlab')
    plt.plot(time, D2[ksta].data[i1 : i2], 'k-', label='Python')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.detrend')
    plt.legend()

    plt.subplot(323)
    dt = 1.0 / D3[ksta].stats.sampling_rate
    nt = D3[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename3 = 'matlab/' + str(D3[ksta].stats.station) + '_3.mat'
    data = loadmat(filename3)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], data_m[i1 : i2], 'r-', label='Matlab')
    plt.plot(time, D3[ksta].data[i1 : i2], 'k-', label='Python')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.taper')
    plt.legend()

    plt.subplot(324)
    dt = 1.0 / D4[ksta].stats.sampling_rate
    nt = D4[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename4 = 'matlab/' + str(D4[ksta].stats.station) + '_4.mat'
    data = loadmat(filename4)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], data_m[i1 : i2], 'r-', label='Matlab')
    plt.plot(time, D4[ksta].data[i1 : i2], 'k-', label='Python')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.remove_response')
    plt.legend()

    plt.subplot(325)
    dt = 1.0 / D5[ksta].stats.sampling_rate
    nt = D5[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename5 = 'matlab/' + str(D5[ksta].stats.station) + '_5.mat'
    data = loadmat(filename5)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], data_m[i1 : i2], 'r-', label='Matlab')
    plt.plot(time, D5[ksta].data[i1 : i2], 'k-', label='Python')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.filter')
    plt.legend()

    plt.subplot(326)
    dt = 1.0 / D8[ksta].stats.sampling_rate
    nt = D8[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename8 = 'matlab/' + str(D8[ksta].stats.station) + '_8.mat'
    data = loadmat(filename8)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], data_m[i1 : i2], 'r-', label='Matlab')
    plt.plot(time, D8[ksta].data[i1 : i2], 'k-', label='Python')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.interpolate + decimate')
    plt.legend()

    plt.suptitle('Station {}'.format(str(D1[ksta].stats.station)))
    plt.savefig('python_earthworm/' + str(D1[ksta].stats.station) + '.eps')
    plt.close()

# Plot differences
for ksta in range(0, len(D1)):
    plt.figure(1, figsize=(24, 20))

    plt.subplot(321)
    dt = 1.0 / D1[ksta].stats.sampling_rate
    nt = D1[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename1 = 'matlab/' + str(D1[ksta].stats.station) + '_1.mat'
    data = loadmat(filename1)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], error(data_m[i1 : i2], \
        D1[ksta].data.reshape(nt, 1)[i1 : i2]), 'k-')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.clients.earthworm.client.Client.get_waveforms')

    plt.subplot(322)
    dt = 1.0 / D2[ksta].stats.sampling_rate
    nt = D2[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename2 = 'matlab/' + str(D2[ksta].stats.station) + '_2.mat'
    data = loadmat(filename2)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], error(data_m[i1 : i2], \
        D2[ksta].data.reshape(nt, 1)[i1 : i2]), 'k-')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.detrend')

    plt.subplot(323)
    dt = 1.0 / D3[ksta].stats.sampling_rate
    nt = D3[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename3 = 'matlab/' + str(D3[ksta].stats.station) + '_3.mat'
    data = loadmat(filename3)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], error(data_m[i1 : i2], \
        D3[ksta].data.reshape(nt, 1)[i1 : i2]), 'k-')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.taper')

    plt.subplot(324)
    dt = 1.0 / D4[ksta].stats.sampling_rate
    nt = D4[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename4 = 'matlab/' + str(D4[ksta].stats.station) + '_4.mat'
    data = loadmat(filename4)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], error(data_m[i1 : i2], \
        D4[ksta].data.reshape(nt, 1)[i1 : i2]), 'k-')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.remove_response')

    plt.subplot(325)
    dt = 1.0 / D5[ksta].stats.sampling_rate
    nt = D5[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename5 = 'matlab/' + str(D5[ksta].stats.station) + '_5.mat'
    data = loadmat(filename5)
    time_m = data['time']
    data_m = data['data']
    maxv = np.max(np.abs(data_m))
    plt.plot(time_m[i1 : i2], error(data_m[i1 : i2], \
        D5[ksta].data.reshape(nt, 1)[i1 : i2]), 'k-')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.filter')

    plt.subplot(326)
    dt = 1.0 / D8[ksta].stats.sampling_rate
    nt = D8[ksta].stats.npts
    i1 = int(TDUR / dt)
    i2 = nt - i1
    time = np.linspace(0.0, (nt - 2 * i1) * dt, nt - 2 * i1, endpoint=False)
    filename6 = 'matlab/' + str(D8[ksta].stats.station) + '_6.mat'
    data = loadmat(filename6)
    time_m = data['time']
    data_m = data['data']
    plt.plot(time_m[i1 : i2], error(data_m[i1 : i2], \
        D8[ksta].data.reshape(nt, 1)[i1 : i2]), 'k-')
    plt.xlim(0.0, 60.0)
    plt.title('obspy.core.stream.Stream.interpolate + decimate')

    plt.suptitle('Station {}'.format(str(D1[ksta].stats.station)))
    plt.savefig('python_earthworm/' + str(D1[ksta].stats.station) + \
        '_diff.eps')
    plt.close()

# Plot Fourier transform
    for ksta in range(0, len(D1)):
        plt.figure(3, figsize=(24, 20))

        # FFT before interpolation
        dt5 = 1.0 / D5[ksta].stats.sampling_rate
        nt5 = D5[ksta].stats.npts
        df5 = 1.0 / (dt5 * nt5)
        nf5 = int((nt5 + 1) / 2)
        fft5 = np.fft.rfft(D5[ksta].data)
        # FFT after resampling
        dt6 = 1.0 / D6[ksta].stats.sampling_rate
        nt6 = D6[ksta].stats.npts
        df6 = 1.0 / (dt6 * nt6)
        nf6 = int(nt6 / 2) + 1
        fft6 = np.fft.rfft(D6[ksta].data)
        # FFT after interpolation
        dt7 = 1.0 / D7[ksta].stats.sampling_rate
        nt7 = D7[ksta].stats.npts
        df7 = 1.0 / (dt7 * nt7)
        nf7 = int((nt7 + 1) / 2)
        fft7 = np.fft.rfft(D7[ksta].data)
        # FFT after interpolation and decimation
        dt8 = 1.0 / D8[ksta].stats.sampling_rate
        nt8 = D8[ksta].stats.npts
        df8 = 1.0 / (dt8 * nt8)
        nf8 = int((nt8 + 1) / 2)
        fft8 = np.fft.rfft(D8[ksta].data)
        # FFT of Matlab file before interpolation
        filename5 = 'matlab/' + str(D1[ksta].stats.station) + '_5.mat'
        data = loadmat(filename5)
        data_m5 = data['data']
        dtm5 = 0.02
        ntm5 = len(data_m5)
        dfm5 = 1.0 / (dtm5 * ntm5)
        nfm5 = int((ntm5 + 1) / 2)
        fftm5 = np.fft.rfft(data_m5.reshape(ntm5,))
        # FFT of Matlab file after resampling
        filename6 = 'matlab/' + str(D1[ksta].stats.station) + '_6.mat'
        data = loadmat(filename6)
        data_m6 = data['data']
        dtm6 = 0.05
        ntm6 = len(data_m6)
        dfm6 = 1.0 / (dtm6 * ntm6)
        nfm6 = int((ntm6 + 1) / 2)
        fftm6 = np.fft.rfft(data_m6.reshape(ntm6,))
        # FFT of Matlab file after interpolation
        filename7 = 'matlab/' + str(D1[ksta].stats.station) + '_7.mat'
        data = loadmat(filename7)
        data_m7 = data['data']
        dtm7 = 0.01
        ntm7 = len(data_m7)
        dfm7 = 1.0 / (dtm7 * ntm7)
        nfm7 = int(ntm7 / 2) + 1
        fftm7 = np.fft.rfft(data_m7.reshape(ntm7,))
        # FFT of Matlab file after interpolation and decimation
        filename8 = 'matlab/' + str(D1[ksta].stats.station) + '_8.mat'
        data = loadmat(filename8)
        data_m8 = data['data']
        dtm8 = 0.05
        ntm8 = len(data_m8)
        dfm8 = 1.0 / (dtm8 * ntm8)
        nfm8 = int((ntm8 + 1) / 2)
        fftm8 = np.fft.rfft(data_m8.reshape(ntm8,))
        # Plot initial values
        plt.subplot(221)
        freqm5 = np.linspace(0, nfm5 * dfm5, nfm5, endpoint=False)
        plt.plot(freqm5, dtm5 * np.abs(fftm5), 'r-', label='Matlab')
        freq5 = np.linspace(0, nf5 * df5, nf5, endpoint=False)
        plt.plot(freq5, dt5 * np.abs(fft5), 'k-', label='Python')
        plt.xlim(0.0, 10.0)
        plt.title('Fourier transform before resampling')
        plt.legend()
        # Plot values after resampling
        plt.subplot(222)
        freqm6 = np.linspace(0, nfm6 * dfm6, nfm6, endpoint=False)
        plt.plot(freqm6, dtm6 * np.abs(fftm6), 'r-', label='Matlab')
        freq6 = np.linspace(0, nf6 * df6, nf6, endpoint=False)
        plt.plot(freq6, dt6 * np.abs(fft6), 'k-', label='Python')
        plt.xlim(0.0, 10.0)
        plt.title('Fourier transform after resampling')
        plt.legend()
        # Plot values after interpolation
        plt.subplot(223)
        freqm7 = np.linspace(0, nfm7 * dfm7, nfm7, endpoint=False)
        plt.plot(freqm7, dtm7 * np.abs(fftm7), 'r-', label='Matlab')
        freq7 = np.linspace(0, nf7 * df7, nf7, endpoint=False)
        plt.plot(freq7, dt7 * np.abs(fft7), 'k-', label='Python')
        plt.xlim(0.0, 10.0)
        plt.title('Fourier transform after interpolation')
        plt.legend()
        # Plot values after interpolation and decimation
        plt.subplot(224)
        freqm8 = np.linspace(0, nfm8 * dfm8, nfm8, endpoint=False)
        plt.plot(freqm8, dt8 * np.abs(fftm8), 'r-', label='Matlab')
        freq8 = np.linspace(0, nf8 * df8, nf8, endpoint=False)
        plt.plot(freq8, dt8 * np.abs(fft8), 'k-', label='Python')
        plt.xlim(0.0, 10.0)
        plt.title('Fourier transform after interpolation and decimation')
        plt.legend()
        plt.suptitle('Station {}'.format(str(D1[ksta].stats.station)))
        plt.savefig('python_earthworm/' + str(D1[ksta].stats.station) + \
            '_fft.eps')
        plt.close()
