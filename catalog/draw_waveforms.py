"""
This module contains functions to transform the LFE waveforms previously
downloaded and draw them
"""

import obspy
from obspy.core.stream import Stream

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from math import log, sqrt

import MODWT

from stacking import linstack

def remove_trace(stream):
    """
    """
    lengths = []
    for trace in stream:
        lengths.append(trace.stats.npts)
    freqlen = max(set(lengths), key=lengths.count)
    newstream = stream.select(npts=freqlen)
    return newstream

def plot_figure(EW, NS, UD, station, filename, method='RMS',
    draw_stack=True, draw_template=True, save_template=True):
    """
    """
    # Draw the stack
    if (draw_stack == True):
        plt.figure(1, figsize=(30, 15))
        # EW component
        if (len(EW) > 0):
            EWstack = linstack([EW], normalize=True, method=method) 
            ax1 = plt.subplot(131)
            for i in range(0, len(EW)):
                dt = EW[i].stats.delta
                nt = EW[i].stats.npts
                t = dt * np.arange(0, nt)
                datanorm = EW[i].data / np.max(np.abs(EW[i].data))
                plt.plot(t, (2.0 * i + 1) + datanorm, 'k-')
            datanorm = EWstack[0].data / np.max(np.abs(EWstack[0].data))
            plt.plot(t, - 2.0 + datanorm, 'r-')
            plt.xlim(0.0, 60.0)
            plt.ylim(- 3.0, 2.0 * len(EW))
            plt.title('East component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.ylabel('Velocity (m/s)', fontsize=24)
            ax1.set_yticklabels([])
            ax1.tick_params(labelsize=20)
        # NS component
        if (len(NS) > 0):
            NSstack = linstack([NS], normalize=True, method=method)
            ax2 = plt.subplot(132)
            for i in range(0, len(NS)):
                dt = NS[i].stats.delta
                nt = NS[i].stats.npts
                t = dt * np.arange(0, nt)
                datanorm = NS[i].data / np.max(np.abs(NS[i].data))
                plt.plot(t, (2.0 * i + 1) + datanorm, 'k-')
            datanorm = NSstack[0].data / np.max(np.abs(NSstack[0].data))
            plt.plot(t, - 2.0 + datanorm, 'r-')
            plt.xlim(0.0, 60.0)
            plt.ylim(- 3.0, 2.0 * len(NS))
            plt.title('North component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.ylabel('Velocity (m/s)', fontsize=24)
            ax2.set_yticklabels([])
            ax2.tick_params(labelsize=20)
        # UD component
        if (len(UD) > 0):
            UDstack = linstack([UD], normalize=True, method=method)
            ax3 = plt.subplot(133)
            for i in range(0, len(UD)):
                dt = UD[i].stats.delta
                nt = UD[i].stats.npts
                t = dt * np.arange(0, nt)
                datanorm = UD[i].data / np.max(np.abs(UD[i].data))
                plt.plot(t, (2.0 * i + 1) + datanorm, 'k-')  
            datanorm = UDstack[0].data / np.max(np.abs(UDstack[0].data))
            plt.plot(t, - 2.0 + datanorm, 'r-')
            plt.xlim(0.0, 60.0)
            plt.ylim(- 3.0, 2.0 * len(UD))
            plt.title('Vertical component', fontsize=24)
            plt.xlabel('Time (s)', fontsize=24)
            plt.ylabel('Velocity (m/s)', fontsize=24)
            ax3.set_yticklabels([])
            ax3.tick_params(labelsize=20)
        # End and save figure
        plt.suptitle(station, fontsize=24)
        plt.savefig(filename + '_stack.eps', format='eps')
        if (len(EW) > 0):
            ax1.clear()
        if (len(NS) > 0):
            ax2.clear()
        if (len(UD) > 0):
            ax3.clear()
        plt.close(1)
    # Draw the template
    if (draw_template == True):
        plt.figure(2, figsize=(20, 15))
        # EW component
        if (len(EW) > 0):
            EWstack = linstack([EW], normalize=True, method=method)
            ax1 = plt.subplot(311)
            dt = EWstack[0].stats.delta
            nt = EWstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, EWstack[0].data, 'k')
            plt.xlim(0.0, 60.0)
            plt.title('East component ({:.3f})'.format( \
                np.max(np.abs(EWstack[0].data)) / np.sqrt(np.mean(np.square( \
                EWstack[0].data)))), fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Velocity (m/s)', fontsize=16)
        # NS component
        if (len(NS) > 0):
            NSstack = linstack([NS], normalize=True, method=method)
            ax2 = plt.subplot(312)
            dt = NSstack[0].stats.delta
            nt = NSstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, NSstack[0].data, 'k')
            plt.xlim(0.0, 60.0)
            plt.title('North component ({:.3f})'.format( \
                np.max(np.abs(NSstack[0].data)) / np.sqrt(np.mean(np.square( \
                NSstack[0].data)))), fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Velocity (m/s)', fontsize=16)
        # UD component
        if (len(UD) > 0):
            UDstack = linstack([UD], normalize=True, method=method)
            ax3 = plt.subplot(313)
            dt = UDstack[0].stats.delta
            nt = UDstack[0].stats.npts
            t = dt * np.arange(0, nt)
            plt.plot(t, UDstack[0].data, 'k')
            plt.xlim(0.0, 60.0)
            plt.title('Vertical component ({:.3f})'.format( \
                np.max(np.abs(UDstack[0].data)) / np.sqrt(np.mean(np.square( \
                UDstack[0].data)))), fontsize=16)
            plt.xlabel('Time (s)', fontsize=16)
            plt.ylabel('Velocity (m/s)', fontsize=16)
        # End and save figure
        plt.suptitle(station, fontsize=24)
        plt.savefig(filename + '_template.eps', format='eps')
        if (len(EW) > 0):
            ax1.clear()
        if (len(NS) > 0):
            ax2.clear()
        if (len(UD) > 0):
            ax3.clear()
        plt.close(1)
    # Save the template
    if (save_template == True):
        channel = []
        template = []
        # EW component
        if (len(EW) > 0):
            EWstack = linstack([EW], normalize=True, method=method)
            channel.append('EW')
            template.append(EWstack[0])
        # NS component
        if (len(NS) > 0):
            NSstack = linstack([NS], normalize=True, method=method)
            channel.append('NS')
            template.append(NSstack[0])
        # UD component
        if (len(UD) > 0):
            UDstack = linstack([UD], normalize=True, method=method)
            channel.append('UD')
            template.append(UDstack[0])
        savename = filename + '.pkl'
        pickle.dump([channel, template], open(savename, 'wb'))
 
def draw(filename, method='RMS', draw_stack=True, \
    draw_template=True, save_template=True):
    """
    """
    # Create directory to store the figures
    namedir = 'data/raw/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    data = pickle.load(open('data/' + filename + '.pkl', 'rb'))
    stations = data[0]
    EWall = data[1]
    NSall = data[2]
    UDall = data[3]

    for station, index in zip(stations, range(0, len(stations))):
        EW = EWall[index]
        NS = NSall[index]
        UD = UDall[index]
        if (len(EW) > 0 or len(NS) > 0 or len(UD) > 0):
            # Remove traces with different lengths
            if (len(EW) > 0):
                EW = remove_trace(EW)
            if (len(NS) > 0):
                NS = remove_trace(NS)
            if (len(UD) > 0):
                UD = remove_trace(UD)
            # Plot
            plot_figure(EW, NS, UD, station, 'data/raw/' + \
				filename + '/' + station, method, draw_stack, \
                draw_template, save_template)

def draw_MODWT(filename, name, J, method='RMS', draw_stack=True, \
    draw_template=True, save_template=True):
    """
    """
    # Create directory to store the figures
    namedir = 'data/MODWT/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    data = pickle.load(open('data/' + filename + '.pkl', 'rb'))
    stations = data[0]
    EWall = data[1]
    NSall = data[2]
    UDall = data[3]

    for station, index in zip(stations, range(0, len(stations))):
        EW = EWall[index]
        NS = NSall[index]
        UD = UDall[index]
        if (len(EW) > 0 or len(NS) > 0 or len(UD) > 0):
            # Remove traces with different lengths
            if (len(EW) > 0):
                EW = remove_trace(EW)
            if (len(NS) > 0):
                NS = remove_trace(NS)
            if (len(UD) > 0):
                UD = remove_trace(UD)
            # Initializations for MODWT
            W_EW = []
            W_NS = []
            W_UD = []
            D_EW = []
            D_NS = []
            D_UD = []
            for j in range(0, J):
                W_EW.append(Stream())
                W_NS.append(Stream())
                W_UD.append(Stream())
                D_EW.append(Stream())
                D_NS.append(Stream())
                D_UD.append(Stream())
            V_EW = Stream()
            V_NS = Stream()
            V_UD = Stream()
            S_EW = Stream()
            S_NS = Stream()
            S_UD = Stream()
            # MODWT
            if (len(EW) > 0):
                 for i in range(0, len(EW)):
                     (W, V) = MODWT.pyramid(EW[i].data, name, J)
                     (D, S) = MODWT.get_DS(EW[i].data, W, name, J)
                     VJ = EW[i].copy()
                     VJ.data = V
                     V_EW.append(VJ)
                     SJ = EW[i].copy()
                     SJ.data = S[J]
                     S_EW.append(SJ)
                     for j in range(0, J):
                         Wj = EW[i].copy()
                         Wj.data = W[j]
                         W_EW[j].append(Wj)
                         Dj = EW[i].copy()
                         Dj.data = D[j]
                         D_EW[j].append(Dj)
            if (len(NS) > 0):
                 for i in range(0, len(NS)):
                     (W, V) = MODWT.pyramid(NS[i].data, name, J)
                     (D, S) = MODWT.get_DS(NS[i].data, W, name, J)
                     VJ = NS[i].copy()
                     VJ.data = V
                     V_NS.append(VJ)
                     SJ = NS[i].copy()
                     SJ.data = S[J]
                     S_NS.append(SJ)
                     for j in range(0, J):
                         Wj = NS[i].copy()
                         Wj.data = W[j]
                         W_NS[j].append(Wj)
                         Dj = NS[i].copy()
                         Dj.data = D[j]
                         D_NS[j].append(Dj)
            if (len(UD) > 0):
                 for i in range(0, len(UD)):
                     (W, V) = MODWT.pyramid(UD[i].data, name, J)
                     (D, S) = MODWT.get_DS(UD[i].data, W, name, J)
                     VJ = UD[i].copy()
                     VJ.data = V
                     V_UD.append(VJ)
                     SJ = UD[i].copy()
                     SJ.data = S[J]
                     S_UD.append(SJ)
                     for j in range(0, J):
                         Wj = UD[i].copy()
                         Wj.data = W[j]
                         W_UD[j].append(Wj)
                         Dj = UD[i].copy()
                         Dj.data = D[j]
                         D_UD[j].append(Dj)
            # Plot
            for j in range(0, J):
                plot_figure(W_EW[j], W_NS[j], W_UD[j], station, \
                    'data/MODWT/' + filename + '/' + station + \
					'_W' + str(j + 1), method, draw_stack, \
                    draw_template, save_template)
                plot_figure(D_EW[j], D_NS[j], D_UD[j], station, \
                    'data/MODWT/' + filename + '/' + station + \
					'_D' + str(j + 1), method, draw_stack, \
                    draw_template, save_template)
            plot_figure(V_EW, V_NS, V_UD, station, 'data/MODWT/' + \
				filename + '/' + station + '_V' + str(J), method, \
                draw_stack, draw_template, save_template)
            plot_figure(S_EW, S_NS, S_UD, station, 'data/MODWT/' + \
				filename + '/' + station + '_S' + str(J), method, \
                draw_stack, draw_template, save_template)

def draw_denoised(filename, name, J, method='RMS', draw_stack=True, \
    draw_template=True, save_template=True):
    """
    """
    # Create directory to store the figures
    namedir = 'data/denoised/' + filename
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    data = pickle.load(open('data/' + filename + '.pkl', 'rb'))
    stations = data[0]
    EWall = data[1]
    NSall = data[2]
    UDall = data[3]

    for station, index in zip(stations, range(0, len(stations))):
        EW = EWall[index]
        NS = NSall[index]
        UD = UDall[index]
        if (len(EW) > 0 or len(NS) > 0 or len(UD) > 0):
            # Remove traces with different lengths
            if (len(EW) > 0):
                EW = remove_trace(EW)
            if (len(NS) > 0):
                NS = remove_trace(NS)
            if (len(UD) > 0):
                UD = remove_trace(UD)
            # Initializations for denoising
            EW_denoised = Stream()
            NS_denoised = Stream()
            UD_denoised = Stream()
            # MODWT
            if (len(EW) > 0):
                 for i in range(0, len(EW)):
                     sigE2 = np.mean(np.square(EW[i].data))
                     N = len(EW[i].data)
                     (W, V) = MODWT.pyramid(EW[i].data, name, J)
                     Wt = []
                     for j in range(1, J + 1):
                         Wj = W[j - 1]
                         deltaj = sqrt(2.0 * sigE2 * log(N) / (2.0 ** j))
                         Wjt = np.where(np.abs(Wj) >= deltaj, Wj, 0.0)
                         if (j == J):
                             Vt = np.where(np.abs(V) >= deltaj, V, 0.0)
                         Wt.append(Wjt)
                     denoised = EW[i].copy()
                     denoised.data = MODWT.inv_pyramid(Wt, Vt, name, J)
                     EW_denoised.append(denoised)
            if (len(NS) > 0):
                 for i in range(0, len(NS)):
                     sigE2 = np.mean(np.square(NS[i].data))
                     N = len(NS[i].data)
                     (W, V) = MODWT.pyramid(NS[i].data, name, J)
                     Wt = []
                     for j in range(1, J + 1):
                         Wj = W[j - 1]
                         deltaj = sqrt(2.0 * sigE2 * log(N) / (2.0 ** j))
                         Wjt = np.where(np.abs(Wj) >= deltaj, Wj, 0.0)
                         if (j == J):
                             Vt = np.where(np.abs(V) >= deltaj, V, 0.0)
                         Wt.append(Wjt)
                     denoised = NS[i].copy()
                     denoised.data = MODWT.inv_pyramid(Wt, Vt, name, J)
                     NS_denoised.append(denoised)
            if (len(UD) > 0):
                 for i in range(0, len(UD)):
                     sigE2 = np.mean(np.square(UD[i].data))
                     N = len(UD[i].data)
                     (W, V) = MODWT.pyramid(UD[i].data, name, J)
                     Wt = []
                     for j in range(1, J + 1):
                         Wj = W[j - 1]
                         deltaj = sqrt(2.0 * sigE2 * log(N) / (2.0 ** j))
                         Wjt = np.where(np.abs(Wj) >= deltaj, Wj, 0.0)
                         if (j == J):
                             Vt = np.where(np.abs(V) >= deltaj, V, 0.0)
                         Wt.append(Wjt)
                     denoised = UD[i].copy()
                     denoised.data = MODWT.inv_pyramid(Wt, Vt, name, J)
                     UD_denoised.append(denoised)
            # Plot
            plot_figure(EW_denoised, NS_denoised, UD_denoised, station, \
                'data/denoised/' + filename + '/' + station, method, \
                draw_stack, draw_template, save_template)

if __name__ == '__main__':

    filename = '080421.14.048'
    name = 'LA8'
    J = 6
    method = 'RMS'
    draw_stack=False
    draw_template=True
    save_template=True
#    draw(filename, method, draw_stack, draw_template, \
#        save_template)
#    draw_MODWT(filename, name, J, method, draw_stack, \
#    draw_template, save_template)
    draw_denoised(filename, name, J, method, draw_stack, \
        draw_template, save_template)
