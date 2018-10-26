"""
This module contains different methods of stacking of seismic signal
"""

from obspy.core.stream import Stream
from obspy.signal.filter import envelope

import numpy as np

from scipy.signal import hilbert

def linstack(streams, normalize=True, method='RMS'):
    """
    Compute the linear stack of a list of streams
    Several streams -> returns a stack for each station and each channel
    One stream -> returns a stack for each channel (and merge stations)

    Input:
        type streams = list of streams
        streams = List of streams to stack
        type normalize = boolean
        normalize = Normalize traces by RMS amplitude before stacking
    Output:
        type stack = stream
        stack = Stream with stacked traces for each channel (and each station)
    """
    # If there are several streams in the list,
    # return one stack for each station and each channel
    if len(streams) > 1:
        stack = streams[np.argmax([len(stream) for stream in streams])].copy()
    # If there is only one stream in the list,
    # return one stack for each channel and merge the stations
    else:
        channels = []
        for tr in streams[0]:
            if not(tr.stats.channel in channels):
                channels.append(tr.stats.channel)
        stack = Stream()
        for i in range(0, len(channels)):
            stack.append(streams[0][0].copy())
            stack[-1].stats['channel'] = channels[i]
            stack[-1].stats['station'] = 'all'
    # Initialize trace to 0
    for tr in stack:
        tr.data = np.zeros(tr.stats.npts)
    # Initialize number of traces stacked to 0
    ntr = np.zeros((len(stack)))
    # Stack traces
    for i in range(0, len(streams)):
        for k in range (0, len(stack)):
            if len(streams) > 1:
                matchtr = streams[i].select(station=stack[k].stats.station, \
                                            channel=stack[k].stats.channel)
            else:
                matchtr = streams[i].select(channel=stack[k].stats.channel)
            for j in range(0, len(matchtr)):
                ntr[k] = ntr[k] + 1
                # Normalize the data before stacking
                if normalize:
                    if (method == 'RMS'):
                        norm = matchtr[j].data / \
                            np.sqrt(np.mean(np.square(matchtr[j].data)))
                    elif (method == 'Max'):
                        norm = matchtr[j].data / \
                            np.max(np.abs(matchtr[j].data))
                    else:
                        raise ValueError( \
                            'Method must be RMS or Max')
                    norm = np.nan_to_num(norm)
                else:
                    norm = matchtr[j].data
                stack[k].data = np.sum((norm, stack[k].data), axis=0)
    # Divide by the number of traces stacked
    for k in range (0, len(stack)):
        stack[k].data = stack[k].data / ntr[k]
    return stack

def powstack(streams, weight=2.0, normalize=True):
    """
    Compute the power (Nth-root) stack of a list of streams
    Several streams -> returns a stack for each station and each channel
    One stream -> returns a stack for each channel (and merge stations)

    Input:
        type streams = list of streams
        streams = List of streams to stack
        type weight = float
        weight = Power of the stack (usually integer greater than 1)
        type normalize = boolean
        normalize = Normalize traces by RMS amplitude before stacking
    Output:
        type stack = stream
        stack = Stream with stacked traces for each channel (and each station)
    """
    # If there are several streams in the list,
    # return one stack for each station and each channel
    if len(streams) > 1:
        stack = streams[np.argmax([len(stream) for stream in streams])].copy()
    # If there is only one stream in the list,
    # return one stack for each channel and merge the stations
    else:
        channels = []
        for tr in streams[0]:
            if not(tr.stats.channel in channels):
                channels.append(tr.stats.channel)
        stack = Stream()
        for i in range(0, len(channels)):
            stack.append(streams[0][0].copy())
            stack[-1].stats['channel'] = channels[i]
            stack[-1].stats['station'] = 'all'
    # Initialize trace to 0
    for tr in stack:
        tr.data = np.zeros(tr.stats.npts)
    # Initialize number of traces stacked to 0
    ntr = np.zeros((len(stack)))
    # Stack traces
    for i in range(0, len(streams)):
        for k in range (0, len(stack)):
            if len(streams) > 1:
                matchtr = streams[i].select(station=stack[k].stats.station, \
                                            channel=stack[k].stats.channel)
            else:
                matchtr = streams[i].select(channel=stack[k].stats.channel)
            for j in range(0, len(matchtr)):
                ntr[k] = ntr[k] + 1
                # Normalize the data before stacking
                if normalize:
                    norm = matchtr[j].data / \
                        np.sqrt(np.mean(np.square(matchtr[j].data)))
                    norm = np.nan_to_num(norm)
                else:
                    norm = matchtr[j].data
                stack[k].data = np.sum((np.power(np.abs(norm), 1.0 / weight) \
                     * np.sign(norm), stack[k].data), axis=0)
    # Take the power of the stack and divide by the number of traces stacked
    for k in range (0, len(stack)):
        stack[k].data = np.sign(stack[k].data) * np.power(stack[k].data, \
            weight) / ntr[k]
    return stack

def PWstack(streams, weight=2, normalize=True):
    """
    Compute the phase-weighted stack of a list of streams
    Several streams -> returns a stack for each station and each channel
    One stream -> returns a stack for each channel (and merge stations)

    Input:
        type streams = list of streams
        streams = List of streams to stack
        type weight = float
        weight = Power of the stack (usually integer greater than 1)
        type normalize = boolean
        normalize = Normalize traces by RMS amplitude before stacking
    Output:
        type stack = stream
        stack = Stream with stacked traces for each channel (and each station)
    """
    # First get the linear stack which we will weight by the phase stack
    Linstack = linstack(streams, normalize=normalize)
    # Compute the instantaneous phase
    instaphases = []
    for stream in streams:
        instaphase = stream.copy()
        for tr in instaphase:
            analytic = hilbert(tr.data)
            env = envelope(tr.data)
            tr.data = analytic / env
            tr.data = np.nan_to_num(tr.data)
        instaphases.append(instaphase)
    # Compute the phase stack
    Phasestack = linstack(instaphases, normalize=False)
    # Compute the phase-weighted stack
    for tr in Phasestack:
        tr.data = Linstack.select(station=tr.stats.station, \
                                  channel=tr.stats.channel)[0].data \
            * np.power(np.abs(tr.data), weight)
    return Phasestack
