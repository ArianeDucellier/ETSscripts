"""
This module contains a function to carry out a statistical analysis
of the values of the cross correlation between the waveform for
an LFE and the template for this family
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def compute_stats(filename):
    """
    Input:
        type filename = string
        filename = Name of the template
    Output:
        type df = pandas dataframe
        df = Statistics on cross correlation
    """
    # Get the names of the stations which have a waveform for this LFE family
    file = open('../data/Plourde_2015/detections/' + filename + \
        '_detect5_cull.txt')
    first_line = file.readline().strip()
    staNames = first_line.split()
    file.close()

    # Initialize lists
    site = []
    component = []
    cmean = []
    cmin = []
    cmax = []
    cmed = []
    cstd = []

    # Initialize figure
    params = {'xtick.labelsize':16,
              'ytick.labelsize':16}
    pylab.rcParams.update(params)   
    plt.figure(1, figsize=(21, 5 * len(staNames)))

    # Loop over stations
    for station, i in zip(staNames, range(0, len(staNames))):
        data = pickle.load(open('templates/' + filename + \
            '/' + station + '_cc.pkl', 'rb'))
        cc0EW = data[0]
        cc0NS = data[1]
        cc0UD = data[2]
        for channel, j in zip(['EW', 'NS', 'UD'], range(0, 3)):
            plt.subplot2grid((len(staNames), 3), (i, j))
            if (channel == 'EW'):
                cc = cc0EW
            elif (channel == 'NS'):
                cc = cc0NS
            else:
                cc = cc0UD
            plt.hist(cc, bins =np.linspace(-0.2, 0.5, 15))
            plt.title(station + ' - ' + channel, fontsize=20)
            site.append(station)
            component.append(channel)
            cmean.append(np.mean(cc))
            cmin.append(np.min(cc))
            cmax.append(np.max(cc))
            cmed.append(np.median(cc))
            cstd.append(np.std(cc))

    # End figure
    plt.suptitle('Distribution of cross-correlation values', \
        fontsize=30)
    plt.savefig('templates/' + filename + '/cc_hist.eps', \
        format='eps')
    plt.close(1)

    # Saving into pandas dataframe
    df = pd.DataFrame(
        {'station': site,
         'channel': component,
         'mean': cmean,
         'minimum': cmin,
         'maximum': cmax,
         'median': cmed,
         'standard_deviation': cstd
        })
    return df
