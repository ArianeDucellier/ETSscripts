"""
Scripts to plot envelopes of stacks for all the grid points
"""

import obspy

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from math import sqrt

def plot_envelopes(arrayName, type_stack, cc_stack, mintremor, minratio, Tmax, amp, Vs, Vp, ds):
    """
    """
    # Get depth of plate boundary around the array
    depth_pb = pd.read_csv('depth/' + arrayName + '_depth.txt', sep=' ', \
        header=None)
    depth_pb.columns = ['x', 'y', 'depth']

    # Get number of tremor and ratio peak / RMS
    df = pickle.load(open(arrayName + '_timelag.pkl', 'rb'))

    # Create figure
    params = {'xtick.labelsize':16,
              'ytick.labelsize':16}
    pylab.rcParams.update(params)
    plt.figure(1, figsize=(88, 55))

    # Loop over output files
    for i in range(-5, 6):
        for j in range(-5, 6):
            x0 = i * ds
            y0 = j * ds
            # Get depth
            myx = depth_pb['x'] == x0
            myy = depth_pb['y'] == y0
            myline = depth_pb[myx & myy]
            d0 = myline['depth'].iloc[0]
            # Get number of tremor and ratio
            myx = df['x0'] == x0
            myy = df['y0'] == y0
            myline = df[myx & myy]
            ntremor = myline['ntremor'].iloc[0]
            ratioE = myline['ratio_' + type_stack + '_' + cc_stack + '_EW'].iloc[0]
            ratioN = myline['ratio_' + type_stack + '_' + cc_stack + '_NS'].iloc[0]
            # Plot only best
            if ((ntremor >= mintremor) and \
                ((ratioE >= minratio) or (ratioN >= minratio))):
                # Get file
                filename = 'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_stacks.pkl'.format( \
                    arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), type_stack, cc_stack)
                # Read file
                data = pickle.load(open(filename, 'rb'))
                EW = data[0]
                NS = data[1]
                # Plot
                plt.subplot2grid((11, 11), (5 - j, i + 5))
                npts = int((EW.stats.npts - 1) / 2)
                dt = EW.stats.delta
                t = dt * np.arange(- npts, npts + 1)
                dist = sqrt(d0 ** 2.0 + x0 ** 2 + y0 ** 2)
                time = dist * (1.0 / Vs - 1.0 / Vp)
                plt.axvline(time, linewidth=4, color='grey')
                plt.plot(t, EW.data, 'r-')
                plt.plot(t, NS.data, 'b-')
                plt.xlim(0, Tmax)
                plt.ylim(0, amp)
                plt.title('Tremor at {}, {} km ({}))'.format(x0, y0, ntremor), fontsize=24)

    # Save figure
    plt.savefig('cc/{}/{}_{}_{}.eps'.format(arrayName, arrayName, type_stack, cc_stack), format='eps')
    plt.close(1)

if __name__ == '__main__':

    arrayName = 'BS'
    mintremor = 30
    Tmax = 15.0
    Vs = 3.6
    Vp = 6.4
    ds = 5.0

#    amp = plot_envelopes(arrayName, 'lin', 'lin', mintremor, 10.0, Tmax, 0.1, Vs, Vp, ds)
#    amp = plot_envelopes(arrayName, 'lin', 'pow', mintremor, 10.0, Tmax, 3.0, Vs, Vp, ds)
#    amp = plot_envelopes(arrayName, 'lin', 'PWS', mintremor, 50.0, Tmax, 0.05, Vs, Vp, ds)
#    amp = plot_envelopes(arrayName, 'pow', 'lin', mintremor, 10.0, Tmax, 0.5, Vs, Vp, ds)
#    amp = plot_envelopes(arrayName, 'pow', 'pow', mintremor, 30.0, Tmax, 10.0, Vs, Vp, ds)
#    amp = plot_envelopes(arrayName, 'pow', 'PWS', mintremor, 50.0, Tmax, 0.2, Vs, Vp, ds)
#    amp = plot_envelopes(arrayName, 'PWS', 'lin', mintremor, 15.0, Tmax, 0.05, Vs, Vp, ds)
#    amp = plot_envelopes(arrayName, 'PWS', 'pow', mintremor, 40.0, Tmax, 1.0, Vs, Vp, ds)
    amp = plot_envelopes(arrayName, 'PWS', 'PWS', mintremor, 100.0, Tmax, 0.01, Vs, Vp, ds)
