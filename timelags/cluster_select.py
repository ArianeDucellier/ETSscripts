"""
This module contains a function that sorts the tremor windows into several
clusters and stacks the cross correlations and the autocorrelations inside
each cluster. We also plot the tremor windows color-coded for the cluster
"""

import obspy
from obspy.core.stream import Stream
from obspy.signal.cross_correlation import correlate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering, KMeans

from stacking import linstack, powstack, PWstack

def cluster_select(arrayName, x0, y0, type_stack, w, cc_stack, ncor, Tmin, \
        Tmax, RMSmin, RMSmax, xmax, ymax, typecluster, nc, palette, amp, \
        n1, n2):
    """
    Sort the tremor windows into several clusters and stack cross and
    autocorrelation inside each cluster

    Input:
        type arrayName = string
        arrayName = Name of seismic array
        type x0 = float
        x0 = Distance of the center of the cell from the array (east)
        type y0 = float
        y0 = Distance of the center of the cell from the array (north)
        type type_stack = string
        type_stack = Type of stack ('lin', 'pow', 'PWS')
        type w = float
        w = Power of the stack (for 'pow' and 'PWS')
        type cc_stack = string
        cc_stack = Type of stack ('lin', 'pow', 'PWS') over tremor windows
        type ncor = integer
        ncor = Number of points for the cross correlation with the stack
        type Tmin = float
        Tmin = Minimum time lag for comparing cross correlation with the stack
        type Tmax = float
        Tmax = Maximum time lag for comparing cross correlation with the stack
        type RMSmin = float
        RMSmin = Minimum time lag to compute the RMS
        type RMSmax = float
        RMSmax = Maximum time lag to compute the RMS
        type xmax = float
        xmax = Horizontal axis limit for plot
        type ymax = float
        ymax = Vertical axis limit for plot
        type typecluster = string
        typecluster = Clustering method ('kmeans' or 'agglo')
        type nc = integer
        nc = Number of clusters
        type palette = dictionary
        palette = Names of colors to plot the clusters
        type amp = float
        amp = Amplification factor of cross correlation for plotting
        type n1 = integer
        n1 = Index of first tremor to be plotted
        type n2 = integer
        n2 = Index of last tremor to be plotted
    Output:
        type clusters = 1D numpy array
        clusters = List of cluster index to which the tremor window belongs
    """
    # Read file containing data from stack_ccorr_tremor
    filename = 'cc/{}_{:03d}_{:03d}_{}.pkl'.format(arrayName, int(x0), \
        int(y0), type_stack)
    data = pickle.load(open(filename, 'rb'))
    EW_UD = data[6]
    NS_UD = data[7]
    # Read file containing data from stack_acorr_tremor
    filename = 'ac/{}_{:03d}_{:03d}_{}.pkl'.format(arrayName, int(x0), \
        int(y0), type_stack)
    data = pickle.load(open(filename, 'rb'))
    EW = data[6]
    NS = data[7]
    UD = data[8]
    # Stack over all tremor windows
    if (cc_stack == 'lin'):
        EW_UD_stack = linstack([EW_UD], normalize=False)[0]
        NS_UD_stack = linstack([NS_UD], normalize=False)[0]
        EW_stack = linstack([EW], normalize=False)[0]
        NS_stack = linstack([NS], normalize=False)[0]
        UD_stack = linstack([UD], normalize=False)[0]
    elif (cc_stack == 'pow'):
        EW_UD_stack = powstack([EW_UD], w, normalize=False)[0]
        NS_UD_stack = powstack([NS_UD], w, normalize=False)[0]
        EW_stack = powstack([EW], w, normalize=False)[0]
        NS_stack = powstack([NS], w, normalize=False)[0]
        UD_stack = powstack([UD], w, normalize=False)[0]
    elif (cc_stack == 'PWS'):
        EW_UD_stack = PWstack([EW_UD], w, normalize=False)[0]
        NS_UD_stack = PWstack([NS_UD], w, normalize=False)[0]
        EW_stack = PWstack([EW], w, normalize=False)[0]
        NS_stack = PWstack([NS], w, normalize=False)[0]
        UD_stack = PWstack([UD], w, normalize=False)[0]
    else:
        raise ValueError( \
            'Type of stack must be lin, pow, or PWS')
    # Initialize indicators of cross correlation fit
    nt = len(EW_UD)
    ccmaxEW = np.zeros(nt)
    cc0EW = np.zeros(nt)
    timedelayEW = np.zeros(nt)
    rmsEW = np.zeros(nt)
    ccmaxNS = np.zeros(nt)
    cc0NS = np.zeros(nt)
    timedelayNS = np.zeros(nt)
    rmsNS = np.zeros(nt)
    # Windows of the cross correlation to look at
    i0 = int((len(EW_UD_stack) - 1) / 2)
    ibegin = i0 + int(Tmin / EW_UD_stack.stats.delta)
    iend = i0 + int(Tmax / EW_UD_stack.stats.delta) + 1
    rmsb = i0 + int(RMSmin / EW_UD_stack.stats.delta)
    rmse = i0 + int(RMSmax / EW_UD_stack.stats.delta) + 1
    for i in range(0, nt):
        rmsEW[i] = np.max(np.abs(EW_UD[i][ibegin : iend])) / \
            np.sqrt(np.mean(np.square(EW_UD[i][rmsb:rmse])))       
        rmsNS[i] = np.max(np.abs(NS_UD[i][ibegin : iend])) / \
            np.sqrt(np.mean(np.square(NS_UD[i][rmsb:rmse])))
        # Cross correlate cc for EW with stack       
        cc_EW = correlate(EW_UD[i][ibegin : iend], \
            EW_UD_stack[ibegin : iend], ncor)
        ccmaxEW[i] = np.max(cc_EW)
        cc0EW[i] = cc_EW[ncor]
        timedelayEW[i] = (np.argmax(cc_EW) - ncor) * EW_UD_stack.stats.delta
        # Cross correlate cc for NS with stack
        cc_NS = correlate(NS_UD[i][ibegin : iend], \
            NS_UD_stack[ibegin : iend], ncor)
        ccmaxNS[i] = np.max(cc_NS)
        cc0NS[i] = cc_NS[ncor]
        timedelayNS[i] = (np.argmax(cc_NS) - ncor) * NS_UD_stack.stats.delta
    # Clustering
    df = pd.DataFrame({'ccmaxEW' : ccmaxEW, 'ccmaxNS' : ccmaxNS, \
        'cc0EW' : cc0EW, 'cc0NS' : cc0NS, 'timedelayEW' : timedelayEW, \
        'timedelayNS' : timedelayNS, 'rmsEW' : rmsEW, 'rmsNS' : rmsNS})
    df = preprocessing.scale(df)
    df = pd.DataFrame(df)
    df.columns = ['ccmaxEW', 'ccmaxNS', 'cc0EW', 'cc0NS', 'timedelayEW', \
        'timedelayNS', 'rmsEW', 'rmsNS']
    if (typecluster == 'kmeans'):
        clusters = KMeans(n_clusters=nc, random_state=0).fit_predict(df)
    elif (typecluster == 'agglo'):
        clustering = AgglomerativeClustering(n_clusters=nc).fit(df)
        clusters = clustering.labels_
    else:
        raise ValueError( \
            'Type of clustering must be kmeans or agglo')
    # Scatter plot
    colors = [palette[c] for c in clusters]
    pd.plotting.scatter_matrix(df, c=colors, figsize=(20, 20))
    plt.savefig('cc/{}_{:03d}_{:03d}_{}_{}_cluster_scatter.eps'.format( \
        arrayName, int(x0), int(y0), type_stack, cc_stack), format='eps')
    plt.close()
    # Plot cross correlation
    plt.figure(1, figsize=(10 * nc, 16))
    npts = int((EW_UD_stack.stats.npts - 1) / 2)
    dt = EW_UD_stack.stats.delta
    t = dt * np.arange(- npts, npts + 1)
    # EW / Vertical
    for j in range(0, nc):
        plt.subplot2grid((2, nc), (0, j))
        plt.plot(t, EW_UD_stack.data, 'k-', label='All')
        EWselect = Stream()
        for i in range(0, nt):
            if (clusters[i] == j):
                EWselect.append(EW_UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            EWselect_stack = linstack([EWselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            EWselect_stack = powstack([EWselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            EWselect_stack = PWstack([EWselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, EWselect_stack.data, color=palette[j], \
            label='Cluster {:d}'.format(j))
        plt.xlim(0, xmax)
        plt.ylim(- ymax, ymax)
        plt.title('EW / UD - Cluster {:d} ({:d} tremor windows)'.format(j, \
            len(EWselect)), fontsize=24)
        plt.xlabel('Lag time (s)', fontsize=24)
        plt.legend(loc=1)
    # NS / Vertical
    for j in range(0, nc):
        plt.subplot2grid((2, nc), (1, j))
        plt.plot(t, NS_UD_stack.data, 'k-', label='All')
        NSselect = Stream()
        for i in range(0, nt):
            if (clusters[i] == j):
                NSselect.append(NS_UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            NSselect_stack = linstack([NSselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            NSselect_stack = powstack([NSselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            NSselect_stack = PWstack([NSselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, NSselect_stack.data, color=palette[j], \
            label='Cluster {:d}'.format(j, ))
        plt.xlim(0, xmax)
        plt.ylim(- ymax, ymax)
        plt.title('NS / UD - Cluster {:d} ({:d} tremor windows)'.format(j, \
            len(NSselect)), fontsize=24)
        plt.xlabel('Lag time (s)', fontsize=24)
        plt.legend(loc=1)
    # End figure
    plt.suptitle('{} at {} km, {} km ({} - {})'.format(arrayName, x0, y0, \
        type_stack, cc_stack), fontsize=24)
    plt.savefig('cc/{}_{:03d}_{:03d}_{}_{}_cluster_stackcc.eps'.format( \
        arrayName, int(x0), int(y0), type_stack, cc_stack), format='eps')
    plt.close(1)
    # Plot autocorrelation
    plt.figure(2, figsize=(10 * nc, 24))
    npts = int((EW_stack.stats.npts - 1) / 2)
    dt = EW_stack.stats.delta
    t = dt * np.arange(- npts, npts + 1)
    # EW
    for j in range(0, nc):
        plt.subplot2grid((3, nc), (0, j))
        plt.plot(t, EW_stack.data, 'k-', label='All')
        EWselect = Stream()
        for i in range(0, nt):
            if (clusters[i] == j):
                EWselect.append(EW[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            EWselect_stack = linstack([EWselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            EWselect_stack = powstack([EWselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            EWselect_stack = PWstack([EWselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, EWselect_stack.data, color=palette[j], \
            label='Cluster {:d}'.format(j))
        plt.xlim(0, xmax)
        plt.ylim(- ymax, ymax)
        plt.title('EW - Cluster {:d} ({:d} tremor windows)'.format(j, \
            len(EWselect)), fontsize=24)
        plt.xlabel('Lag time (s)', fontsize=24)
        plt.legend(loc=1)
    # NS
    for j in range(0, nc):
        plt.subplot2grid((3, nc), (1, j))
        plt.plot(t, NS_stack.data, 'k-', label='All')
        NSselect = Stream()
        for i in range(0, nt):
            if (clusters[i] == j):
                NSselect.append(NS[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            NSselect_stack = linstack([NSselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            NSselect_stack = powstack([NSselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            NSselect_stack = PWstack([NSselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, NSselect_stack.data, color=palette[j], \
            label='Cluster {:d}'.format(j))
        plt.xlim(0, xmax)
        plt.ylim(- ymax, ymax)
        plt.title('NS - Cluster {:d} ({:d} tremor windows)'.format(j, \
            len(NSselect)), fontsize=24)
        plt.xlabel('Lag time (s)', fontsize=24)
        plt.legend(loc=1)
    # UD
    for j in range(0, nc):
        plt.subplot2grid((3, nc), (2, j))
        plt.plot(t, UD_stack.data, 'k-', label='All')
        UDselect = Stream()
        for i in range(0, nt):
            if (clusters[i] == j):
                UDselect.append(UD[i])
        # Stack over selected tremor windows
        if (cc_stack == 'lin'):
            UDselect_stack = linstack([UDselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            UDselect_stack = powstack([UDselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            UDselect_stack = PWstack([UDselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        plt.plot(t, UDselect_stack.data, color=palette[j], \
            label='Cluster {:d}'.format(j))
        plt.xlim(0, xmax)
        plt.ylim(- ymax, ymax)
        plt.title('UD - Cluster {:d} ({:d} tremor windows)'.format(j, \
            len(UDselect)), fontsize=24)
        plt.xlabel('Lag time (s)', fontsize=24)
        plt.legend(loc=1)
    # End figure
    plt.suptitle('{} at {} km, {} km ({} - {})'.format(arrayName, x0, y0, \
        type_stack, cc_stack), fontsize=24)
    plt.savefig('ac/{}_{:03d}_{:03d}_{}_{}_cluster_stackac.eps'.format( \
        arrayName, int(x0), int(y0), type_stack, cc_stack), format='eps')
    plt.close(2)
    # Plot colored cross correlation windows
    plt.figure(3, figsize=(20, 16))
    # EW - UD cross correlation
    ax1 = plt.subplot(121)
    for i in range(n1, n2):
        dt = EW_UD[i].stats.delta
        ncor = int((EW_UD[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW_UD[i].data, \
            color=colors[i])
    plt.xlim(0, xmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('East / Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Cross correlation', fontsize=24)
    ax1.set_yticklabels([])
    ax1.tick_params(labelsize=20)
    # NS - UD cross correlation
    ax2 = plt.subplot(122)
    for i in range(n1, n2):
        dt = NS_UD[i].stats.delta
        ncor = int((NS_UD[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS_UD[i].data, \
            color=colors[i])
    plt.xlim(0, xmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('North / Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Cross correlation', fontsize=24)
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=20)
    # End figure
    plt.suptitle('{} at {} km, {} km'.format(arrayName, x0, y0), fontsize=24)
    plt.savefig('cc/{}_{:03d}_{:03d}_{}_{}_cluster_ccwin.eps'.format( \
        arrayName, int(x0), int(y0), type_stack, cc_stack), format='eps')
    ax1.clear()
    ax2.clear()
    plt.close(3)
    # Plot colored autocorrelation windows
    plt.figure(4, figsize=(20, 24))
    # EW autocorrelation
    ax1 = plt.subplot(131)
    for i in range(n1, n2):
        dt = EW[i].stats.delta
        ncor = int((EW[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * EW[i].data, color=colors[i])
    plt.xlim(0, xmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('East component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax1.set_yticklabels([])
    ax1.tick_params(labelsize=20)
    # NS autocorrelation
    ax2 = plt.subplot(132)
    for i in range(n1, n2):
        dt = NS[i].stats.delta
        ncor = int((NS[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * NS[i].data, color=colors[i])
    plt.xlim(0, xmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('North component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax2.set_yticklabels([])
    ax2.tick_params(labelsize=20)
    # UD autocorrelation
    ax3 = plt.subplot(133)
    for i in range(n1, n2):
        dt = UD[i].stats.delta
        ncor = int((UD[i].stats.npts - 1) / 2)
        t = dt * np.arange(- ncor, ncor + 1)
        plt.plot(t, (2.0 * i + 1) - 2 * n1 + amp * UD[i].data, color=colors[i])
    plt.xlim(0, xmax)
    plt.ylim(0.0, 2.0 * (n2 - n1))
    plt.title('Vertical component', fontsize=24)
    plt.xlabel('Lag time (s)', fontsize=24)
    plt.ylabel('Autocorrelation', fontsize=24)
    ax3.set_yticklabels([])
    ax3.tick_params(labelsize=20)
    # End figure and plot
    plt.suptitle('{} at {} km, {} km'.format(arrayName, x0, y0), fontsize=24)
    plt.savefig('ac/{}_{:03d}_{:03d}_{}_{}_cluster_acwin.eps'.format( \
        arrayName, int(x0), int(y0), type_stack, cc_stack), format='eps')
    ax1.clear()
    ax2.clear()
    ax3.clear()
    plt.close(4)
    return clusters

if __name__ == '__main__':

    # Set the parameters
    arrayName = 'BS'
    x0 = 5.0
    y0 = 5.0
    w = 2.0
    ncor = 40
    Tmin = 4.5
    Tmax = 6.5
    RMSmin = 12.0
    RMSmax = 14.0
    xmax = 15.0
    nc = 2
    palette = {0: 'tomato', 1: 'royalblue', 2:'forestgreen', 3:'gold', \
        4: 'lightpink', 5:'skyblue'}
    n1 = 0
    n2 = 63

    # Linear stack
    amp = 10.0
    clusters = cluster_select(arrayName, x0, y0, 'lin', w, 'lin', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 0.1, 'kmeans', nc, palette, amp, \
        n1, n2)
    clusters = cluster_select(arrayName, x0, y0, 'lin', w, 'pow', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 0.2, 'kmeans', nc, palette, amp, \
        n1, n2)
    clusters = cluster_select(arrayName, x0, y0, 'lin', w, 'PWS', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 0.05, 'kmeans', nc, palette, amp, \
        n1, n2)

    # Power stack
    amp = 2.0
    clusters = cluster_select(arrayName, x0, y0, 'pow', w, 'lin', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 0.2, 'kmeans', nc, palette, amp, \
        n1, n2)
    clusters = cluster_select(arrayName, x0, y0, 'pow', w, 'pow', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 1.0, 'kmeans', nc, palette, amp, \
        n1, n2)
    clusters = cluster_select(arrayName, x0, y0, 'pow', w, 'PWS', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 0.15, 'kmeans', nc, palette, amp, \
        n1, n2)

    # Phase-weighted stack
    amp = 20.0
    clusters = cluster_select(arrayName, x0, y0, 'PWS', w, 'lin', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 0.02, 'kmeans', nc, palette, amp, \
        n1, n2)
    clusters = cluster_select(arrayName, x0, y0, 'PWS', w, 'pow', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 0.2, 'kmeans', nc, palette, amp, \
        n1, n2)
    clusters = cluster_select(arrayName, x0, y0, 'PWS', w, 'PWS', ncor, \
        Tmin, Tmax, RMSmin, RMSmax, xmax, 0.01, 'kmeans', nc, palette, amp, \
        n1, n2)