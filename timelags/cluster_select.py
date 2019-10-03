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
        n1, n2, draw_scatter=True, draw_hist=True, envelope=True, \
        draw_cc=True, draw_ac=True, draw_colored_cc=True, draw_colored_ac=True):
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
        type draw_scatter = boolean
        draw_scatter = Scatter plot of criteria for k-means clustering
        type draw_hist = boolean
        draw_hist = Histograms of time delays per cluster
        type envelope = boolean
        envelope = Do we compute the envelope before getting the maximum?
        type draw_cc = boolean
        draw_cc = Do we draw stacked cross correlation functions?
        type draw_ac = boolean
        draw_ac = Do we draw stacked autocorrelation functions?
        type draw_colored_cc = boolean
        draw_colored_cc = Do we draw colored cross correlation by cluster?
        type draw_colored_ac = boolean
        draw_colored_ac = Do we draw colored autocorrelation by cluster?
    Output:
        type clusters = 1D numpy array
        clusters = List of cluster index to which the tremor window belongs
        type t_EW = float
        t_EW = Time of maximum cross-correlation for the EW component
        type t_NS = float
        t_NS = Time of maximum cross-correlation for the NS component
        type cc_EW = float
        cc_EW = Maximum cross-correlation for the EW component
        type cc_NS = float
        cc_NS = Maximum cross-correlation for the NS component
        type ratio_EW = float
        ratio_EW = Ratio between max cc and RMS for the EW component
        type ratio_NS = float
        ratio_NS = Ratio between max cc and RMS for the NS component
        type std_EW = float
        std_EW = Standard deviation of the time lags for the EW component
        type std_NS = float
        std_NS = Standard deviation of the time lags for the NS component
    """
    # Read file containing data from stack_ccorr_tremor
    filename = 'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}.pkl'.format( \
        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), \
        type_stack)
    data = pickle.load(open(filename, 'rb'))
    EW_UD = data[6]
    NS_UD = data[7]
    # Read file containing data from stack_acorr_tremor
#    filename = 'ac/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}.pkl'.format( \
#        arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), int(y0), \
#        type_stack)
#    data = pickle.load(open(filename, 'rb'))
#    EW = data[6]
#    NS = data[7]
#    UD = data[8]
    # Stack over all tremor windows
    if (cc_stack == 'lin'):
        EW_UD_stack = linstack([EW_UD], normalize=False)[0]
        NS_UD_stack = linstack([NS_UD], normalize=False)[0]
#        EW_stack = linstack([EW], normalize=False)[0]
#        NS_stack = linstack([NS], normalize=False)[0]
#        UD_stack = linstack([UD], normalize=False)[0]
    elif (cc_stack == 'pow'):
        EW_UD_stack = powstack([EW_UD], w, normalize=False)[0]
        NS_UD_stack = powstack([NS_UD], w, normalize=False)[0]
#        EW_stack = powstack([EW], w, normalize=False)[0]
#        NS_stack = powstack([NS], w, normalize=False)[0]
#        UD_stack = powstack([UD], w, normalize=False)[0]
    elif (cc_stack == 'PWS'):
        EW_UD_stack = PWstack([EW_UD], w, normalize=False)[0]
        NS_UD_stack = PWstack([NS_UD], w, normalize=False)[0]
#        EW_stack = PWstack([EW], w, normalize=False)[0]
#        NS_stack = PWstack([NS], w, normalize=False)[0]
#        UD_stack = PWstack([UD], w, normalize=False)[0]
    else:
        raise ValueError( \
            'Type of stack must be lin, pow, or PWS')
    # Initialize indicators of cross correlation fit
    nt = len(EW_UD)
    ccmaxEW = np.zeros(nt)
    cc0EW = np.zeros(nt)
    timelagEW = np.zeros(nt)
    timedelayEW = np.zeros(nt)
    rmsEW = np.zeros(nt)
    ccmaxNS = np.zeros(nt)
    cc0NS = np.zeros(nt)
    timelagNS = np.zeros(nt)
    timedelayNS = np.zeros(nt)
    rmsNS = np.zeros(nt)
    # Windows of the cross correlation to look at
    i0 = int((len(EW_UD_stack) - 1) / 2)
    ibegin = i0 + int(Tmin / EW_UD_stack.stats.delta)
    iend = i0 + int(Tmax / EW_UD_stack.stats.delta) + 1
    rmsb = i0 + int(RMSmin / EW_UD_stack.stats.delta)
    rmse = i0 + int(RMSmax / EW_UD_stack.stats.delta) + 1
    # Time function
    dt = EW_UD_stack.stats.delta
    imax = int((EW_UD_stack.stats.npts - 1) / 2)
    t = dt * np.arange(- imax, imax + 1)
    for i in range(0, nt):
        rmsEW[i] = np.max(np.abs(EW_UD[i][ibegin:iend])) / \
            np.sqrt(np.mean(np.square(EW_UD[i][rmsb:rmse])))
        rmsNS[i] = np.max(np.abs(NS_UD[i][ibegin:iend])) / \
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
        # Time lags
        i0 = np.argmax(np.abs(EW_UD[i].data[ibegin:iend]))
        timelagEW[i] = t[ibegin:iend][i0]
        i0 = np.argmax(np.abs(NS_UD[i].data[ibegin:iend]))
        timelagNS[i] = t[ibegin:iend][i0]
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
    if (draw_scatter == True):
        colors = [palette[c] for c in clusters]
        pd.plotting.scatter_matrix(df, c=colors, figsize=(20, 20))
        plt.savefig( \
            'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_scatter.eps'. \
            format(arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), \
            int(y0), type_stack, cc_stack), format='eps')
        plt.close()
    # Compute width of timelags distribution
    timelags = pd.DataFrame({'timelagEW' : timelagEW, 'timelagNS' : timelagNS})
    width_clust_EW = []
    width_clust_NS = []
    timelag_clust_EW = []
    timelag_clust_NS = []
    for j in range(0, nc):
        times = timelags['timelagEW'].iloc[clusters == j]
        width_clust_EW.append(np.std(times))
        timelag_clust_EW.append(times)
        times = timelags['timelagNS'].iloc[clusters == j]
        width_clust_NS.append(np.std(times))
        timelag_clust_NS.append(times)
    # Save timelags into file
    filename = 'cc/{}/{}_{:03d}_{:03d}/'.format(arrayName, arrayName, \
        int(x0), int(y0)) + '{}_{:03d}_{:03d}_{}_{}_cluster_timelags.pkl'. \
        format(arrayName, int(x0), int(y0), type_stack, cc_stack)
    pickle.dump([timelag_clust_EW, timelag_clust_NS], open(filename, 'wb'))
    # Plot histogram of timelags
    if (draw_hist == True):
        plt.figure(1, figsize=(10 * nc, 16))
        # EW / Vertical
        for j in range(0, nc):
            plt.subplot2grid((2, nc), (0, j))
            times = timelags['timelagEW'].iloc[clusters == j]
            m = np.mean(times)
            s = np.std(times)
            plt.hist(times)
            plt.axvline(m + s, color='grey', linestyle='--')
            plt.axvline(m - s, color='grey', linestyle='--')
            plt.title('EW / UD - Cluster {:d} ({:d} tremor windows)'.format(j, \
                len(times)), fontsize=24)
            plt.xlabel('Time lag (s)', fontsize=24)
        # NS / Vertical
        for j in range(0, nc):
            plt.subplot2grid((2, nc), (1, j))
            times = timelags['timelagNS'].iloc[clusters == j]
            m = np.mean(times)
            s = np.std(times)
            plt.hist(times)
            plt.title('NS / UD - Cluster {:d} ({:d} tremor windows)'.format(j, \
                len(times)), fontsize=24)
            plt.axvline(m + s, color='grey', linestyle='--')
            plt.axvline(m - s, color='grey', linestyle='--')
            plt.xlabel('Lag time difference (s)', fontsize=24)
        # End figure
        plt.suptitle('{} at {} km, {} km ({} - {})'.format(arrayName, x0, y0, \
            type_stack, cc_stack), fontsize=24)
        plt.savefig( \
            'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_timelags.eps'. \
            format(arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), \
                int(y0), type_stack, cc_stack), format='eps')
        plt.close(1)
    # Plot stacked cross correlation
    if (draw_cc == True):
        plt.figure(2, figsize=(10 * nc, 16))
    # Time function
    npts = int((EW_UD_stack.stats.npts - 1) / 2)
    dt = EW_UD_stack.stats.delta
    t = dt * np.arange(- npts, npts + 1)
    # EW / Vertical
    cc_clust_EW = []
    t_clust_EW = []
    ratio_clust_EW = []
    EW_UD_stacks = Stream()
    for j in range(0, nc):
        # Stack over selected tremor windows
        EWselect = Stream()
        for i in range(0, nt):
            if (clusters[i] == j):
                EWselect.append(EW_UD[i])
        if (cc_stack == 'lin'):
            EWselect_stack = linstack([EWselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            EWselect_stack = powstack([EWselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            EWselect_stack = PWstack([EWselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        # Max cc and ratio with RMS
        if (envelope == True):
            EWselect_stack.data = obspy.signal.filter.envelope(EWselect_stack.data)
        cc_clust_EW.append(np.max(np.abs(EWselect_stack.data[ibegin:iend])))
        i0 = np.argmax(np.abs(EWselect_stack.data[ibegin:iend]))
        t_clust_EW.append(t[ibegin:iend][i0])
        RMS = np.sqrt(np.mean(np.square(EWselect_stack.data[rmsb:rmse])))
        ratio_clust_EW.append(np.max(np.abs(EWselect_stack.data[ibegin:iend])) / RMS)
        # Plot
        if (draw_cc == True):
            plt.subplot2grid((2, nc), (0, j))
            plt.plot(t, EW_UD_stack.data, 'k-', label='All')
            plt.plot(t, EWselect_stack.data, color=palette[j], \
                label='Cluster {:d}'.format(j))
            plt.xlim(0, xmax)
            plt.ylim(- ymax, ymax)
            plt.title('EW / UD - Cluster {:d} ({:d} tremor windows)'.format(j, \
                len(EWselect)), fontsize=24)
            plt.xlabel('Lag time (s)', fontsize=24)
            plt.legend(loc=1)
        # Save into stream
        EW_UD_stacks.append(EWselect_stack)
    # Get the best stack
    i0 = cc_clust_EW.index(max(cc_clust_EW))
    t_EW = t_clust_EW[i0]
    cc_EW = max(cc_clust_EW)
    ratio_EW = ratio_clust_EW[i0]
    width_EW = width_clust_EW[i0]
    stack_EW = EW_UD_stacks[i0]
    # NS / Vertical
    cc_clust_NS = []
    t_clust_NS = []
    ratio_clust_NS = []
    NS_UD_stacks = Stream()
    for j in range(0, nc):
        # Stack over selected tremor windows
        NSselect = Stream()
        for i in range(0, nt):
            if (clusters[i] == j):
                NSselect.append(NS_UD[i])
        if (cc_stack == 'lin'):
            NSselect_stack = linstack([NSselect], normalize=False)[0]
        elif (cc_stack == 'pow'):
            NSselect_stack = powstack([NSselect], w, normalize=False)[0]
        elif (cc_stack == 'PWS'):
            NSselect_stack = PWstack([NSselect], w, normalize=False)[0]
        else:
            raise ValueError( \
                'Type of stack must be lin, pow, or PWS')
        # Max cc and ratio with RMS
        if (envelope == True):
            NSselect_stack.data = obspy.signal.filter.envelope(NSselect_stack.data)
        cc_clust_NS.append(np.max(np.abs(NSselect_stack.data[ibegin:iend])))
        i0 = np.argmax(np.abs(NSselect_stack.data[ibegin:iend]))
        t_clust_NS.append(t[ibegin:iend][i0])
        RMS = np.sqrt(np.mean(np.square(NSselect_stack.data[rmsb:rmse])))
        ratio_clust_NS.append(np.max(np.abs(NSselect_stack.data[ibegin:iend])) \
            / RMS)  
        # Plot
        if (draw_cc == True):
            plt.subplot2grid((2, nc), (1, j))
            plt.plot(t, NS_UD_stack.data, 'k-', label='All')
            plt.plot(t, NSselect_stack.data, color=palette[j], \
                label='Cluster {:d}'.format(j, ))
            plt.xlim(0, xmax)
            plt.ylim(- ymax, ymax)
            plt.title('NS / UD - Cluster {:d} ({:d} tremor windows)'.format(j, \
                len(NSselect)), fontsize=24)
            plt.xlabel('Lag time (s)', fontsize=24)
            plt.legend(loc=1)
        # Save into stream
        NS_UD_stacks.append(NSselect_stack)
    # Get the best stack
    i0 = cc_clust_NS.index(max(cc_clust_NS))
    t_NS = t_clust_NS[i0]
    cc_NS = max(cc_clust_NS)
    ratio_NS = ratio_clust_NS[i0]
    width_NS = width_clust_NS[i0]
    stack_NS = NS_UD_stacks[i0]
    # End figure
    if (draw_cc == True):
        plt.suptitle('{} at {} km, {} km ({} - {})'.format(arrayName, x0, y0, \
            type_stack, cc_stack), fontsize=24)
        plt.savefig( \
            'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_stackcc.eps'. \
            format(arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), \
                int(y0), type_stack, cc_stack), format='eps')
        plt.close(2)
    # Save best stacks into file
    filename = 'cc/{}/{}_{:03d}_{:03d}/'.format(arrayName, arrayName, \
        int(x0), int(y0)) + '{}_{:03d}_{:03d}_{}_{}_cluster_stacks.pkl'. \
        format(arrayName, int(x0), int(y0), type_stack, cc_stack)
    pickle.dump([stack_EW, stack_NS], open(filename, 'wb'))
    # Plot stacked autocorrelation
    if (draw_ac == True):
        plt.figure(3, figsize=(10 * nc, 24))
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
        plt.savefig( \
            'ac/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_stackac.eps'. \
            format(arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), \
            int(y0), type_stack, cc_stack), format='eps')
        plt.close(3)
    # Plot colored cross correlation windows
    if (draw_colored_cc == True):
        plt.figure(4, figsize=(20, 16))
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
        plt.savefig( \
            'cc/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_ccwin.eps'. \
            format(arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), \
            int(y0), type_stack, cc_stack), format='eps')
        ax1.clear()
        ax2.clear()
        plt.close(4)
    # Plot colored autocorrelation windows
    if (draw_colored_ac == True):
        plt.figure(5, figsize=(20, 24))
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
        plt.savefig( \
            'ac/{}/{}_{:03d}_{:03d}/{}_{:03d}_{:03d}_{}_{}_cluster_acwin.eps'. \
            format(arrayName, arrayName, int(x0), int(y0), arrayName, int(x0), \
            int(y0), type_stack, cc_stack), format='eps')
        ax1.clear()
        ax2.clear()
        ax3.clear()
        plt.close(5)
    return (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, \
        width_EW, width_NS)

if __name__ == '__main__':

    # Set the parameters
    arrayName = 'BS'
    x0 = 0.0
    y0 = 0.0
    w = 2.0
    ncor = 40
    Tmin = 4.0
    Tmax = 6.0
    RMSmin = 12.0
    RMSmax = 14.0
    xmax = 15.0
    nc = 2
    palette = {0: 'tomato', 1: 'royalblue', 2:'forestgreen', 3:'gold', \
        4: 'lightpink', 5:'skyblue'}
    n1 = 0
    n2 = 82
    draw_scatter = True
    draw_hist = True
    envelope = True
    draw_cc = True
    draw_ac = False
    draw_colored_cc = True
    draw_colored_ac = False

    # Linear stack
    amp = 10.0
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'lin', w, 'lin', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 0.06, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'lin', w, 'pow', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 2.0, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'lin', w, 'PWS', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 0.04, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)

    # Power stack
    amp = 2.0
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'pow', w, 'lin', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 0.3, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'pow', w, 'pow', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 10.0, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'pow', w, 'PWS', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 0.16, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)

    # Phase-weighted stack
    amp = 20.0
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'PWS', w, 'lin', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 0.02, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'PWS', w, 'pow', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 0.4, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)
    (clusters, t_EW, t_NS, cc_EW, cc_NS, ratio_EW, ratio_NS, std_EW, std_NS) = \
        cluster_select(arrayName, x0, y0, 'PWS', w, 'PWS', ncor, Tmin, Tmax, \
        RMSmin, RMSmax, xmax, 0.01, 'kmeans', nc, palette, amp, n1, n2, \
        draw_scatter, draw_hist, envelope, draw_cc, draw_ac, draw_colored_cc, draw_colored_ac)
