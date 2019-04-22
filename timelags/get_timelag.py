"""
Script to compute the time lag between the direct P-wave
and the direct S-wave using cross-correlation of seicmic
signal recorded during tectonic tremor
"""

import numpy as np
import obspy
import os
import pandas as pd
import pickle

from cluster_select import cluster_select
from plot_stack_acorr import plot_stack_acorr
from plot_stack_ccorr import plot_stack_ccorr
from stacking import linstack, powstack, PWstack

# Set parameters
arrayName = 'BS'
w = 2.0
Tmax = 15.0
ds = 5.0
tbegin = 3.8
tend = 7.6
ncor_cluster = 40
RMSmin = 12.0
RMSmax = 14.0
xmax = 15.0
nc = 2
palette = {0: 'tomato', 1: 'royalblue', 2:'forestgreen', 3:'gold', \
    4: 'lightpink', 5:'skyblue'}

# Loop on tremor location
for i in range(5, 6):
    for j in range(5, 6):
        x0 = i * ds
        y0 = j * ds
        filename = '{}_{:03d}_{:03d}/{}_{:03d}_{:03d}'.format( \
            arrayName, int(x0), int(y0), arrayName, int(x0), int(y0))

        # Get the number of tremor
        data = pickle.load(open('cc/' + filename + '_lin.pkl', 'rb'))
        nlincc = len(data[6])
        data = pickle.load(open('cc/' + filename + '_pow.pkl', 'rb'))
        npowcc = len(data[6])
        data = pickle.load(open('cc/' + filename + '_PWS.pkl', 'rb'))
        nPWScc = len(data[6])
        data = pickle.load(open('ac/' + filename + '_lin.pkl', 'rb'))
        nlinac = len(data[6])
        data = pickle.load(open('ac/' + filename + '_pow.pkl', 'rb'))
        npowac = len(data[6])
        data = pickle.load(open('ac/' + filename + '_PWS.pkl', 'rb'))
        nPWSac = len(data[6])

        # If there are tremor at this location
        if ((nlincc == npowcc) and (nlincc == nPWScc) and \
            (nlincc == nlinac) and (nlincc == npowac) and (nlincc == nPWSac)):
            ntremor = nlincc
            n1 = 0
            n2 = ntremor
            EW_UD = data[6]
            dt = EW_UD[0].stats.delta
            ncor = int((EW_UD[0].stats.npts - 1) / 2)
            t = dt * np.arange(- ncor, ncor + 1)
            ibegin = int(ncor + tbegin / dt)
            iend = int(ncor + tend / dt)

            # Plot auto and cross-correlation for linear stack
            type_stack = 'lin'
            amp = 10.0
            amp_lin = 100.0
            amp_pow = 2.0
            amp_PWS = 200.0
            plot_stack_ccorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
                amp_pow, amp_PWS, n1, n2)
            plot_stack_acorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
                amp_pow, amp_PWS, n1, n2)

            # Plot auto and cross-correlation for power stack
            type_stack = 'pow'
            amp = 2.0
            amp_lin = 15.0
            amp_pow = 1.0
            amp_PWS = 100.0
            plot_stack_ccorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
                amp_pow, amp_PWS, n1, n2)
            plot_stack_acorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
                amp_pow, amp_PWS, n1, n2)

            # Plot auto and cross-correlation for phase-weighted stack
            type_stack = 'PWS'
            amp = 20.0
            amp_lin = 200.0
            amp_pow = 10.0
            amp_PWS = 1000.0
            plot_stack_ccorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
                amp_pow, amp_PWS, n1, n2)
            plot_stack_acorr(arrayName, x0, y0, type_stack, w, Tmax, amp, amp_lin, \
                amp_pow, amp_PWS, n1, n2)

            # Find time of maximum cross-correlation
            # Linear stack
            data = pickle.load(open('cc/' + filename + '_lin.pkl', 'rb'))
            EW_UD = data[6]
            NS_UD = data[7]
            EW_lin = linstack([EW_UD], normalize=False)[0]
            NS_lin = linstack([NS_UD], normalize=False)[0]
            i0 = np.argmax(np.abs(EW_lin.data[ibegin:iend]))
            t_lin_lin_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_lin.data[ibegin:iend]))
            t_lin_lin_NS = t[ibegin:iend][i0]

            EW_pow = powstack([EW_UD], w, normalize=False)[0]
            NS_pow = powstack([NS_UD], w, normalize=False)[0]
            i0 = np.argmax(np.abs(EW_pow.data[ibegin:iend]))
            t_lin_pow_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_pow.data[ibegin:iend]))
            t_lin_pow_NS = t[ibegin:iend][i0]
            
            EW_PWS = PWstack([EW_UD], w, normalize=False)[0]
            NS_PWS = PWstack([NS_UD], w, normalize=False)[0]
            i0 = np.argmax(np.abs(EW_PWS.data[ibegin:iend]))
            t_lin_PWS_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_PWS.data[ibegin:iend]))
            t_lin_PWS_NS = t[ibegin:iend][i0]

            # Power stack
            data = pickle.load(open('cc/' + filename + '_pow.pkl', 'rb'))
            EW_UD = data[6]
            NS_UD = data[7]
            EW_lin = linstack([EW_UD], normalize=False)[0]
            NS_lin = linstack([NS_UD], normalize=False)[0]
            i0 = np.argmax(np.abs(EW_lin.data[ibegin:iend]))
            t_pow_lin_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_lin.data[ibegin:iend]))
            t_pow_lin_NS = t[ibegin:iend][i0]

            EW_pow = powstack([EW_UD], w, normalize=False)[0]
            NS_pow = powstack([NS_UD], w, normalize=False)[0]
            i0 = np.argmax(np.abs(EW_pow.data[ibegin:iend]))
            t_pow_pow_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_pow.data[ibegin:iend]))
            t_pow_pow_NS = t[ibegin:iend][i0]
            
            EW_PWS = PWstack([EW_UD], w, normalize=False)[0]
            NS_PWS = PWstack([NS_UD], w, normalize=False)[0]
            i0 = np.argmax(np.abs(EW_PWS.data[ibegin:iend]))
            t_pow_PWS_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_PWS.data[ibegin:iend]))
            t_pow_PWS_NS = t[ibegin:iend][i0]

            # Phase-weighted stack
            data = pickle.load(open('cc/' + filename + '_PWS.pkl', 'rb'))
            EW_UD = data[6]
            NS_UD = data[7]
            EW_lin = linstack([EW_UD], normalize=False)[0]
            NS_lin = linstack([NS_UD], normalize=False)[0]
            i0 = np.argmax(np.abs(EW_lin.data[ibegin:iend]))
            t_PWS_lin_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_lin.data[ibegin:iend]))
            t_PWS_lin_NS = t[ibegin:iend][i0]

            EW_pow = powstack([EW_UD], w, normalize=False)[0]
            NS_pow = powstack([NS_UD], w, normalize=False)[0]
            i0 = np.argmax(np.abs(EW_pow.data[ibegin:iend]))
            t_PWS_pow_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_pow.data[ibegin:iend]))
            t_PWS_pow_NS = t[ibegin:iend][i0]
            
            EW_PWS = PWstack([EW_UD], w, normalize=False)[0]
            NS_PWS = PWstack([NS_UD], w, normalize=False)[0]
            i0 = np.argmax(np.abs(EW_PWS.data[ibegin:iend]))
            t_PWS_PWS_EW = t[ibegin:iend][i0]
            i0 = np.argmax(np.abs(NS_PWS.data[ibegin:iend]))
            t_PWS_PWS_NS = t[ibegin:iend][i0]
 
            Tmin = min([t_lin_lin_EW, t_lin_pow_EW, t_lin_PWS_EW, \
                        t_lin_lin_NS, t_lin_pow_NS, t_lin_PWS_NS, \
                        t_pow_lin_EW, t_pow_pow_EW, t_pow_PWS_EW, \
                        t_pow_lin_NS, t_pow_pow_NS, t_pow_PWS_NS, \
                        t_PWS_lin_EW, t_PWS_pow_EW, t_PWS_PWS_EW, \
                        t_PWS_lin_NS, t_PWS_pow_NS, t_PWS_PWS_NS]) - 1.0
            Tmax = max([t_lin_lin_EW, t_lin_pow_EW, t_lin_PWS_EW, \
                        t_lin_lin_NS, t_lin_pow_NS, t_lin_PWS_NS, \
                        t_pow_lin_EW, t_pow_pow_EW, t_pow_PWS_EW, \
                        t_pow_lin_NS, t_pow_pow_NS, t_pow_PWS_NS, \
                        t_PWS_lin_EW, t_PWS_pow_EW, t_PWS_PWS_EW, \
                        t_PWS_lin_NS, t_PWS_pow_NS, t_PWS_PWS_NS]) + 1.0

            # Cluster tremor for better peak
            # Linear stack
            amp = 10.0
            (clusters, t_lin_lin_EW_cluster, t_lin_lin_NS_cluster, \
                 cc_lin_lin_EW_cluster, cc_lin_lin_NS_cluster) = \
                 cluster_select(arrayName, x0, y0, 'lin', w, 'lin', ncor_cluster, \
                 Tmin, Tmax, RMSmin, RMSmax, xmax, 0.1, 'kmeans', nc, palette, amp, \
                 n1, n2)
            (clusters, t_lin_pow_EW_cluster, t_lin_pow_NS_cluster, \
                cc_lin_pow_EW_cluster, cc_lin_pow_NS_cluster) = \
                cluster_select(arrayName, x0, y0, 'lin', w, 'pow', ncor, \
                Tmin, Tmax, RMSmin, RMSmax, xmax, 0.2, 'kmeans', nc, palette, amp, \
                n1, n2)
            (clusters, t_lin_PWS_EW_cluster, t_lin_PWS_NS_cluster, \
                cc_lin_PWS_EW_cluster, cc_lin_PWS_NS_cluster) = \
                cluster_select(arrayName, x0, y0, 'lin', w, 'PWS', ncor, \
                Tmin, Tmax, RMSmin, RMSmax, xmax, 0.05, 'kmeans', nc, palette, amp, \
                n1, n2)

            # Power stack
            amp = 2.0
            (clusters, t_pow_lin_EW_cluster, t_pow_lin_NS_cluster, \
                cc_pow_lin_EW_cluster, cc_pow_lin_NS_cluster) = \
                cluster_select(arrayName, x0, y0, 'pow', w, 'lin', ncor, \
                Tmin, Tmax, RMSmin, RMSmax, xmax, 0.2, 'kmeans', nc, palette, amp, \
                n1, n2)
            (clusters, t_pow_pow_EW_cluster, t_pow_pow_NS_cluster, \
                cc_pow_pow_EW_cluster, cc_pow_pow_NS_cluster) = \
                cluster_select(arrayName, x0, y0, 'pow', w, 'pow', ncor, \
                Tmin, Tmax, RMSmin, RMSmax, xmax, 1.0, 'kmeans', nc, palette, amp, \
                n1, n2)
            (clusters, t_pow_PWS_EW_cluster, t_pow_PWS_NS_cluster, 
                cc_pow_PWS_EW_cluster, cc_pow_PWS_NS_cluster) = \
                cluster_select(arrayName, x0, y0, 'pow', w, 'PWS', ncor, \
                Tmin, Tmax, RMSmin, RMSmax, xmax, 0.15, 'kmeans', nc, palette, amp, \
                n1, n2)

            # Phase-weighted stack
            amp = 20.0
            (clusters, t_PWS_lin_EW_cluster, t_PWS_lin_NS_cluster, \
                cc_PWS_lin_EW_cluster, cc_PWS_lin_NS_cluster) = \
                cluster_select(arrayName, x0, y0, 'PWS', w, 'lin', ncor, \
                Tmin, Tmax, RMSmin, RMSmax, xmax, 0.02, 'kmeans', nc, palette, amp, \
                n1, n2)
            (clusters, t_PWS_pow_EW_cluster, t_PWS_pow_NS_cluster, \
                cc_PWS_pow_EW_cluster, cc_PWS_pow_NS_cluster) = \
                cluster_select(arrayName, x0, y0, 'PWS', w, 'pow', ncor, \
                Tmin, Tmax, RMSmin, RMSmax, xmax, 0.2, 'kmeans', nc, palette, amp, \
                n1, n2)
            (clusters, t_PWS_PWS_EW_cluster, t_PWS_PWS_NS_cluster, \
                cc_PWS_PWS_EW_cluster, cc_PWS_PWS_NS_cluster) = \
                cluster_select(arrayName, x0, y0, 'PWS', w, 'PWS', ncor, \
                Tmin, Tmax, RMSmin, RMSmax, xmax, 0.01, 'kmeans', nc, palette, amp, \
                n1, n2)

            # Store results in pandas dataframe
            namefile = arrayName + '_timelag.pkl'
            if os.path.exists(namefile):
                df = pickle.load(open(namefile, 'rb'))
            else:
                df = pd.DataFrame(columns=['x0', 'y0', 'ntremor', \
                    't_lin_lin_EW', 't_lin_pow_EW', 't_lin_PWS_EW', \
                    't_lin_lin_NS', 't_lin_pow_NS', 't_lin_PWS_NS', \
                    't_pow_lin_EW', 't_pow_pow_EW', 't_pow_PWS_EW', \
                    't_pow_lin_NS', 't_pow_pow_NS', 't_pow_PWS_NS', \
                    't_PWS_lin_EW', 't_PWS_pow_EW', 't_PWS_PWS_EW', \
                    't_PWS_lin_NS', 't_PWS_pow_NS', 't_PWS_PWS_NS', \
                    't_lin_lin_EW_cluster', 't_lin_pow_EW_cluster', 't_lin_PWS_EW_cluster', \
                    't_lin_lin_NS_cluster', 't_lin_pow_NS_cluster', 't_lin_PWS_NS_cluster', \
                    't_pow_lin_EW_cluster', 't_pow_pow_EW_cluster', 't_pow_PWS_EW_cluster', \
                    't_pow_lin_NS_cluster', 't_pow_pow_NS_cluster', 't_pow_PWS_NS_cluster', \
                    't_PWS_lin_EW_cluster', 't_PWS_pow_EW_cluster', 't_PWS_PWS_EW_cluster', \
                    't_PWS_lin_NS_cluster', 't_PWS_pow_NS_cluster', 't_PWS_PWS_NS_cluster', \
                    'cc_lin_lin_EW_cluster', 'cc_lin_pow_EW_cluster', 'cc_lin_PWS_EW_cluster', \
                    'cc_lin_lin_NS_cluster', 'cc_lin_pow_NS_cluster', 'cc_lin_PWS_NS_cluster', \
                    'cc_pow_lin_EW_cluster', 'cc_pow_pow_EW_cluster', 'cc_pow_PWS_EW_cluster', \
                    'cc_pow_lin_NS_cluster', 'cc_pow_pow_NS_cluster', 'cc_pow_PWS_NS_cluster', \
                    'cc_PWS_lin_EW_cluster', 'cc_PWS_pow_EW_cluster', 'cc_PWS_PWS_EW_cluster', \
                    'cc_PWS_lin_NS_cluster', 'cc_PWS_pow_NS_cluster', 'cc_PWS_PWS_NS_cluster'])
            i0 = len(df.index)
            df.loc[i0] = [x0, y0, ntremor, \
                t_lin_lin_EW, t_lin_pow_EW, t_lin_PWS_EW, \
                t_lin_lin_NS, t_lin_pow_NS, t_lin_PWS_NS, \
                t_pow_lin_EW, t_pow_pow_EW, t_pow_PWS_EW, \
                t_pow_lin_NS, t_pow_pow_NS, t_pow_PWS_NS, \
                t_PWS_lin_EW, t_PWS_pow_EW, t_PWS_PWS_EW, \
                t_PWS_lin_NS, t_PWS_pow_NS, t_PWS_PWS_NS, \
                t_lin_lin_EW_cluster, t_lin_pow_EW_cluster, t_lin_PWS_EW_cluster, \
                t_lin_lin_NS_cluster, t_lin_pow_NS_cluster, t_lin_PWS_NS_cluster, \
                t_pow_lin_EW_cluster, t_pow_pow_EW_cluster, t_pow_PWS_EW_cluster, \
                t_pow_lin_NS_cluster, t_pow_pow_NS_cluster, t_pow_PWS_NS_cluster, \
                t_PWS_lin_EW_cluster, t_PWS_pow_EW_cluster, t_PWS_PWS_EW_cluster, \
                t_PWS_lin_NS_cluster, t_PWS_pow_NS_cluster, t_PWS_PWS_NS_cluster, \
                cc_lin_lin_EW_cluster, cc_lin_pow_EW_cluster, cc_lin_PWS_EW_cluster, \
                cc_lin_lin_NS_cluster, cc_lin_pow_NS_cluster, cc_lin_PWS_NS_cluster, \
                cc_pow_lin_EW_cluster, cc_pow_pow_EW_cluster, cc_pow_PWS_EW_cluster, \
                cc_pow_lin_NS_cluster, cc_pow_pow_NS_cluster, cc_pow_PWS_NS_cluster, \
                cc_PWS_lin_EW_cluster, cc_PWS_pow_EW_cluster, cc_PWS_PWS_EW_cluster, \
                cc_PWS_lin_NS_cluster, cc_PWS_pow_NS_cluster, cc_PWS_PWS_NS_cluster]
            df['ntremor'] = df['ntremor'].astype('int')
            pickle.dump(df, open(namefile, 'wb'))

        else:
            print(x0, y0, nlincc, npowcc, nPWScc, nlinac, npowac, nPWSac)
