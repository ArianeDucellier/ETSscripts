"""
This module contains functions to test for long range dependence in time
series. The tests come from Taqqu and Teverovsky (1998).
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

from datetime import datetime, timedelta
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

def aggregate(X, m):
    """
    Function to return aggregated time series

    Input:
        type X = numpy array
        X = Time series
        type m = integer
        m = Number of values for the aggregation
    Output:
        type Xm = numpy array
        Xm = Aggregated time series
    """
    N = len(X)
    N2 = int(N / m)
    X2 = X[0 : N2 * m]
    X2 = np.reshape(X2, (N2, int(m)))
    Xm = np.mean(X2, axis=1)
    return Xm

def absolutevalue(dirname, filename, m):
    """
    Function to plot the first absolute moment of the aggregated series
    in function of m
    The slope is equal to H - 1 (Hurst parameter)

    Input:
        type dirname = string
        dirname = Repertory where to find the time series file
        type filename = string
        filename = Name of the time series file
        type m = numpy array of integers
        m = List of values for the aggregation
    Output:
        type H = float
        H = Hurst parameter        
    """
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X = data[3]
    AM = np.zeros(len(m))
    for i in range(0, len(m)):
        Xm = aggregate(X, m[i])
        AM[i] = np.mean(np.abs(Xm - np.mean(X)))
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(AM), (len(AM), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    H = regr.coef_[0][0] + 1
     # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(m), np.log10(AM), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (aggregation size)', fontsize=24)
    plt.ylabel('Log (absolute moment)', fontsize=24)
    plt.title('{:d} LFEs - H = {:4.2f} - R2 = {:4.2f}'.format( \
        np.sum(X), H, R2), fontsize=24)
    plt.savefig('absolutevalue/' + filename + '.eps', format='eps')
    plt.close(1)
    return H

def variance(dirname, filename, m):
    """
    Function to plot the sample variance of the aggregated series
    in function of m
    The slope is equal to 2 d - 1 (fractional index)

    Input:
        type dirname = string
        dirname = Repertory where to find the time series file
        type filename = string
        filename = Name of the time series file
        type m = numpy array of integers
        m = List of values for the aggregation
    Output:
        type d = float
        d = Fractional index
    """
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X = data[3]
    V = np.zeros(len(m))
    for i in range(0, len(m)):
        Xm = aggregate(X, m[i])
        V[i] = np.var(Xm)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(V), (len(V), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = 0.5 * (regr.coef_[0][0] + 1)
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(m), np.log10(V), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (aggregation size)', fontsize=24)
    plt.ylabel('Log (sample variance)', fontsize=24)
    plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
        np.sum(X), d, R2), fontsize=24)
    plt.savefig('variance/' + filename + '.eps', format='eps')
    plt.close(1)
    return d

def varianceresiduals(dirname, filename, m, method):
    """
    Function to plot the median / mean of the variance of residuals
    in function of m
    The slope is equal to 2 H (Hurst parameter) for the median
    The slope is equal to 2 d + 1 (fractional index) for the mean

    Input:
        type dirname = string
        dirname = Repertory where to find the time series file
        type filename = string
        filename = Name of the time series file
        type m = numpy array of integers
        m = List of values for the aggregation
        type method = string
        method = 'median' or 'mean'
    Output (median):
        type H = float
        H = Hurst parameter
    Output (mean):
        type d = float
        d = Fractional index
    """
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X = data[3]
    Vm = np.zeros(len(m))
    for i in range(0, len(m)):
        N = int(len(X) / m[i])
        V = np.zeros(N)
        for j in range(0, N):
            Y = np.cumsum(X[j * m[i] : (j + 1) * m[i]])
            # Linear regression
            t = np.arange(0, m[i])
            x = np.reshape(t, (len(t), 1))
            y = np.reshape(Y, (len(Y), 1))
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(x, y)
            y_pred = regr.predict(x)
            V[j] = mean_squared_error(y, y_pred)
        if (method == 'median'):
            Vm[i] = np.median(V)
        elif (method == 'mean'):
            Vm[i] = np.mean(V)
        else:
            raise ValueError('Method must be median or mean')
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(Vm), (len(Vm), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    if (method == 'median'):
        H = 0.5 + regr.coef_[0][0]
    elif (method == 'mean'):
        d = 0.5 * (regr.coef_[0][0] - 1)
    else:
        raise ValueError('Method must be median or mean')
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(m), np.log10(Vm), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (block size)', fontsize=24)
    if (method == 'median'):
        plt.ylabel('Log (median variance residuals)', fontsize=24)
        plt.title('{:d} LFEs - H = {:4.2f} - R2 = {:4.2f}'.format( \
            np.sum(X), H, R2), fontsize=24)
    else:
        plt.ylabel('Log (mean variance residuals)', fontsize=24)
        plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
            np.sum(X), d, R2), fontsize=24)
    plt.savefig('varianceresiduals/' + filename + '.eps', format='eps')
    plt.close(1)
    if (method == 'median'):
        return H
    else:
        return d

def RS(X, filename):
    """
    Function to plot the R/S statistic in function of m
    The slope is equal to d + 1/2 (fractional index)

    Input:
        type X = numpy array
        X = Time series
        type filename = string
        filename = Name of file to save the plot
    Output:
        type d = float
        d = Fractional index
    """
    N = len(X)
    Y = np.zeros(N)
    S = np.zeros(N)
    for n in range(0, N):
        Y[n] = np.sum(X[0 : n + 1])
        S[n] = np.mean(np.power(X[0 : n + 1], 2.0)) - np.power(np.mean( \
            X[0 : n + 1]), 2.0)
    RS = np.zeros(N)
    for n in range(0, N):
        t = np.arange(0, n + 1)
        if S[n] > 0.0:
            RS[n] = (np.max(Y[0 : n + 1] - t * Y[n] / (n + 1)) - np.min( \
                Y[0 : n + 1] - t * Y[n] / (n + 1))) / S[n]
    # Linear regression
    n = np.arange(0, N)
    k = np.where(S > 0.0)
    x = np.log10(n[k])
    y = np.log10(RS[k])
    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = regr.coef_[0][0] - 0.5
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(x, y, 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (n)', fontsize=24)
    plt.ylabel('Log (R/S)', fontsize=24)
    plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
        int(np.sum(X)), d, R2), fontsize=24)
    plt.savefig(filename, format='eps')
    return d    
    
def compute_variance(filename, winlen, tbegin, tend):
    
    # Get the time of LFE detections
    LFEtime = np.loadtxt('../data/LFEcatalog/detections/' + filename + \
        '_detect5_cull.txt', \
        dtype={'names': ('unknown', 'day', 'hour', 'second', 'threshold'), \
             'formats': (np.float, '|S6', np.int, np.float, np.float)}, \
        skiprows=2)
    # Initialize variance
    v = np.zeros(len(winlen))
    for l in range(0, len(winlen)):
        dt = tend - tbegin
        duration = dt.days * 86400.0 + dt.seconds + dt.microseconds * 0.000001
        nw = int(duration / winlen[l])
        nLFE = np.zeros(nw)
        # Loop on time windows
        for i in range(0, nw):
            t1 = tbegin + timedelta(seconds=np.float64(i * winlen[l]))
            t2 = tbegin + timedelta(seconds=np.float64((i + 1) * winlen[l]))
            # Loop on LFEs
            for j in range(0, np.shape(LFEtime)[0]):
                YMD = LFEtime[j][1]
                myYear = 2000 + int(YMD[0 : 2])
                myMonth = int(YMD[2 : 4])
                myDay = int(YMD[4 : 6])
                myHour = LFEtime[j][2] - 1
                myMinute = int(LFEtime[j][3] / 60.0)
                mySecond = int(LFEtime[j][3] - 60.0 * myMinute)
                myMicrosecond = int(1000000.0 * \
                    (LFEtime[j][3] - 60.0 * myMinute - mySecond))
                t = datetime(myYear, myMonth, myDay, myHour, myMinute, \
                    mySecond, myMicrosecond)
                if ((t >= t1) and (t < t2)):
                    nLFE[i] = nLFE[i] + 1
        v[l] = np.var(nLFE)
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(winlen), np.log10(v / winlen), 'ko')
    x = np.reshape(np.log10(winlen), (len(winlen), 1))
    y = np.reshape(np.log10(v / winlen), (len(winlen), 1))
    regr = linear_model.LinearRegression(normalize=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (window length)', fontsize=24)
    plt.ylabel('Log (variance / window length)', fontsize=24)
    plt.title('{:d} LFEs - alpha = {:4.2f} - R2 = {:4.2f}'.format( \
        np.shape(LFEtime)[0], - regr.coef_[0][0], R2), fontsize=24)
    plt.savefig('variance2/' + filename + '.eps', format='eps')

def variance_moulines(dirname, filename, m):
    """
    Function to plot the sample variance of the aggregated series
    in function of m
    The slope is equal to 2 H (Hurst parameter)

    Input:
        type dirname = string
        dirname = Repertory where to find the time series file
        type filename = string
        filename = Name of the time series file
        type m = numpy array of integers
        m = List of values for the aggregation
    Output:
        type H = float
        H = Hurst parameter        
    """
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X = data[3]
    N = len(X)
    Vm = np.zeros(len(m))
    for i in range(0, len(m)):
        N2 = int(N / m[i])
        X2 = X[0 : N2 * m[i]]
        X2 = np.reshape(X2, (N2, int(m[i])))
        Xm = np.sum(X2, axis=1)
        Vm[i] = np.var(Xm)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(Vm / m), (len(m), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    H = 0.5 * (regr.coef_[0][0] + 1.0)
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(m), np.log10(Vm / m), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (aggregation size)', fontsize=24)
    plt.ylabel('Log (variance / aggregation size)', fontsize=24)
    plt.title('{:d} LFEs - H = {:4.2f} - R2 = {:4.2f}'.format( \
        np.sum(X), H, R2), fontsize=24)
    plt.savefig('variance_moulines/' + filename + '.eps', format='eps')
    plt.close(1)
    return H
