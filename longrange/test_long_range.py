"""
This module contains functions to test for long range dependence in time
series. The tests come from Taqqu and Teverovsky (1998) and from
Reisen et al. (2017).
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

from math import floor, pi, sqrt
from scipy.fftpack import fft
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
    plt.savefig('variancemoulines/' + filename + '.eps', format='eps')
    plt.close(1)
    return H

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

def RS(dirname, filename, m):
    """
    Function to plot the R/S statistic in function of m
    The slope is equal to d + 1/2 (fractional index)

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
    N = len(X)
    Y = np.cumsum(X)
    RS = []
    lag = []
    for i in range(0, len(m)):
        K = int(N / m[i])
        for t in range(0, K):
            index = np.arange(0, m[i])
            Rmax = np.max(Y[t * m[i] : (t + 1) * m[i]] - Y[t * m[i]] - index * \
                (Y[(t + 1) * m[i] - 1] - Y[t * m[i]]) / m[i])
            Rmin = np.min(Y[t * m[i] : (t + 1) * m[i]] - Y[t * m[i]] - index * \
                (Y[(t + 1) * m[i] - 1] - Y[t * m[i]]) / m[i])
            R = Rmax - Rmin
            S = sqrt(np.var(Y[t * m[i] : (t + 1) * m[i]]))
            if (S != 0.0):
                RS.append(R / S)
                lag.append(m[i])
    RS = np.asarray(RS)
    lag = np.asarray(lag)
    # Linear regression
    x = np.reshape(np.log10(lag), (len(lag), 1))
    y = np.reshape(np.log10(RS), (len(RS), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = regr.coef_[0][0] - 0.5
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(lag), np.log10(RS), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (aggregation size)', fontsize=24)
    plt.ylabel('Log (R/S statistic)', fontsize=24)
    plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
        np.sum(X), d, R2), fontsize=24)
    plt.savefig('RS/' + filename + '.eps', format='eps')
    plt.close(1)
    return d

def periodogram(dirname, filename, dt):
    """
    Function to plot the periodogram of the aggregated series
    in function of m
    The slope is equal to - 2 d (fractional index)

    Input:
        type dirname = string
        dirname = Repertory where to find the time series file
        type filename = string
        filename = Name of the time series file
        type dt = float
        dt = Time step of the time series (one minute)
    Output:
        type d = float
        d = Fractional index
    """
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X = data[3]
    N = len(X)
    Y = fft(X)
    I = np.power(np.abs(Y[1 : int(N / 20) + 1]), 2.0)
    nu = (1.0 / (N * dt)) * np.arange(1, int(N / 20) + 1)
    # Linear regression
    x = np.reshape(np.log10(nu[I > 0.0]), (len(nu[I > 0.0]), 1))
    y = np.reshape(np.log10(I[I > 0.0]), (len(I[I > 0.0]), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = - 0.5 * regr.coef_[0][0]
    # Plot
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(nu), np.log10(I), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (frequency)', fontsize=24)
    plt.ylabel('Log (spectral density)', fontsize=24)
    plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
        np.sum(X), d, R2), fontsize=24)
    plt.savefig('periodogram/' + filename + '.eps', format='eps')
    plt.close(1)
    return d

def periodogram_GPH(dirname, filename, beta):
    """
    Function to compute the periodogram using the method of
    Geweke and Porter-Hudak (1983) and compute the
    fractional index

    Input:
        type dirname = string
        dirname = Repertory where to find the time series file
        type filename = string
        filename = Name of the time series file
        type beta = floor
        beta = Number of frequencies used = N ** beta
    Output:
        type d = float
        d = Fractional index
    """
    assert (beta > 0.0 and beta < 1.0), \
        'We must have 0 < beta < 1'
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X = data[3]
    N = len(X)
    k = np.arange(1, N + 1)
    m = int(floor(N ** beta))
    l = (2.0 * pi / N) * np.arange(1, m + 1)
    I = np.zeros(m)
    for i in range(0, m):
        b1 = (2.0 / N) * np.sum(X * np.cos(k * l[i]))
        b2 = (2.0 / N) * np.sum(X * np.sin(k * l[i]))
        I[i] = (N / (8.0 * pi)) * (b1 * b1 + b2 * b2)
    Y = np.log(np.abs(2.0 * np.sin(l / 2.0)))
    d = - 0.5 * np.sum((Y - np.mean(Y)) * np.log(I))/ \
        np.sum(np.square(Y - np.mean(Y)))
    return d
