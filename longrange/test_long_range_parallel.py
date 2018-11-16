"""
This module contains functions to test for long range dependence in time
series. The tests come from Taqqu and Teverovsky (1998).
"""

import multiprocessing
import numpy as np
import pickle

from functools import partial
from math import sqrt
from multiprocessing import Pool
from scipy.fftpack import fft
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

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

def get_absval(X, m, i):
    Xm = aggregate(X, m[i])
    AM = np.mean(np.abs(Xm - np.mean(X)))
    return AM

def absolutevalue(dirname, filename, m):
    """
    Function to compute the first absolute moment of the aggregated series
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
    map_func = partial(get_absval, X, m)
    with Pool(len(m)) as pool:
        result = pool.map(map_func, iter(range(0, len(m))))
    AM = np.array(result)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(AM), (len(AM), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    H = regr.coef_[0][0] + 1
    pickle.dump([m, AM, np.sum(X)], open('absolutevalue/' + filename + \
        '.pkl', 'wb'))
    return H

def get_var(X, m, i):
    Xm = aggregate(X, m[i])
    V = np.var(Xm)
    return V
 
def variance(dirname, filename, m):
    """
    Function to compute the sample variance of the aggregated series
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
    map_func = partial(get_var, X, m)
    with Pool(len(m)) as pool:
        result = pool.map(map_func, iter(range(0, len(m))))
    V = np.array(result)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(V), (len(V), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    d = 0.5 * (regr.coef_[0][0] + 1)
    pickle.dump([m, V, np.sum(X)], open('variance/' + filename + \
        '.pkl', 'wb'))
    return d

def get_varm(X, m, i):
    N = len(X)
    N2 = int(N / m[i])
    X2 = X[0 : N2 * m[i]]
    X2 = np.reshape(X2, (N2, int(m[i])))
    Xm = np.sum(X2, axis=1)
    Vm = np.var(Xm)
    return Vm

def variance_moulines(dirname, filename, m):
    """
    Function to compute the sample variance of the aggregated series
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
    map_func = partial(get_varm, X, m)
    with Pool(len(m)) as pool:
        result = pool.map(map_func, iter(range(0, len(m))))
    Vm = np.array(result)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(Vm / m), (len(m), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    H = 0.5 * (regr.coef_[0][0] + 1.0)
    pickle.dump([m, Vm, np.sum(X)], open('variancemoulines/' + filename + \
        '.pkl', 'wb'))
    return H

def get_varres(X, m, method, i):
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
        Vm = np.median(V)
    elif (method == 'mean'):
        Vm = np.mean(V)
    else:
        raise ValueError('Method must be median or mean')
    return Vm

def varianceresiduals(dirname, filename, m, method):
    """
    Function to compute the median / mean of the variance of residuals
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
    map_func = partial(get_varres, X, m, method)
    with Pool(len(m)) as pool:
        result = pool.map(map_func, iter(range(0, len(m))))
    Vm = np.array(result)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(Vm), (len(Vm), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    if (method == 'median'):
        H = 0.5 + regr.coef_[0][0]
    elif (method == 'mean'):
        d = 0.5 * (regr.coef_[0][0] - 1)
    else:
        raise ValueError('Method must be median or mean')
    pickle.dump([m, Vm, np.sum(X)], open('varianceresiduals/' + filename + \
        '.pkl', 'wb'))
    if (method == 'median'):
        return H
    else:
        return d

def get_RS(X, m, i):
    N = len(X)
    Y = np.cumsum(X)
    RS = []
    lag = []
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
    return (RS, lag)

def RSstatistic(dirname, filename, m):
    """
    Function to compute the R/S statistic in function of m
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
    map_func = partial(get_RS, X, m)
    with Pool(len(m)) as pool:
        result = pool.map(map_func, iter(range(0, len(m))))
    RS = []
    lag = []
    for i in range(0, len(m)):
        K = len(result[i][0])
        for j in range(0, K):
            RS.append(result[i][0][j])
            lag.append(result[i][1][j])
    RS = np.array(RS)
    lag = np.array(lag)
    # Linear regression
    x = np.reshape(np.log10(lag), (len(lag), 1))
    y = np.reshape(np.log10(RS), (len(RS), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    d = regr.coef_[0][0] - 0.5
    pickle.dump([lag, RS, np.sum(X)], open('RS/' + filename + \
        '.pkl', 'wb'))
    return d

def periodogram(dirname, filename, dt):
    """
    Function to compute the periodogram of the aggregated series
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
    d = - 0.5 * regr.coef_[0][0]
    pickle.dump([nu, I, np.sum(X)], open('periodogram/' + filename + \
        '.pkl', 'wb'))
    return d
