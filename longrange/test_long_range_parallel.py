"""
This module contains functions to test for long range dependence in time
series. The tests come from Taqqu and Teverovsky (1998).
"""

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pickle

from functools import partial
from math import sqrt
from multiprocessing import Pool
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

def get_absval(X, m, i):
    Xm = aggregate(X, m[i])
    AM = np.mean(np.abs(Xm - np.mean(X)))
    return AM

def absolutevalue(dirname, filename, m, draw=True):
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
        type draw = boolean
        draw = Do we plot the linear regression?
    Output:
        type H = float
        H = Hurst parameter        
    """
    infile = open(dirname + filename + '.pkl', 'rb')
    data = pickle.load(infile)
    infile.close()
    X = data[3]
    pool = Pool(len(m))
    map_func = partial(get_absval, X, m)
    result = pool.map(map_func, iter(range(0, len(m))))
    pool.close()
    pool.join()
    AM = np.array(result)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(AM), (len(AM), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    H = regr.coef_[0][0] + 1
     # Plot
    if (draw==True):
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

def get_var(X, m, i):
    Xm = aggregate(X, m[i])
    V = np.var(Xm)
    return V
 
def variance(dirname, filename, m, draw=True):
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
        type draw = boolean
        draw = Do we plot the linear regression?
    Output:
        type d = float
        d = Fractional index
    """
    infile = open(dirname + filename + '.pkl', 'rb')
    data = pickle.load(infile)
    infile.close()
    X = data[3]
    pool = Pool(len(m))
    map_func = partial(get_var, X, m)
    result = pool.map(map_func, iter(range(0, len(m))))
    pool.close()
    pool.join()
    V = np.array(result)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(V), (len(V), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = 0.5 * (regr.coef_[0][0] + 1)
    # Plot
    if (draw==True):
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

def get_varm(X, m, i):
    N = len(X)
    N2 = int(N / m[i])
    X2 = X[0 : N2 * m[i]]
    X2 = np.reshape(X2, (N2, int(m[i])))
    Xm = np.sum(X2, axis=1)
    Vm = np.var(Xm)
    return Vm

def variance_moulines(dirname, filename, m, draw=True):
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
        type draw = boolean
        draw = Do we plot the linear regression?
    Output:
        type H = float
        H = Hurst parameter        
    """
    infile = open(dirname + filename + '.pkl', 'rb')
    data = pickle.load(infile)
    infile.close()
    X = data[3]
    pool = Pool(len(m))
    map_func = partial(get_varm, X, m)
    result = pool.map(map_func, iter(range(0, len(m))))
    pool.close()
    pool.join()
    Vm = np.array(result)
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(Vm / m), (len(m), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    H = 0.5 * (regr.coef_[0][0] + 1.0)
    # Plot
    if (draw==True):
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

def varianceresiduals(dirname, filename, m, method, draw=True):
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
        type draw = boolean
        draw = Do we plot the linear regression?
    Output (median):
        type H = float
        H = Hurst parameter
    Output (mean):
        type d = float
        d = Fractional index
    """
    infile = open(dirname + filename + '.pkl', 'rb')
    data = pickle.load(infile)
    infile.close()
    X = data[3]
    pool = Pool(len(m))
    map_func = partial(get_varres, X, m, method)
    result = pool.map(map_func, iter(range(0, len(m))))
    Vm = np.array(result)
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
    if (draw==True):
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

def RSstatistic(dirname, filename, m, draw=True):
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
        type draw = boolean
        draw = Do we plot the linear regression?
    Output:
        type d = float
        d = Fractional index
    """
    infile = open(dirname + filename + '.pkl', 'rb')
    data = pickle.load(infile)
    infile.close()
    X = data[3]
    pool = Pool(len(m))
    map_func = partial(get_RS, X, m)
    result = pool.map(map_func, iter(range(0, len(m))))
    pool.close()
    pool.join()
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
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = regr.coef_[0][0] - 0.5
    # Plot
    if (draw==True):
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
