"""
This module contains functions to draw the results of the tests for long range
dependence in time series. The tests come from Taqqu and Teverovsky (1998).
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle

from sklearn import linear_model
from sklearn.metrics import r2_score

def draw_absolutevalue(filename):
    """
    Function to plot the first absolute moment of the aggregated series
    in function of m
    The slope is equal to H - 1 (Hurst parameter)

    Input:
        type filename = string
        filename = Name of results file
    Output:
        None       
    """
    data = pickle.load(open('absolutevalue/' + filename + '.pkl', 'rb'))
    m = data[0]
    AM = data[1]
    nLFE = data[2]
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(AM), (len(AM), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    H = regr.coef_[0][0] + 1
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(m), np.log10(AM), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (aggregation size)', fontsize=24)
    plt.ylabel('Log (absolute moment)', fontsize=24)
    plt.title('{:d} LFEs - H = {:4.2f} - R2 = {:4.2f}'.format( \
        nLFE, H, R2), fontsize=24)
    plt.savefig('absolutevalue/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_variance(filename):
    """
    Function to plot the sample variance of the aggregated series
    in function of m
    The slope is equal to 2 d - 1 (fractional index)

    Input:
        type filename = string
        filename = Name of results file
    Output:
        None
    """
    data = pickle.load(open('variance/' + filename + '.pkl', 'rb'))
    m = data[0]
    V = data[1]
    nLFE = data[2]
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(V), (len(V), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = 0.5 * (regr.coef_[0][0] + 1)
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(m), np.log10(V), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (aggregation size)', fontsize=24)
    plt.ylabel('Log (sample variance)', fontsize=24)
    plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
        nLFE, d, R2), fontsize=24)
    plt.savefig('variance/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_variance_moulines(filename):
    """
    Function to plot the sample variance of the aggregated series
    in function of m
    The slope is equal to 2 H (Hurst parameter)

    Input:
        type filename = string
        filename = Name of results file
    Output:
        None
    """
    data = pickle.load(open('variancemoulines/' + filename + '.pkl', 'rb'))
    m = data[0]
    Vm = data[1]
    nLFE = data[2]
    # Linear regression
    x = np.reshape(np.log10(m), (len(m), 1))
    y = np.reshape(np.log10(Vm / m), (len(m), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    H = 0.5 * (regr.coef_[0][0] + 1.0)
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(m), np.log10(Vm / m), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (aggregation size)', fontsize=24)
    plt.ylabel('Log (variance / aggregation size)', fontsize=24)
    plt.title('{:d} LFEs - H = {:4.2f} - R2 = {:4.2f}'.format( \
        nLFE, H, R2), fontsize=24)
    plt.savefig('variancemoulines/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_varianceresiduals(filename, method):
    """
    Function to plot the median / mean of the variance of residuals
    in function of m
    The slope is equal to 2 H (Hurst parameter) for the median
    The slope is equal to 2 d + 1 (fractional index) for the mean

    Input:
        type filename = string
        filename = Name of results file
        type method = string
        method = 'median' or 'mean'
    Output:
        None
    """
    data = pickle.load(open('varianceresiduals/' + filename + '.pkl', 'rb'))
    m = data[0]
    Vm = data[1]
    nLFE = data[2]
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
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(m), np.log10(Vm), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (block size)', fontsize=24)
    if (method == 'median'):
        plt.ylabel('Log (median variance residuals)', fontsize=24)
        plt.title('{:d} LFEs - H = {:4.2f} - R2 = {:4.2f}'.format( \
            nLFE, H, R2), fontsize=24)
    else:
        plt.ylabel('Log (mean variance residuals)', fontsize=24)
        plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
            nLFE, d, R2), fontsize=24)
    plt.savefig('varianceresiduals/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_RSstatistic(filename):
    """
    Function to plot the R/S statistic in function of m
    The slope is equal to d + 1/2 (fractional index)

    Input:
        type filename = string
        filename = Name of results file
    Output:
        None
    """
    data = pickle.load(open('RS/' + filename + '.pkl', 'rb'))
    lag = data[0]
    RS = data[1]
    nLFE = data[2]
    # Linear regression
    x = np.reshape(np.log10(lag), (len(lag), 1))
    y = np.reshape(np.log10(RS), (len(RS), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = regr.coef_[0][0] - 0.5
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(lag), np.log10(RS), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (aggregation size)', fontsize=24)
    plt.ylabel('Log (R/S statistic)', fontsize=24)
    plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
        nLFE, d, R2), fontsize=24)
    plt.savefig('RS/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_periodogram(filename):
    """
    Function to plot the periodogram of the aggregated series
    in function of m
    The slope is equal to - 2 d (fractional index)

    Input:
        type filename = string
        filename = Name of results file
    Output:
        None
    """
    data = pickle.load(open('periodogram/' + filename + '.pkl', 'rb'))
    nu = data[0]
    I = data[1]
    nLFE = data[2]
    # Linear regression
    x = np.reshape(np.log10(nu[I > 0.0]), (len(nu[I > 0.0]), 1))
    y = np.reshape(np.log10(I[I > 0.0]), (len(I[I > 0.0]), 1))
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(x, y)
    y_pred = regr.predict(x)
    R2 = r2_score(y, y_pred)
    d = - 0.5 * regr.coef_[0][0]
    plt.figure(1, figsize=(10, 10))
    plt.plot(np.log10(nu), np.log10(I), 'ko')
    plt.plot(x, y_pred, 'r-')
    plt.xlabel('Log (frequency)', fontsize=24)
    plt.ylabel('Log (spectral density)', fontsize=24)
    plt.title('{:d} LFEs - d = {:4.2f} - R2 = {:4.2f}'.format( \
        nLFE, d, R2), fontsize=24)
    plt.savefig('periodogram/' + filename + '.eps', format='eps')
    plt.close(1)
