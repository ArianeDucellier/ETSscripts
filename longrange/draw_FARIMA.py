"""
This script draws the results of the tests with the module draw_long_range
for the synthetic FARIMA time series
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn import linear_model
from sklearn.metrics import r2_score

def draw_absolutevalue(filename, true_d):
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
    plt.title('H = {:4.2f} (true value = {:4.2f}) - R2 = {:4.2f}'.format( \
        H, true_d + 0.5, R2), fontsize=24)
    plt.savefig('absolutevalue/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_variance(filename, true_d):
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
    plt.title('d = {:4.2f} (true value = {:4.2f}) - R2 = {:4.2f}'.format( \
        d, true_d, R2), fontsize=24)
    plt.savefig('variance/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_variance_moulines(filename, true_d):
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
    plt.title('H = {:4.2f} (true value = {:4.2f}) - R2 = {:4.2f}'.format( \
        H, true_d + 0.5, R2), fontsize=24)
    plt.savefig('variancemoulines/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_varianceresiduals(filename, method, true_d):
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
        plt.title('H = {:4.2f} (true value = {:4.2f}) - R2 = {:4.2f}'.format( \
            H, true_d + 0.5, R2), fontsize=24)
    else:
        plt.ylabel('Log (mean variance residuals)', fontsize=24)
        plt.title('d = {:4.2f} (true value = {:4.2f}) - R2 = {:4.2f}'.format( \
            d, true_d, R2), fontsize=24)
    plt.savefig('varianceresiduals/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_RSstatistic(filename, true_d):
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
    plt.title('d = {:4.2f} (true value = {:4.2f}) - R2 = {:4.2f}'.format( \
        d, true_d, R2), fontsize=24)
    plt.savefig('RS/' + filename + '.eps', format='eps')
    plt.close(1)

def draw_periodogram(filename, true_d):
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
    plt.title('d = {:4.2f} (true value = {:4.2f}) - R2 = {:4.2f}'.format( \
        d, true_d, R2), fontsize=24)
    plt.savefig('periodogram/' + filename + '.eps', format='eps')
    plt.close(1)

files = ['series1_1', 'series1_2', 'series1_3', 'series1_4', 'series1_5', \
         'series2_1', 'series2_2', 'series2_3', 'series2_4', 'series2_5', \
         'series3_1', 'series3_2', 'series3_3', 'series3_4', 'series3_5', \
         'series4_1', 'series4_2', 'series4_3', 'series4_4', 'series4_5', \
         'series5_1', 'series5_2', 'series5_3', 'series5_4', 'series5_5', \
         'series6_1', 'series6_2', 'series6_3', 'series6_4', 'series6_5']

true_d = [0.0, 0.1, 0.2, 0.3, 0.4, \
          0.0, 0.1, 0.2, 0.3, 0.4, \
          0.0, 0.1, 0.2, 0.3, 0.4, \
          0.0, 0.1, 0.2, 0.3, 0.4, \
          0.0, 0.1, 0.2, 0.3, 0.4, \
          0.0, 0.1, 0.2, 0.3, 0.4]

# Absolute value method
for i in range(0, len(files)):
    draw_absolutevalue(files[i], true_d[i])

os.rename('absolutevalue', 'absolutevalue_FARIMA')

# Variance method
for i in range(0, len(files)):
    draw_variance(files[i], true_d[i])

os.rename('variance', 'variance_FARIMA')

# Variance method (from Moulines's paper)
for i in range(0, len(files)):
    draw_variance_moulines(files[i], true_d[i])

os.rename('variancemoulines', 'variancemoulines_FARIMA')

# Variance of residuals method
for i in range(0, len(files)):
    draw_varianceresiduals(files[i], 'mean', true_d[i])

os.rename('varianceresiduals', 'varianceresiduals_FARIMA')

# R/S method
for i in range(0, len(files)):
    draw_RSstatistic(files[i], true_d[i])

os.rename('RS', 'RS_FARIMA')

# Periodogram method
for i in range(0, len(files)):
    draw_periodogram(files[i], true_d[i])

os.rename('periodogram', 'periodogram_FARIMA')
