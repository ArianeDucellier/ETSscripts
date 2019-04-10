"""
Script to make figure illustrating the presence or absence of
long-range dependence for different time series
"""
import matplotlib.pylab as pylab
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pickle

from sklearn import linear_model

dfP = pickle.load(open('Poisson.pkl', 'rb'))
dfE = pickle.load(open('ETAS.pkl', 'rb'))
dfF = pickle.load(open('FARIMA.pkl', 'rb'))

# Figure
plt.figure(1, figsize=(10, 10))
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params)

# Linear regression for Poisson
x = np.reshape(np.log10(np.array(dfP['m'])), (len(dfP['m']), 1))
y = np.reshape(np.log10(np.array(dfP['var']) / np.array(dfP['m'])), (len(dfP['var']), 1))
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(x, y)
y_pred = regr.predict(x)
    
plt.plot(np.log10(dfP['m']), np.log10(dfP['var'] / dfP['m']), 'ro', \
    label='Poisson - Slope = {:6.4f}'.format(regr.coef_[0][0]))
plt.plot(x, y_pred, 'r-')

# Linear regression for ETAS
x = np.reshape(np.log10(np.array(dfE['m'])), (len(dfE['m']), 1))
y = np.reshape(np.log10(np.array(dfE['var']) / np.array(dfE['m'])), (len(dfE['var']), 1))
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(x, y)
y_pred = regr.predict(x)
    
plt.plot(np.log10(dfE['m']), np.log10(dfE['var'] / dfE['m']), 'bo', \
    label='ETAS - Slope = {:6.4f}'.format(regr.coef_[0][0]))
plt.plot(x, y_pred, 'b-')

# Linear regression for FARIMA
x = np.reshape(np.log10(np.array(dfF['m'])), (len(dfF['m']), 1))
y = np.reshape(np.log10(np.array(dfF['var']) / np.array(dfF['m'])), (len(dfF['var']), 1))
regr = linear_model.LinearRegression(fit_intercept=True)
regr.fit(x, y)
y_pred = regr.predict(x)
    
plt.plot(np.log10(dfF['m']), np.log10(dfF['var'] / dfF['m']), 'go', \
    label='FARIMA - Slope = {:6.4f}'.format(regr.coef_[0][0]))
plt.plot(x, y_pred, 'g-')

plt.xlabel('Log (Aggregation size)', fontsize=20)
plt.ylabel('Log (Variance / Aggregation size)', fontsize=20)
plt.legend(loc=2, fontsize=20)

plt.savefig('comparison.eps', format='eps')
