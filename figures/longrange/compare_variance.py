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

# Linear regression for Poisson
x_P = np.reshape(np.log10(np.array(dfP['m'])), (len(dfP['m']), 1))
y_P = np.reshape(np.log10(np.array(dfP['var']) * np.array(dfP['m'])), (len(dfP['var']), 1))
regr_P = linear_model.LinearRegression(fit_intercept=True)
regr_P.fit(x_P, y_P)
y_pred_P = regr_P.predict(x_P)
   
# Linear regression for ETAS
x_E = np.reshape(np.log10(np.array(dfE['m'])), (len(dfE['m']), 1))
y_E = np.reshape(np.log10(np.array(dfE['var']) * np.array(dfE['m'])), (len(dfE['var']), 1))
regr_E = linear_model.LinearRegression(fit_intercept=True)
regr_E.fit(x_E, y_E)
y_pred_E = regr_E.predict(x_E)
print(regr_E.coef_[0][0])
   
# Linear regression for FARIMA
x_F = np.reshape(np.log10(np.array(dfF['m'])), (len(dfF['m']), 1))
y_F = np.reshape(np.log10(np.array(dfF['var']) * np.array(dfF['m'])), (len(dfF['var']), 1))
regr_F = linear_model.LinearRegression(fit_intercept=True)
regr_F.fit(x_F, y_F)
y_pred_F = regr_F.predict(x_F)

# Figures
params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params)

# Plot Poisson
plt.figure(1, figsize=(10, 10))
plt.plot(np.log10(dfP['m']), np.log10(dfP['var'] * dfP['m']), 'ro', \
    label='Poisson - Slope = {:4.2f}'.format(regr_P.coef_[0][0]))
plt.plot(x_P, y_pred_P, 'r-')
plt.xlabel('Log (Sample size)', fontsize=20)
plt.ylabel('Log (Sample size * Variance)', fontsize=20)
plt.ylim([-1.0, 1.5])
plt.legend(loc=2, fontsize=20)
plt.savefig('comparison_1.eps', format='eps')
plt.close(1)

# Plot Poisson and ETAS
plt.figure(1, figsize=(10, 10))
plt.plot(np.log10(dfP['m']), np.log10(dfP['var'] * dfP['m']), 'ro', \
    label='Poisson - Slope = {:4.2f}'.format(regr_P.coef_[0][0]))
plt.plot(x_P, y_pred_P, 'r-')
plt.plot(np.log10(dfE['m']), np.log10(dfE['var'] * dfE['m']), 'bo', \
    label='ETAS - Slope = {:4.2f}'.format(regr_E.coef_[0][0]))
plt.plot(x_E, y_pred_E, 'b-')
plt.xlabel('Log (Sample size)', fontsize=20)
plt.ylabel('Log (Sample size * Variance)', fontsize=20)
plt.ylim([-1.0, 1.5])
plt.legend(loc=2, fontsize=20)
plt.savefig('comparison_2.eps', format='eps')
plt.close(1)

# Plot Poisson, ETAS and FARIMA
plt.figure(1, figsize=(10, 10))
plt.plot(np.log10(dfP['m']), np.log10(dfP['var'] * dfP['m']), 'ro', \
    label='Poisson - Slope = {:4.2f}'.format(regr_P.coef_[0][0]))
plt.plot(x_P, y_pred_P, 'r-')
plt.plot(np.log10(dfE['m']), np.log10(dfE['var'] * dfE['m']), 'bo', \
    label='ETAS - Slope = {:4.2f}'.format(regr_E.coef_[0][0]))
plt.plot(x_E, y_pred_E, 'b-')
plt.plot(np.log10(dfF['m']), np.log10(dfF['var'] * dfF['m']), 'go', \
    label='FARIMA - Slope = {:4.2f}'.format(regr_F.coef_[0][0]))
plt.plot(x_F, y_pred_F, 'g-')
plt.xlabel('Log (Sample size)', fontsize=20)
plt.ylabel('Log (Sample size * Variance)', fontsize=20)
plt.ylim([-1.0, 1.5])
plt.legend(loc=2, fontsize=20)
plt.savefig('comparison_3.eps', format='eps')
plt.close(1)
