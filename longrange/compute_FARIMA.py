"""
This script runs the tests for long range dependence from the module
test_long_range_parallel for the synthetic FARIMA time series
"""
import numpy as np
import os
import pandas as pd
import pickle

from test_long_range_parallel import absolutevalue
from test_long_range_parallel import variance
from test_long_range_parallel import variance_moulines
from test_long_range_parallel import varianceresiduals
from test_long_range_parallel import RSstatistic
from test_long_range_parallel import periodogram

dirname = 'FARIMA/'
files = ['series1_1', 'series1_2', 'series1_3', 'series1_4', 'series1_5', \
         'series2_1', 'series2_2', 'series2_3', 'series2_4', 'series2_5', \
         'series3_1', 'series3_2', 'series3_3', 'series3_4', 'series3_5', \
         'series4_1', 'series4_2', 'series4_3', 'series4_4', 'series4_5', \
         'series5_1', 'series5_2', 'series5_3', 'series5_4', 'series5_5', \
         'series6_1', 'series6_2', 'series6_3', 'series6_4', 'series6_5']

# Create pandas dataframe to store the results
df = pd.DataFrame(data={'series': files})

# Store time series into pickle files
for i in range(0, len(files)):
    ts = np.loadtxt(dirname + files[i] + '.txt')
    pickle.dump([0, 0, 0, ts], open(dirname + files[i] + '.pkl', 'wb'))

# Aggregation sizes
m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314], dtype=int)

# Absolute value method
newpath = 'absolutevalue' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

H_absval = np.zeros(len(files))

for i in range(0, len(files)):
    H_absval[i] = absolutevalue(dirname, files[i], m)

df['H_absval'] = H_absval

# Variance method
newpath = 'variance' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_var = np.zeros(len(files))

for i in range(0, len(files)):
    d_var[i] = variance(dirname, files[i], m)

df['d_var'] = d_var

# Variance method (from Moulines's paper)
newpath = 'variancemoulines' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

H_varm = np.zeros(len(files))

for i in range(0, len(files)):
    H_varm[i] = variance_moulines(dirname, files[i], m)

df['H_varm'] = H_varm

# Variance of residuals method
newpath = 'varianceresiduals' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_varres = np.zeros(len(files))

for i in range(0, len(files)):
    d_varres[i] = varianceresiduals(dirname, files[i], m, 'mean')

df['d_varres'] = d_varres

# R/S method
newpath = 'RS' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_RS = np.zeros(len(files))

for i in range(0, len(files)):
    d_RS[i] = RSstatistic(dirname, files[i], m)

df['d_RS'] = d_RS

# Periodogram method
newpath = 'periodogram'
if not os.path.exists(newpath):
    os.makedirs(newpath)

dt = 1.0

d_p = np.zeros(len(files))

for i in range(0, len(files)):
    d_p[i] = periodogram(dirname, files[i], dt)

df['d_p'] = d_p

# Save dataframe into file
filename = 'FARIMA.pkl'
pickle.dump([df], open(filename, 'wb'))
