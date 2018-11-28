"""
This script runs the tests for long range dependence from the module
test_long_range_parallel for the LFE catalog of Shelly (2017)
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

# Read the LFE file
LFEtime = pd.read_csv('../data/Shelly_2017/jgrb52060-sup-0002-datas1.txt', \
    delim_whitespace=True, header=None, skiprows=2)
LFEtime.columns = ['year', 'month', 'day', 's_of_day', 'hr', 'min', 'sec', \
    'ccsum', 'meancc', 'med_cc', 'seqday', 'ID', 'latitude', 'longitude', \
    'depth', 'n_chan']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

dirname = '../data/Shelly_2017/timeseries/'

# Create pandas dataframe to store the results
df = pd.DataFrame(data={'family': families})
 
# Absolute value method
newpath = 'absolutevalue' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980, \
    3827, 4914, 6310, 8103, 10404, 13359, 17154, 22026, 28282, 36315, \
    46630, 59874, 76879, 98715, 126753, 162754, 208981, 268337, 344551], \
    dtype=int)

H_absval = np.zeros(len(families))

for i in range(0, len(families)):
    filename = families[i]
    H_absval[i] = absolutevalue(dirname, filename, m)

df['H_absval'] = H_absval

# Variance method
newpath = 'variance' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_var = np.zeros(len(families))

for i in range(0, len(families)):
    filename = families[i]
    d_var[i] = variance(dirname, filename, m)

df['d_var'] = d_var

# Variance method (from Moulines's paper)
newpath = 'variancemoulines' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

H_varm = np.zeros(len(families))

for i in range(0, len(families)):
    filename = families[i]
    H_varm[i] = variance_moulines(dirname, filename, m)

df['H_varm'] = H_varm

# Variance of residuals method
newpath = 'varianceresiduals' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_varres = np.zeros(len(families))

for i in range(0, len(families)):
    filename = families[i]
    d_varres[i] = varianceresiduals(dirname, filename, m, 'mean')

df['d_varres'] = d_varres

# R/S method
newpath = 'RS' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_RS = np.zeros(len(families))

for i in range(0, len(families)):
    filename = families[i]
    d_RS[i] = RSstatistic(dirname, filename, m)

df['d_RS'] = d_RS

# Periodogram method
newpath = 'periodogram'
if not os.path.exists(newpath):
    os.makedirs(newpath)

dt = 60.0

d_p = np.zeros(len(families))

for i in range(0, len(families)):
    filename = families[i]
    d_p[i] = periodogram(dirname, filename, dt)

df['d_p'] = d_p

# Save dataframe into file
filename = 'Shelly_2017.pkl'
pickle.dump([df], open(filename, 'wb'))
