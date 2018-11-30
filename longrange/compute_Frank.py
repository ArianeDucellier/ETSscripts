"""
This script runs the tests for long range dependence from the module
test_long_range_parallel for the LFE catalog of Frank et al. (2014)
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
LFEtime = pd.read_csv('../data/Frank_2014/frank_jgr_2014_lfe_catalog.txt', \
    delim_whitespace=True, header=None, skiprows=0)
LFEtime.columns = ['year', 'month', 'day', 'hour', 'minute', 'second', \
    'ID', 'latitude', 'longitude', 'depth']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()
    
dirname = '../data/Frank_2014/timeseries/'

# Create pandas dataframe to store the results
df = pd.DataFrame(data={'family': families})
 
# Absolute value method
newpath = 'absolutevalue' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980, \
    3827, 4914, 6310, 8103, 10404, 13359, 17154, 22026, 28282, 36315, \
    46630], dtype=int)

H_absval = np.zeros(len(families))

for i in range(0, len(families)):
    filename = str(families[i])
    H_absval[i] = absolutevalue(dirname, filename, m)

df['H_absval'] = H_absval

# Variance method
newpath = 'variance' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_var = np.zeros(len(families))

for i in range(0, len(families)):
    filename = str(families[i])
    d_var[i] = variance(dirname, filename, m)

df['d_var'] = d_var

# Variance method (from Moulines's paper)
newpath = 'variancemoulines' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

H_varm = np.zeros(len(families))

for i in range(0, len(families)):
    filename = str(families[i])
    H_varm[i] = variance_moulines(dirname, filename, m)

df['H_varm'] = H_varm

# Variance of residuals method
newpath = 'varianceresiduals' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_varres = np.zeros(len(families))

for i in range(0, len(families)):
    filename = str(families[i])
    d_varres[i] = varianceresiduals(dirname, filename, m, 'mean')

df['d_varres'] = d_varres

# R/S method
newpath = 'RS' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_RS = np.zeros(len(families))

for i in range(0, len(families)):
    filename = str(families[i])
    d_RS[i] = RSstatistic(dirname, filename, m)

df['d_RS'] = d_RS

# Periodogram method
newpath = 'periodogram'
if not os.path.exists(newpath):
    os.makedirs(newpath)

dt = 60.0

d_p = np.zeros(len(families))

for i in range(0, len(families)):
    filename = str(families[i])
    d_p[i] = periodogram(dirname, filename, dt)

df['d_p'] = d_p

# Save dataframe into file
filename = 'Frank_2014.pkl'
pickle.dump([df], open(filename, 'wb'))
