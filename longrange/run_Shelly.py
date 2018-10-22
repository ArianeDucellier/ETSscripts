"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Shelly (2017)
"""

import numpy as np
import os
import pandas as pd

from test_long_range import variance_moulines
from test_long_range import periodogram
from test_long_range import absolutevalue
from test_long_range import variance
from test_long_range import varianceresiduals
from test_long_range import RS

# Read the LFE file
LFEtime = pd.read_csv('../data/Shelly_2017/jgrb52060-sup-0002-datas1.txt', \
    delim_whitespace=True, header=None, skiprows=2)
LFEtime.columns = ['year', 'month', 'day', 's_of_day', 'hr', 'min', 'sec', \
    'ccsum', 'meancc', 'med_cc', 'seqday', 'ID', 'latitude', 'longitude', \
    'depth', 'n_chan']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

dirname = '../data/Shelly_2017/timeseries/'

# Absolute value method
#newpath = 'absolutevalue' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980, \
    3827, 4914, 6310, 8103, 10404, 13359, 17154, 22026, 28282, 36315, \
    46630, 59874, 76879, 98715, 126753, 162754, 208981, 268337, 344551], \
    dtype=int)

#for i in range(0, len(families)):
#    filename = families[i]
#    H = absolutevalue(dirname, filename, m)

#os.rename('absolutevalue', 'absolutevalue_Shelly')

# Variance method
#newpath = 'variance' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, len(families)):
#    filename = families[i]
#    d = variance(dirname, filename, m)

#os.rename('variance', 'variance_Shelly')

# Variance method (from Moulines's paper)
#newpath = 'variancemoulines' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, len(families)):
#    filename = families[i]
#    H = variance_moulines(dirname, filename, m)

#os.rename('variancemoulines', 'variancemoulines_Shelly')

# Variance of residuals method
#newpath = 'varianceresiduals' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, len(families)):
#    filename = families[i]
#    d = varianceresiduals(dirname, filename, m, 'mean')

#os.rename('varianceresiduals', 'varianceresiduals_Shelly')

# R/S method
#newpath = 'RS' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, len(families)):
#    filename = families[i]
#    d = RS(dirname, filename, m)

#os.rename('RS', 'RS_Shelly')

# Periodogram method
newpath = 'periodogram' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

dt = 60.0

for i in range(0, len(families)):
    filename = families[i]
    d = periodogram(dirname, filename, dt)

os.rename('periodogram', 'periodogram_Shelly')
