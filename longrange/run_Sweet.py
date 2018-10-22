"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Sweet (2014)
"""

import numpy as np
import os

from test_long_range import absolutevalue
from test_long_range import periodogram
from test_long_range import variance
from test_long_range import variance_moulines
from test_long_range import varianceresiduals
from test_long_range import RS

# Get the number of LFE families
nf = 9

dirname = '../data/Sweet_2014/timeseries/'

# Absolute value method
#newpath = 'absolutevalue' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980, \
    3827, 4914, 6310, 8103, 10404, 13359, 17154, 22026, 28282, 36315, \
    46630, 59874, 76879, 98715], dtype=int)

#for i in range(0, nf):
#    filename = 'LFE' + str(i + 1) 
#    H = absolutevalue(dirname, filename, m)

#os.rename('absolutevalue', 'absolutevalue_Sweet')

# Variance method
#newpath = 'variance' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, nf):
#    filename = 'LFE' + str(i + 1) 
#    d = variance(dirname, filename, m)

#os.rename('variance', 'variance_Sweet')

# Variance method (from Moulines's paper)
#newpath = 'variancemoulines' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, nf):
#    filename = 'LFE' + str(i + 1) 
#    H = variance_moulines(dirname, filename, m)

#os.rename('variancemoulines', 'variancemoulines_Sweet')

# Variance of residuals method
#newpath = 'varianceresiduals' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, nf):
#    filename = 'LFE' + str(i + 1) 
#    d = varianceresiduals(dirname, filename, m, 'mean')

#os.rename('varianceresiduals', 'varianceresiduals_Sweet')

# R/S method
#newpath = 'RS' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, nf):
#    filename = 'LFE' + str(i + 1) 
#    d = RS(dirname, filename, m)

#os.rename('RS', 'RS_Sweet')

# Periodogram method
newpath = 'periodogram' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

dt = 60.0

for i in range(0, nf):
    filename = 'LFE' + str(i + 1) 
    d = periodogram(dirname, filename, dt)

os.rename('periodogram', 'periodogram_Sweet')
