"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Plourde et al. (2015)
"""

import numpy as np
import os

from test_long_range import absolutevalue
from test_long_range import variance
from test_long_range import variance_moulines
from test_long_range import varianceresiduals
from test_long_range import RS

# Get the names of the template detection files
templates = np.loadtxt('../data/Plourde_2015/template_locations.txt', \
    dtype={'names': ('name', 'day', 'uk1', 'uk2', 'lat1', 'lat2', \
    'lon1', 'lon2', 'uk3', 'uk4', 'uk5', 'uk6'), \
         'formats': ('S13', 'S10', np.int, np.float, np.int, np.float, \
    np.int, np.float, np.float, np.float, np.float, np.float)})

dirname = '../data/Plourde_2015/timeseries/'

# Absolute value method
#newpath = 'absolutevalue' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980], \
    dtype=int)

#for i in range(0, np.shape(templates)[0]):
#    filename = templates[i][0].astype(str)   
#    H = absolutevalue(dirname, filename, m)

#os.rename('absolutevalue', 'absolutevalue_Plourde')

# Variance method
#newpath = 'variance' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, np.shape(templates)[0]):
#    filename = templates[i][0].astype(str)   
#    d = variance(dirname, filename, m)

#os.rename('variance', 'variance_Plourde')

# Variance method (from Moulines's paper)
#newpath = 'variancemoulines' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, np.shape(templates)[0]):
#    filename = templates[i][0].astype(str)   
#    H = variance_moulines(dirname, filename, m)

#os.rename('variancemoulines', 'variancemoulines_Plourde')

# Variance of residuals method
#newpath = 'varianceresiduals' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, np.shape(templates)[0]):
#    filename = templates[i][0].astype(str)   
#    d = varianceresiduals(dirname, filename, m, 'mean')

#os.rename('varianceresiduals', 'varianceresiduals_Plourde')

# R/S method
newpath = 'RS' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(0, np.shape(templates)[0]):
    filename = templates[i][0].astype(str)   
    d = RS(dirname, filename, m)

os.rename('RS', 'RS_Plourde')
