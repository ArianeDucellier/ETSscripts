"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Chestler and Creager
(2017)
"""

import numpy as np
import os

from scipy.io import loadmat

from test_long_range import absolutevalue
from test_long_range import variance
from test_long_range import variance_moulines
from test_long_range import varianceresiduals
from test_long_range import RS

# Get the names of the template detection files
data = loadmat('../data/Chestler_2017/LFEsAll.mat')
LFEs = data['LFEs']
nt = len(LFEs)

dirname = '../data/Chestler_2017/timeseries/'

# Absolute value method
#newpath = 'absolutevalue' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980, \
    3827, 4914, 6310, 8103, 10404, 13359, 17154, 22026, 28282, 36315, \
    46630], dtype=int)

#for i in range(0, nt):
#    LFEs = data['LFEs'][i]
#    filename = LFEs['name'][0][0]
#    H = absolutevalue(dirname, filename, m)

#os.rename('absolutevalue', 'absolutevalue_Chestler')

# Variance method
#newpath = 'variance' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, nt):
#    LFEs = data['LFEs'][i]
#    filename = LFEs['name'][0][0]
#    d = variance(dirname, filename, m)

#os.rename('variance', 'variance_Chestler')

# Variance method (from Moulines's paper)
#newpath = 'variancemoulines' 
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, nt):
#    LFEs = data['LFEs'][i]
#    filename = LFEs['name'][0][0]
#    H = variance_moulines(dirname, filename, m)

#os.rename('variancemoulines', 'variancemoulines_Chestler')

# Variance of residuals method
#newpath = 'varianceresiduals'
#if not os.path.exists(newpath):
#    os.makedirs(newpath)

#for i in range(0, nt):
#    LFEs = data['LFEs'][i]
#    filename = LFEs['name'][0][0]
#    d = varianceresiduals(dirname, filename, m, 'mean')

#os.rename('varianceresiduals', 'varianceresiduals_Chestler')

# R/S method
newpath = 'RS' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0] 
    d = RS(dirname, filename, m)

os.rename('RS', 'RS_Chestler')
