"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Chestler and Creager
(2017)
"""

import numpy as np
import os

from scipy.io import loadmat

from test_long_range import variance_moulines
from test_long_range import absolutevalue
from test_long_range import variance
from test_long_range import varianceresiduals

# Get the names of the template detection files
data = loadmat('../data/Chestler_2017/LFEsAll.mat')
LFEs = data['LFEs']
nt = len(LFEs)

dirname = '../data/Chestler_2017/timeseries/'

# Absolute value method
newpath = 'absolutevalue' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980, \
    3827, 4914, 6310, 8103, 10404, 13359, 17154, 22026, 28282, 36315, \
    46630], dtype=int)

for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0]
    H = absolutevalue(dirname, filename, m)

os.rename('absolutevalue', 'absolutevalue_Chestler')

#    H = variance_moulines(dirname, filename, m)
#    
#    d = variance(dirname, filename, m)

# For variance residuals, we look at the following sizes (in minutes)
#m = np.array([50, 100, 200, 500, 1000, 2000, 5000], dtype=int)
#
#for i in range(0, nt):
#    LFEs = data['LFEs'][i]
#    filename = LFEs['name'][0][0]  
#    d = varianceresiduals(dirname, filename, m, 'mean')
