"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Chestler and Creager
(2017)
"""

import numpy as np

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

# We will look at the following window sizes (in minutes)
m = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], dtype=int)

#for i in range(0, nt):
#    LFEs = data['LFEs'][i]
#    filename = LFEs['name'][0][0]   
#    H = variance_moulines(dirname, filename, m)
#    H = absolutevalue(dirname, filename, m)
#    d = variance(dirname, filename, m)

# For variance residuals, we look at the following sizes (in minutes)
m = np.array([50, 100, 200, 500, 1000, 2000, 5000], dtype=int)

for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0]  
    d = varianceresiduals(dirname, filename, m, 'mean')
