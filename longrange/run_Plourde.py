"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Plourde et al. (2015)
"""

import numpy as np

from test_long_range import variance_moulines, absolutevalue, variance

# Get the names of the template detection files
templates = np.loadtxt('../data/LFEcatalog/template_locations.txt', \
    dtype={'names': ('name', 'day', 'uk1', 'uk2', 'lat1', 'lat2', \
    'lon1', 'lon2', 'uk3', 'uk4', 'uk5', 'uk6'), \
         'formats': ('S13', 'S10', np.int, np.float, np.int, np.float, \
    np.int, np.float, np.float, np.float, np.float, np.float)})

# We will look at the following window sizes (in minutes)
m = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000], dtype=int)
dirname = '../data/LFEcatalog/timeseries/'

for i in range(0, np.shape(templates)[0]):
    filename = templates[i][0].astype(str)   
#    H = variance_moulines(dirname, filename, m)
#    H = absolutevalue(dirname, filename, m)
    d = variance(dirname, filename, m)
