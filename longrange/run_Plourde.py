"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Plourde et al. (2015)
"""

import numpy as np

from datetime import datetime

import test_long_range

# Get the names of the template detection files
templates = np.loadtxt('../data/LFEcatalog/template_locations.txt', \
                dtype={'names': ('name', 'day', 'uk1', 'uk2', 'lat1', 'lat2', \
                'lon1', 'lon2', 'uk3', 'uk4', 'uk5', 'uk6'), \
                     'formats': ('S13', 'S10', np.int, np.float, np.int, \
                np.float, np.int, np.float, np.float, np.float, np.float, \
                np.float)})

# Beginning and end of the period we are looking at
tbegin = datetime(2008, 3, 1, 0, 0, 0)
tend = datetime(2008, 5,  1, 0, 0, 0)

# We construct the time series by counting the number of LFEs
# per one-minute-long time window
window = 60.0

# We will look at the following window sizes (in minutes)
#m = np.array([1, 2, 3, 5, 8, 10, 20, 30, 50, 80, 100, 200, 300, 500, 800, \
#    1000, 2000, 3000, 5000], dtype=int)
m = np.array([5, 10, 50, 100, 500, 1000], dtype=int)

# Loop on templates
for i in range(0, 1): #np.shape(templates)[0]):
    filename = templates[i][0].astype(str)
    X = test_long_range.get_time_series(filename, window, tbegin, tend)
    # Absolute Value Method
    output_name = 'absolutevalue/' + filename + '.eps'
    test_long_range.absolutevalue(X, m, output_name)
    # Variance Method
    output_name = 'variance/' + filename + '.eps'
    test_long_range.variance(X, m, output_name)

    test_long_range.compute_variance(filename, m, tbegin, tend)
