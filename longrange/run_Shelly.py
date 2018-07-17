"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Shelly (2017)
"""

import numpy as np
import pandas as pd

from test_long_range import variance_moulines, absolutevalue, variance

# Read the LFE file
LFEtime = pd.read_csv('../data/Shelly_2017/jgrb52060-sup-0002-datas1.txt', \
    delim_whitespace=True, header=None, skiprows=2)
LFEtime.columns = ['year', 'month', 'day', 's_of_day', 'hr', 'min', 'sec', \
    'ccsum', 'meancc', 'med_cc', 'seqday', 'ID', 'latitude', 'longitude', \
    'depth', 'n_chan']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

# We will look at the following window sizes (in minutes)
m = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, \
    20000, 50000, 100000], dtype=int)
dirname = '../data/Shelly_2017/timeseries/'

# Loop on LFE families
for i in range(0, len(families)):
    filename = families[i]
#    H = variance_moulines(dirname, filename, m)
#    H = absolutevalue(dirname, filename, m)
    d = variance(dirname, filename, m)
