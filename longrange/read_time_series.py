"""
This module contains a function to be used by R
"""

import numpy as np
import pickle

def read_time_series(dirname, filename):
    """
    Read the LFE time series using pickle

    Input:
        type dirname = string
        dirname = Directory where to find the pickle file
        type filename = string
        filename = Name of the file containing the time series
    Output:
        type X = 1D numpy array
        X = Time series
    """
    data = pickle.load(open(dirname + filename + '.pkl', 'rb'))
    X = data[3]
    np.savetxt(filename + '.txt', X)
