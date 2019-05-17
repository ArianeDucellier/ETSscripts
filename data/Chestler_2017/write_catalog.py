"""
This script looks at the catalog from Chestler and Creager (2017)) and
transform the LFE detections for each template into a time series
"""

import numpy as np
import pickle
import pandas as pd

from scipy.io import loadmat

from date import matlab2ymdhms, ymdhms2matlab

def write_catalog(n, tbegin, tend):
    """
    Function to write the LFE catalog into a file suitable
    to be read and analyzed with the R package bayesianETAS

    Input:
        type n = int
        n = Index of the LFE template
        type tbegin = tuple of 6 integers
        tbegin = Beginning time of the catalog
                 (year, month, day, hour, minute, second)
        type tend = tuple of 6 integers
        tend = End time of the catalog
               (year, month, day, hour, minute, second)
    Output:
        type df = pandas dataframe
        df = Time and magnitude of LFEs
    """
    # Get the time of LFE detections
    data = loadmat('LFEsAll.mat')
    LFEs = data['LFEs'][n]
    LFEtime = LFEs['peakTimesMo'][0]
    LFEmag = LFEs['Mw'][0]
    daybegin = ymdhms2matlab(tbegin[0], tbegin[1], tbegin[2], \
        tbegin[3], tbegin[4], tbegin[5])
    dayend = ymdhms2matlab(tend[0], tend[1], tend[2], \
        tend[3], tend[4], tend[5])
    time = LFEtime - daybegin
    df = pd.DataFrame(data={'time':time[:, 0], 'mw':LFEmag[:, 0]})
    return df

if __name__ == '__main__':

    # Get the names of the template detection files
    data = loadmat('LFEsAll.mat')
    LFEs = data['LFEs']
    nt = len(LFEs)

    # Beginning and end of the period we are looking at
    tbegin = (2009, 6, 1, 0, 0, 0)
    tend = (2011, 9, 1, 0, 0, 0)

    # Loop on templates
    for n in range(0, nt):
        df = write_catalog(n, tbegin, tend)
        filename = LFEs[n]['name'][0][0]
        output = 'catalogs/{}.txt'.format(filename)
        tfile = open(output, 'w')
        tfile.write(df.to_string())
        tfile.close()
