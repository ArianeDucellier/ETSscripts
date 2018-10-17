"""
This module contains functions to run the tests for long range dependence
from the module test_long_range with the LFE catalog of Plourde et al. (2015)
"""
import psutil
import numpy as np
import os
import pandas as pd
import pickle

from test_long_range_parallel import absolutevalue
from test_long_range_parallel import variance
from test_long_range_parallel import variance_moulines
from test_long_range_parallel import varianceresiduals
from test_long_range_parallel import RSstatistic

# Get the names of the template detection files
templates = np.loadtxt('../data/Plourde_2015/template_locations.txt', \
    dtype={'names': ('name', 'day', 'uk1', 'uk2', 'lat1', 'lat2', \
    'lon1', 'lon2', 'uk3', 'uk4', 'uk5', 'uk6'), \
         'formats': ('S13', 'S10', np.int, np.float, np.int, np.float, \
    np.int, np.float, np.float, np.float, np.float, np.float)})

dirname = '../data/Plourde_2015/timeseries/'

# Create pandas dataframe to store the results
families = []
for i in range(0, np.shape(templates)[0]):
    filename = templates[i][0].astype(str)
    families.append(filename)

df = pd.DataFrame(data={'family': families})
 
# Absolute value method
newpath = 'absolutevalue' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

m = np.array([4, 5, 7, 9, 12, 15, 20, 25, 33, 42, 54, 70, 90, 115, 148, \
    190, 244, 314, 403, 518, 665, 854, 1096, 1408, 1808, 2321, 2980], \
    dtype=int)

H_absval = np.zeros(np.shape(templates)[0])

for i in range(0, np.shape(templates)[0]):
    filename = templates[i][0].astype(str)   
    H_absval[i] = absolutevalue(dirname, filename, m, True)

os.rename('absolutevalue', 'absolutevalue_Plourde')

df['H_absval'] = H_absval

# Variance method
newpath = 'variance' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_var = np.zeros(np.shape(templates)[0])

for i in range(0, np.shape(templates)[0]):
    filename = templates[i][0].astype(str)   
    d_var[i] = variance(dirname, filename, m, True)

os.rename('variance', 'variance_Plourde')

df['d_var'] = d_var

# Variance method (from Moulines's paper)
newpath = 'variancemoulines' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

H_varm = np.zeros(np.shape(templates)[0])

for i in range(0, np.shape(templates)[0]):
    filename = templates[i][0].astype(str)   
    H_varm[i] = variance_moulines(dirname, filename, m, True)

os.rename('variancemoulines', 'variancemoulines_Plourde')

df['H_varm'] = H_varm

# Variance of residuals method
newpath = 'varianceresiduals' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_varres = np.zeros(np.shape(templates)[0])

for i in range(0, np.shape(templates)[0]):
    proc = psutil.Process()
    print('template {}'.format(i))
    print(proc.open_files())
    filename = templates[i][0].astype(str)   
    d_varres[i] = varianceresiduals(dirname, filename, m, 'mean', True)

os.rename('varianceresiduals', 'varianceresiduals_Plourde')

df['d_varres'] = d_varres

# R/S method
newpath = 'RS' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

d_RS = np.zeros(np.shape(templates)[0])

for i in range(0, np.shape(templates)[0]):
    filename = templates[i][0].astype(str)   
    d_RS[i] = RSstatistic(dirname, filename, m, True)

os.rename('RS', 'RS_Plourde')

df['d_RS'] = d_RS

# Save dataframe into file
filename = 'Plourde_2015.pkl'
pickle.dump([df], open(filename, 'wb'))
