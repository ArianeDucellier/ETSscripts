"""
This script draws the results of the tests with the module draw_long_range
for the LFE catalog of Chestler and Creager (2017)
"""
import numpy as np
import os

from scipy.io import loadmat

from draw_long_range import draw_absolutevalue
from draw_long_range import draw_variance
from draw_long_range import draw_variance_moulines
from draw_long_range import draw_varianceresiduals
from draw_long_range import draw_RSstatistic
from draw_long_range import draw_periodogram

# Get the names of the template detection files
data = loadmat('../data/Chestler_2017/LFEsAll.mat')
LFEs = data['LFEs']
nt = len(LFEs)

# Absolute value method
for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0]
    draw_absolutevalue(filename)

os.rename('absolutevalue', 'absolutevalue_Chestler')

# Variance method
newpath = 'variance' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0]
    draw_variance(filename)

os.rename('variance', 'variance_Chestler')

# Variance method (from Moulines's paper)
newpath = 'variancemoulines' 
for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0]
    draw_variance_moulines(filename)

os.rename('variancemoulines', 'variancemoulines_Chestler')

# Variance of residuals method
for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0]
    draw_varianceresiduals(filename, 'mean')

os.rename('varianceresiduals', 'varianceresiduals_Chestler')

# R/S method
for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0]
    draw_RSstatistic(filename)

os.rename('RS', 'RS_Chestler')

# Periodogram method
for i in range(0, nt):
    LFEs = data['LFEs'][i]
    filename = LFEs['name'][0][0]
    draw_periodogram(filename)

os.rename('periodogram', 'periodogram_Chestler')
