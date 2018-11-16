"""
This script draws the results of the tests with the module draw_long_range
for the LFE catalog of Plourde et al. (2015)
"""
import numpy as np
import os

from draw_long_range import draw_absolutevalue
from draw_long_range import draw_variance
from draw_long_range import draw_variance_moulines
from draw_long_range import draw_varianceresiduals
from draw_long_range import draw_RSstatistic
from draw_long_range import draw_periodogram

# Get the names of the template detection files
LFEloc = np.loadtxt('../data/Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Absolute value method
for i in range(0, len(LFEloc)):
    filename = LFEloc[i][0].decode('utf-8')   
    draw_absolutevalue(filename)

os.rename('absolutevalue', 'absolutevalue_Plourde')

# Variance method
newpath = 'variance' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(0, len(LFEloc)):
    filename = LFEloc[i][0].decode('utf-8')    
    draw_variance(filename)

os.rename('variance', 'variance_Plourde')

# Variance method (from Moulines's paper)
newpath = 'variancemoulines' 
for i in range(0, len(LFEloc)):
    filename = LFEloc[i][0].decode('utf-8')
    draw_variance_moulines(filename)

os.rename('variancemoulines', 'variancemoulines_Plourde')

# Variance of residuals method
for i in range(0, len(LFEloc)):
    filename = LFEloc[i][0].decode('utf-8')
    draw_varianceresiduals(filename, 'mean')

os.rename('varianceresiduals', 'varianceresiduals_Plourde')

# R/S method
for i in range(0, len(LFEloc)):
    filename = LFEloc[i][0].decode('utf-8')
    draw_RSstatistic(filename)

os.rename('RS', 'RS_Plourde')

# Periodogram method
for i in range(0, len(LFEloc)):
    filename = LFEloc[i][0].decode('utf-8')
    draw_periodogram(filename)

os.rename('periodogram', 'periodogram_Plourde')
