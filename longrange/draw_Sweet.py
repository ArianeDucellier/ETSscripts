"""
This script draws the results of the tests with the module draw_long_range
for the LFE catalog of Sweet (2014)
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
nf = 9

# Absolute value method
for i in range(0, nf):
    filename = 'LFE' + str(i + 1)
    draw_absolutevalue(filename)

os.rename('absolutevalue', 'absolutevalue_Sweet')

# Variance method
newpath = 'variance' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(0, nf):
    filename = 'LFE' + str(i + 1)
    draw_variance(filename)

os.rename('variance', 'variance_Sweet')

# Variance method (from Moulines's paper)
newpath = 'variancemoulines' 
for i in range(0, nf):
    filename = 'LFE' + str(i + 1)
    draw_variance_moulines(filename)

os.rename('variancemoulines', 'variancemoulines_Sweet')

# Variance of residuals method
for i in range(0, nf):
    filename = 'LFE' + str(i + 1)
    draw_varianceresiduals(filename, 'mean')

os.rename('varianceresiduals', 'varianceresiduals_Sweet')

# R/S method
for i in range(0, nf):
    filename = 'LFE' + str(i + 1)
    draw_RSstatistic(filename)

os.rename('RS', 'RS_Sweet')

# Periodogram method
for i in range(0, nf):
    filename = 'LFE' + str(i + 1)
    draw_periodogram(filename)

os.rename('periodogram', 'periodogram_Sweet')
