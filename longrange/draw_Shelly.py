"""
This script draws the results of the tests with the module draw_long_range
for the LFE catalog of Shelly (2017)
"""
import numpy as np
import os
import pandas as pd

from draw_long_range import draw_absolutevalue
from draw_long_range import draw_variance
from draw_long_range import draw_variance_moulines
from draw_long_range import draw_varianceresiduals
from draw_long_range import draw_RSstatistic
from draw_long_range import draw_periodogram

# Read the LFE file
LFEtime = pd.read_csv('../data/Shelly_2017/jgrb52060-sup-0002-datas1.txt', \
    delim_whitespace=True, header=None, skiprows=2)
LFEtime.columns = ['year', 'month', 'day', 's_of_day', 'hr', 'min', 'sec', \
    'ccsum', 'meancc', 'med_cc', 'seqday', 'ID', 'latitude', 'longitude', \
    'depth', 'n_chan']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

# Absolute value method
for i in range(0, len(families)):
    filename = families[i]   
    draw_absolutevalue(filename)

os.rename('absolutevalue', 'absolutevalue_Shelly')

# Variance method
newpath = 'variance' 
if not os.path.exists(newpath):
    os.makedirs(newpath)

for i in range(0, len(families)):
    filename = families[i]    
    draw_variance(filename)

os.rename('variance', 'variance_Shelly')

# Variance method (from Moulines's paper)
newpath = 'variancemoulines' 
for i in range(0, len(families)):
    filename = families[i]
    draw_variance_moulines(filename)

os.rename('variancemoulines', 'variancemoulines_Shelly')

# Variance of residuals method
for i in range(0, len(families)):
    filename = families[i]
    draw_varianceresiduals(filename, 'mean')

os.rename('varianceresiduals', 'varianceresiduals_Shelly')

# R/S method
for i in range(0, len(families)):
    filename = families[i]
    draw_RSstatistic(filename)

os.rename('RS', 'RS_Shelly')

# Periodogram method
for i in range(0, len(families)):
    filename = families[i]
    draw_periodogram(filename)

os.rename('periodogram', 'periodogram_Shelly')
