"""
This script draws the results of the tests with the module draw_long_range
for the LFE catalog of Frank et al. (2014)
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
LFEtime = pd.read_csv('../data/Frank_2014/frank_jgr_2014_lfe_catalog.txt', \
					  delim_whitespace=True, header=None, skiprows=0)
LFEtime.columns = ['year', 'month', 'day', 'hour', 'minute', 'second', \
				   'ID', 'latitude', 'longitude', 'depth']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

# Absolute value method
#for i in range(0, len(families)):
#    filename = str(families[i])   
#    draw_absolutevalue(filename)

#os.rename('absolutevalue', 'absolutevalue_Frank')

# Variance method
#for i in range(0, len(families)):
#    filename = str(families[i])    
#    draw_variance(filename)

#os.rename('variance', 'variance_Frank')

# Variance method (from Moulines's paper)
#for i in range(0, len(families)):
#    filename = str(families[i])
#    draw_variance_moulines(filename)

#os.rename('variancemoulines', 'variancemoulines_Frank')

# Variance of residuals method
#for i in range(0, len(families)):
#    filename = str(families[i])
#    draw_varianceresiduals(filename, 'mean')

#os.rename('varianceresiduals', 'varianceresiduals_Frank')

# R/S method
for i in range(0, len(families)):
    filename = str(families[i])
    draw_RSstatistic(filename)

os.rename('RS', 'RS_Frank')

# Periodogram method
for i in range(0, len(families)):
    filename = str(families[i])
    draw_periodogram(filename)

os.rename('periodogram', 'periodogram_Frank')
