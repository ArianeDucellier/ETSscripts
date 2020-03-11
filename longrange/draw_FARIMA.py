"""
This script draws the results of the tests with the module draw_long_range
for the synthetic FARIMA time series
"""
import numpy as np
import os

from draw_long_range import draw_absolutevalue
from draw_long_range import draw_variance
from draw_long_range import draw_variance_moulines
from draw_long_range import draw_varianceresiduals
from draw_long_range import draw_RSstatistic
from draw_long_range import draw_periodogram

files = ['series1_1', 'series1_2', 'series1_3', 'series1_4', 'series1_5', \
         'series2_1', 'series2_2', 'series2_3', 'series2_4', 'series2_5', \
         'series3_1', 'series3_2', 'series3_3', 'series3_4', 'series3_5', \
         'series4_1', 'series4_2', 'series4_3', 'series4_4', 'series4_5', \
         'series5_1', 'series5_2', 'series5_3', 'series5_4', 'series5_5', \
         'series6_1', 'series6_2', 'series6_3', 'series6_4', 'series6_5']

# Absolute value method
for i in range(0, len(files)):
    draw_absolutevalue(files[i])

os.rename('absolutevalue', 'absolutevalue_FARIMA')

# Variance method
for i in range(0, len(files)):
    draw_variance(files[i])

os.rename('variance', 'variance_FARIMA')

# Variance method (from Moulines's paper)
for i in range(0, len(files)):
    draw_variance_moulines(files[i])

os.rename('variancemoulines', 'variancemoulines_FARIMA')

# Variance of residuals method
for i in range(0, len(files)):
    draw_varianceresiduals(files[i], 'mean')

os.rename('varianceresiduals', 'varianceresiduals_FARIMA')

# R/S method
for i in range(0, len(files)):
    draw_RSstatistic(files[i])

os.rename('RS', 'RS_FARIMA')

# Periodogram method
for i in range(0, len(files)):
    draw_periodogram(files[i])

os.rename('periodogram', 'periodogram_FARIMA')
