"""
This script makes maps of the Hurst parameter and the fractional index
for the LFE catalog of Sweet (2014))
"""
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from scipy.io import loadmat

# Number of LFE families
nf = 9

latitude = np.zeros(nf)
longitude = np.zeros(nf)

# Get the location of the LFE families
for n in range(0, nf):
    data = loadmat('../data/Sweet_2014/catalogs/LFE' + str(n + 1) + 'catalog.mat')
    latitude[n] = data['LFE' + str(n + 1)][0][0][1][0][0]
    longitude[n] = data['LFE' + str(n + 1)][0][0][2][0][0]

# Open result file
results = pickle.load(open('Sweet_2014.pkl', 'rb'))[0]
results.drop(['family'], axis=1)
		
# Create dataframe
data = pd.DataFrame(data={'latitude': latitude, \
					      'longitude': longitude})
data['H_absval'] = results['H_absval']
data['d_var'] = results['d_var']
data['H_varm'] = results['H_varm']
data['d_varres'] = results['d_varres']
data['d_RS'] = results['d_RS']
data['d_p'] = results['d_p']

pd.plotting.scatter_matrix(results, figsize=(15, 15))
plt.savefig('scatter_Sweet.eps', format='eps')
plt.close()

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='H_absval:Q'
).properties(
    width=400,
	height=400
)
myChart.save('maps/H_absval_Sweet.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_var:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/d_var_Sweet.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='H_varm:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/H_varm_Sweet.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_varres:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/d_varres_Sweet.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_RS:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/d_RS_Sweet.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_p:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/d_p_Sweet.png')
