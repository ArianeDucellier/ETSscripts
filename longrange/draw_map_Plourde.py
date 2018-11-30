"""
This script makes maps of the Hurst parameter and the fractional index
for the LFE catalog of Plourde et al. (2015)
"""
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Get the names of the template detection files
LFEloc = np.loadtxt('../data/Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Open result file
results = pickle.load(open('Plourde_2015.pkl', 'rb'))[0]
results.drop(['family'], axis=1)
		
# Create dataframe
data = pd.DataFrame(data={'latitude': LFEloc['lat'], \
					      'longitude': LFEloc['lon']})
data['H_absval'] = results['H_absval']
data['d_var'] = results['d_var']
data['H_varm'] = results['H_varm']
data['d_varres'] = results['d_varres']
data['d_RS'] = results['d_RS']
data['d_p'] = results['d_p']

pd.plotting.scatter_matrix(results, figsize=(20, 20))
plt.savefig('scatter_Plourde.eps', format='eps')
plt.close()

myChart = alt.Chart(data).mark_circle(size=10).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='H_absval:Q'
)
myChart.save('maps/H_absval_Plourde.png')

myChart = alt.Chart(data).mark_circle(size=10).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_var:Q'
)
myChart.save('maps/d_var_Plourde.png')

myChart = alt.Chart(data).mark_circle(size=10).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='H_varm:Q'
)
myChart.save('maps/H_varm_Plourde.png')

myChart = alt.Chart(data).mark_circle(size=10).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_varres:Q'
)
myChart.save('maps/d_varres_Plourde.png')

myChart = alt.Chart(data).mark_circle(size=10).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_RS:Q'
)
myChart.save('maps/d_RS_Plourde.png')

myChart = alt.Chart(data).mark_circle(size=10).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_p:Q'
)
myChart.save('maps/d_p_Plourde.png')
