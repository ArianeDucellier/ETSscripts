"""
This script makes maps of the Hurst parameter and the fractional index
for the LFE catalog of Shelly (2017)
"""
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

# Read the LFE file
LFEtime = pd.read_csv('../data/Shelly_2017/jgrb52060-sup-0002-datas1.txt', \
    delim_whitespace=True, header=None, skiprows=2)
LFEtime.columns = ['year', 'month', 'day', 's_of_day', 'hr', 'min', \
    'sec', 'ccsum', 'meancc', 'med_cc', 'seqday', 'ID', 'latitude', \
    'longitude', 'depth', 'n_chan']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

latitude = np.zeros(len(families))
longitude = np.zeros(len(families))

for n in range(0, len(families)):
    latitude[n] = LFEtime[LFEtime.ID == families[n]].latitude.mode()[0]
    longitude[n] = LFEtime[LFEtime.ID == families[n]].longitude.mode()[0]

# Open result file
results = pickle.load(open('Shelly_2017.pkl', 'rb'))[0]
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
plt.savefig('scatter_Shelly.eps', format='eps')
plt.close()

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='H_absval:Q'
).properties(
    width=400,
	height=400
)
myChart.save('maps/H_absval_Shelly.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_var:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/d_var_Shelly.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='H_varm:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/H_varm_Shelly.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_varres:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/d_varres_Shelly.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_RS:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/d_RS_Shelly.png')

myChart = alt.Chart(data).mark_circle(size=50).encode(
    longitude='longitude:Q',
    latitude='latitude:Q',
    color='d_p:Q'
).properties(
    width=400,
    height=400
)
myChart.save('maps/d_p_Shelly.png')
