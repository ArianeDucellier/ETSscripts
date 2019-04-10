import numpy as np
import pandas as pd
import pickle

from scipy.io import loadmat

# Chestler's catalog
# ------------------

# Get the names of the template detection files
data = loadmat('../../data/Chestler_2017/LFEsAll.mat')
LFEs = data['LFEs']
nt = len(LFEs)

latitude = np.zeros(nt)
longitude = np.zeros(nt)

# Get the location of the LFE families
for n in range(0, nt):
    latitude[n] = LFEs[n]['lat'][0][0][0]
    longitude[n] = LFEs[n]['lon'][0][0][0]

# Open result file
results = pickle.load(open('../../longrange/Chestler_2017.pkl', 'rb'))[0]
		
# Create dataframe
data = pd.DataFrame(data={'longitude': longitude, \
                          'latitude': latitude})
data['d_var'] = results['d_var']

data0 = data.loc[(data['d_var'] <= 0.0) | (data['d_var'] >= 0.5)]
data1 = data.loc[(data['d_var'] > 0.0) & (data['d_var'] <= 0.1)]
data2 = data.loc[(data['d_var'] > 0.1) & (data['d_var'] <= 0.2)]
data3 = data.loc[(data['d_var'] > 0.2) & (data['d_var'] <= 0.3)]
data4 = data.loc[(data['d_var'] > 0.3) & (data['d_var'] <= 0.4)]
data5 = data.loc[(data['d_var'] > 0.4) & (data['d_var'] < 0.5)]

data0 = data0.drop(columns=['d_var'])
data1 = data1.drop(columns=['d_var'])
data2 = data2.drop(columns=['d_var'])
data3 = data3.drop(columns=['d_var'])
data4 = data4.drop(columns=['d_var'])
data5 = data5.drop(columns=['d_var'])

# Write input files
tfile = open('chestler0.txt', 'w')
tfile.write(data0.to_string(header=False, index=False))
tfile.close()
tfile = open('chestler1.txt', 'w')
tfile.write(data1.to_string(header=False, index=False))
tfile.close()
tfile = open('chestler2.txt', 'w')
tfile.write(data2.to_string(header=False, index=False))
tfile.close()
tfile = open('chestler3.txt', 'w')
tfile.write(data3.to_string(header=False, index=False))
tfile.close()
tfile = open('chestler4.txt', 'w')
tfile.write(data4.to_string(header=False, index=False))
tfile.close()
tfile = open('chestler5.txt', 'w')
tfile.write(data5.to_string(header=False, index=False))
tfile.close()

# Frank's catalog
# ---------------

# Read the LFE file
LFEtime = pd.read_csv('../../data/Frank_2014/frank_jgr_2014_lfe_catalog.txt', \
    delim_whitespace=True, header=None, skiprows=0)
LFEtime.columns = ['year', 'month', 'day', 'hour', 'minute', 'second', \
    'ID', 'latitude', 'longitude', 'depth']
LFEtime['ID'] = LFEtime.ID.astype('category')
families = LFEtime['ID'].cat.categories.tolist()

latitude = np.zeros(len(families))
longitude = np.zeros(len(families))

for n in range(0, len(families)):
    latitude[n] = LFEtime[LFEtime.ID == families[n]].latitude.mode()[0]
    longitude[n] = LFEtime[LFEtime.ID == families[n]].longitude.mode()[0]

# Open result file
results = pickle.load(open('../../longrange/Frank_2014.pkl', 'rb'))[0]
		
# Create dataframe
data = pd.DataFrame(data={'longitude': longitude, \
                          'latitude': latitude})
data['d_var'] = results['d_var']

data0 = data.loc[(data['d_var'] <= 0.0) | (data['d_var'] >= 0.5)]
data1 = data.loc[(data['d_var'] > 0.0) & (data['d_var'] <= 0.1)]
data2 = data.loc[(data['d_var'] > 0.1) & (data['d_var'] <= 0.2)]
data3 = data.loc[(data['d_var'] > 0.2) & (data['d_var'] <= 0.3)]
data4 = data.loc[(data['d_var'] > 0.3) & (data['d_var'] <= 0.4)]
data5 = data.loc[(data['d_var'] > 0.4) & (data['d_var'] < 0.5)]

data0 = data0.drop(columns=['d_var'])
data1 = data1.drop(columns=['d_var'])
data2 = data2.drop(columns=['d_var'])
data3 = data3.drop(columns=['d_var'])
data4 = data4.drop(columns=['d_var'])
data5 = data5.drop(columns=['d_var'])

# Write input files
tfile = open('frank0.txt', 'w')
tfile.write(data0.to_string(header=False, index=False))
tfile.close()
tfile = open('frank1.txt', 'w')
tfile.write(data1.to_string(header=False, index=False))
tfile.close()
tfile = open('frank2.txt', 'w')
tfile.write(data2.to_string(header=False, index=False))
tfile.close()
tfile = open('frank3.txt', 'w')
tfile.write(data3.to_string(header=False, index=False))
tfile.close()
tfile = open('frank4.txt', 'w')
tfile.write(data4.to_string(header=False, index=False))
tfile.close()
tfile = open('frank5.txt', 'w')
tfile.write(data5.to_string(header=False, index=False))
tfile.close()

# Plourde's catalog
# -----------------

# Get the names of the template detection files
LFEloc = np.loadtxt('../../data/Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Open result file
results = pickle.load(open('../../longrange/Plourde_2015.pkl', 'rb'))[0]
		
# Create dataframe
data = pd.DataFrame(data={'longitude': LFEloc['lon'], \
                          'latitude': LFEloc['lat']})
data['d_var'] = results['d_var']

data0 = data.loc[(data['d_var'] <= 0.0) | (data['d_var'] >= 0.5)]
data1 = data.loc[(data['d_var'] > 0.0) & (data['d_var'] <= 0.1)]
data2 = data.loc[(data['d_var'] > 0.1) & (data['d_var'] <= 0.2)]
data3 = data.loc[(data['d_var'] > 0.2) & (data['d_var'] <= 0.3)]
data4 = data.loc[(data['d_var'] > 0.3) & (data['d_var'] <= 0.4)]
data5 = data.loc[(data['d_var'] > 0.4) & (data['d_var'] < 0.5)]

data0 = data0.drop(columns=['d_var'])
data1 = data1.drop(columns=['d_var'])
data2 = data2.drop(columns=['d_var'])
data3 = data3.drop(columns=['d_var'])
data4 = data4.drop(columns=['d_var'])
data5 = data5.drop(columns=['d_var'])

# Write input files
tfile = open('plourde0.txt', 'w')
tfile.write(data0.to_string(header=False, index=False))
tfile.close()
tfile = open('plourde1.txt', 'w')
tfile.write(data1.to_string(header=False, index=False))
tfile.close()
tfile = open('plourde2.txt', 'w')
tfile.write(data2.to_string(header=False, index=False))
tfile.close()
tfile = open('plourde3.txt', 'w')
tfile.write(data3.to_string(header=False, index=False))
tfile.close()
tfile = open('plourde4.txt', 'w')
tfile.write(data4.to_string(header=False, index=False))
tfile.close()
tfile = open('plourde5.txt', 'w')
tfile.write(data5.to_string(header=False, index=False))
tfile.close()

# Shelly's catalog
# ----------------

# Read the LFE file
LFEtime = pd.read_csv('../../data/Shelly_2017/jgrb52060-sup-0002-datas1.txt', \
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
results = pickle.load(open('../../longrange/Shelly_2017.pkl', 'rb'))[0]
		
# Create dataframe
data = pd.DataFrame(data={'longitude': longitude, \
                          'latitude': latitude})
data['d_var'] = results['d_var']

data0 = data.loc[(data['d_var'] <= 0.0) | (data['d_var'] >= 0.5)]
data1 = data.loc[(data['d_var'] > 0.0) & (data['d_var'] <= 0.1)]
data2 = data.loc[(data['d_var'] > 0.1) & (data['d_var'] <= 0.2)]
data3 = data.loc[(data['d_var'] > 0.2) & (data['d_var'] <= 0.3)]
data4 = data.loc[(data['d_var'] > 0.3) & (data['d_var'] <= 0.4)]
data5 = data.loc[(data['d_var'] > 0.4) & (data['d_var'] < 0.5)]

data0 = data0.drop(columns=['d_var'])
data1 = data1.drop(columns=['d_var'])
data2 = data2.drop(columns=['d_var'])
data3 = data3.drop(columns=['d_var'])
data4 = data4.drop(columns=['d_var'])
data5 = data5.drop(columns=['d_var'])

# Write input files
tfile = open('shelly0.txt', 'w')
tfile.write(data0.to_string(header=False, index=False))
tfile.close()
tfile = open('shelly1.txt', 'w')
tfile.write(data1.to_string(header=False, index=False))
tfile.close()
tfile = open('shelly2.txt', 'w')
tfile.write(data2.to_string(header=False, index=False))
tfile.close()
tfile = open('shelly3.txt', 'w')
tfile.write(data3.to_string(header=False, index=False))
tfile.close()
tfile = open('shelly4.txt', 'w')
tfile.write(data4.to_string(header=False, index=False))
tfile.close()
tfile = open('shelly5.txt', 'w')
tfile.write(data5.to_string(header=False, index=False))
tfile.close()

# Sweet's catalog
# ---------------

# Number of LFE families
nf = 9

latitude = np.zeros(nf)
longitude = np.zeros(nf)

# Get the location of the LFE families
for n in range(0, nf):
    data = loadmat('../../data/Sweet_2014/catalogs/LFE' + str(n + 1) + 'catalog.mat')
    latitude[n] = data['LFE' + str(n + 1)][0][0][1][0][0]
    longitude[n] = data['LFE' + str(n + 1)][0][0][2][0][0]

# Open result file
results = pickle.load(open('../../longrange/Sweet_2014.pkl', 'rb'))[0]
		
# Create dataframe
data = pd.DataFrame(data={'longitude': longitude, \
                          'latitude': latitude})
data['d_var'] = results['d_var']

data0 = data.loc[(data['d_var'] <= 0.0) | (data['d_var'] >= 0.5)]
data1 = data.loc[(data['d_var'] > 0.0) & (data['d_var'] <= 0.1)]
data2 = data.loc[(data['d_var'] > 0.1) & (data['d_var'] <= 0.2)]
data3 = data.loc[(data['d_var'] > 0.2) & (data['d_var'] <= 0.3)]
data4 = data.loc[(data['d_var'] > 0.3) & (data['d_var'] <= 0.4)]
data5 = data.loc[(data['d_var'] > 0.4) & (data['d_var'] < 0.5)]

data0 = data0.drop(columns=['d_var'])
data1 = data1.drop(columns=['d_var'])
data2 = data2.drop(columns=['d_var'])
data3 = data3.drop(columns=['d_var'])
data4 = data4.drop(columns=['d_var'])
data5 = data5.drop(columns=['d_var'])

# Write input files
tfile = open('sweet0.txt', 'w')
tfile.write(data0.to_string(header=False, index=False))
tfile.close()
tfile = open('sweet1.txt', 'w')
tfile.write(data1.to_string(header=False, index=False))
tfile.close()
tfile = open('sweet2.txt', 'w')
tfile.write(data2.to_string(header=False, index=False))
tfile.close()
tfile = open('sweet3.txt', 'w')
tfile.write(data3.to_string(header=False, index=False))
tfile.close()
tfile = open('sweet4.txt', 'w')
tfile.write(data4.to_string(header=False, index=False))
tfile.close()
tfile = open('sweet5.txt', 'w')
tfile.write(data5.to_string(header=False, index=False))
tfile.close()
