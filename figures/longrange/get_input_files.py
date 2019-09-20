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
data['color'] = results['d_var']
data['size'] = results['d_var']

# Write input files
tfile = open('chestler.txt', 'w')
tfile.write(data.to_string(header=False, index=False))
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
data['color'] = results['d_var']
data['size'] = results['d_var']

# Write input files
tfile = open('frank.txt', 'w')
tfile.write(data.to_string(header=False, index=False))
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
data['color'] = results['d_var']
data['size'] = results['d_var']

# Write input files
tfile = open('plourde.txt', 'w')
tfile.write(data.to_string(header=False, index=False))
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
data['color'] = results['d_var']
data['size'] = results['d_var']

# Write input files
tfile = open('shelly.txt', 'w')
tfile.write(data.to_string(header=False, index=False))
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
data['color'] = results['d_var']
data['size'] = results['d_var']

# Write input files
tfile = open('sweet.txt', 'w')
tfile.write(data.to_string(header=False, index=False))
tfile.close()
