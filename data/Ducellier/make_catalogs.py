"""
Script to read monthly catalogs and put them
into a single file
"""
import numpy as np
import os
import pandas as pd
import pickle

# List of LFE families
templates = np.loadtxt('../Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

for i in range(0, np.shape(templates)[0]):

    # Create directory to store the catalog
    namedir = 'catalogs/' + templates[i][0].astype(str)
    if not os.path.exists(namedir):
        os.makedirs(namedir)

    # Read monthly catalogs
    df_2007_07 = pickle.load(open('catalog_2007_07/' + templates[i][0].astype(str) + '/catalog_2007_07.pkl', 'rb'))
    df_2007_08 = pickle.load(open('catalog_2007_08/' + templates[i][0].astype(str) + '/catalog_2007_08.pkl', 'rb'))
    df_2007_09 = pickle.load(open('catalog_2007_09/' + templates[i][0].astype(str) + '/catalog_2007_09.pkl', 'rb'))
    df_2007_10 = pickle.load(open('catalog_2007_10/' + templates[i][0].astype(str) + '/catalog_2007_10.pkl', 'rb'))
    df_2007_11 = pickle.load(open('catalog_2007_11/' + templates[i][0].astype(str) + '/catalog_2007_11.pkl', 'rb'))
    df_2007_12 = pickle.load(open('catalog_2007_12/' + templates[i][0].astype(str) + '/catalog_2007_12.pkl', 'rb'))
    df_2008_01 = pickle.load(open('catalog_2008_01/' + templates[i][0].astype(str) + '/catalog_2008_01.pkl', 'rb'))
    df_2008_02 = pickle.load(open('catalog_2008_02/' + templates[i][0].astype(str) + '/catalog_2008_02.pkl', 'rb'))
    df_2008_03 = pickle.load(open('catalog_2008_03/' + templates[i][0].astype(str) + '/catalog_2008_03.pkl', 'rb'))
    df_2008_04 = pickle.load(open('catalog_2008_04/' + templates[i][0].astype(str) + '/catalog_2008_04.pkl', 'rb'))
    df_2008_05 = pickle.load(open('catalog_2008_05/' + templates[i][0].astype(str) + '/catalog_2008_05.pkl', 'rb'))
    df_2008_06 = pickle.load(open('catalog_2008_06/' + templates[i][0].astype(str) + '/catalog_2008_06.pkl', 'rb'))
    df_2008_07 = pickle.load(open('catalog_2008_07/' + templates[i][0].astype(str) + '/catalog_2008_07.pkl', 'rb'))
    df_2008_08 = pickle.load(open('catalog_2008_08/' + templates[i][0].astype(str) + '/catalog_2008_08.pkl', 'rb'))
    df_2008_09 = pickle.load(open('catalog_2008_09/' + templates[i][0].astype(str) + '/catalog_2008_09.pkl', 'rb'))
    df_2008_10 = pickle.load(open('catalog_2008_10/' + templates[i][0].astype(str) + '/catalog_2008_10.pkl', 'rb'))
    df_2008_11 = pickle.load(open('catalog_2008_11/' + templates[i][0].astype(str) + '/catalog_2008_11.pkl', 'rb'))
    df_2008_12 = pickle.load(open('catalog_2008_12/' + templates[i][0].astype(str) + '/catalog_2008_12.pkl', 'rb'))
    df_2009_01 = pickle.load(open('catalog_2009_01/' + templates[i][0].astype(str) + '/catalog_2009_01.pkl', 'rb'))
    df_2009_02 = pickle.load(open('catalog_2009_02/' + templates[i][0].astype(str) + '/catalog_2009_02.pkl', 'rb'))
    df_2009_03 = pickle.load(open('catalog_2009_03/' + templates[i][0].astype(str) + '/catalog_2009_03.pkl', 'rb'))
    df_2009_04 = pickle.load(open('catalog_2009_04/' + templates[i][0].astype(str) + '/catalog_2009_04.pkl', 'rb'))
    df_2009_05 = pickle.load(open('catalog_2009_05/' + templates[i][0].astype(str) + '/catalog_2009_05.pkl', 'rb'))
    df_2009_06 = pickle.load(open('catalog_2009_06/' + templates[i][0].astype(str) + '/catalog_2009_06.pkl', 'rb'))

    # Concatenate catalogs
    df_2007_2009 = pd.concat([df_2007_07, df_2007_08, df_2007_09, df_2007_10, df_2007_11, df_2007_12, \
        df_2008_01, df_2008_02, df_2008_03, df_2008_04, df_2008_05, df_2008_06, \
        df_2008_07, df_2008_08, df_2008_09, df_2008_10, df_2008_11, df_2008_12, \
        df_2009_01, df_2009_02, df_2009_03, df_2009_04, df_2009_05, df_2009_06], ignore_index=True)

    # Write catalog into file
    namefile = namedir + '/catalog_2007_2009.pkl'
    pickle.dump(df_2007_2009, open(namefile, 'wb'))
    tfile = open(namedir + '/catalog_2007_2009.txt', 'w')
    tfile.write(df_2007_2009.to_string())
    tfile.close()
