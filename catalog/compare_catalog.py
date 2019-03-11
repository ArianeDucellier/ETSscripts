import pandas as pd
import pickle

filename = '080421.14.048'

namefile = 'LFEs/' + filename + '/catalog.pkl'
df1 = pickle.load(open(namefile, 'rb'))

namefile = '../data/Plourde_2015/detections/' + filename + '_detect5_cull.txt'
df2 = pd.read_csv(namefile, delimiter = ' ', header=None, skiprows=2)
