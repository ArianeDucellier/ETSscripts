import numpy as np
import pickle

from math import sqrt
from sklearn import decomposition

def transform(filename, station, channel):

    # Get data
    data = pickle.load(open('data/' + filename + '.pkl', 'rb'))
    stations = data[0]
    for site, index in zip(stations, range(0, len(stations))):
        if site == station:
            if channel == 'East':
                stream = data[1][index]
            elif channel == 'North':
                stream = data[2][index]
            elif channel == 'Vertical':
                stream = data[3][index]
    # Normalize data
    X = np.zeros((len(stream), len(stream[0].data)))
    for i in range(0, len(stream)):
        X[i, :] = (stream[i].data - np.mean(stream[i].data)) / np.std(stream[i].data)
    Xstack = np.mean(X, axis=0)
    Xcor = np.zeros(len(stream))
    for i in range(0, len(stream)):
        Xcor[i] = np.dot(X[i], Xstack) / (sqrt(np.dot(X[i], X[i])) * sqrt(np.dot(Xstack, Xstack)))
    # Sparse coding
    model = decomposition.DictionaryLearning()
    XP = model.fit(X).transform(X)
    XPstack = np.mean(XP, axis=0)
    XPcor = np.zeros(len(stream))
    for i in range(0, len(stream)):
        if np.dot(XP[i], XP[i]) != 0.0:
            XPcor[i] = np.dot(XP[i], XPstack) / (sqrt(np.dot(XP[i], XP[i])) * sqrt(np.dot(XPstack, XPstack)))
        else:
            XPcor[i] = 0.0
    return (Xcor, XPcor)

if __name__ == '__main__':

    # Set the parameters
    filename = '080401.05.050'
    station = 'WDC'
    channel = 'East'

    (Xcor, XPcor) = transform(filename, station, channel)
