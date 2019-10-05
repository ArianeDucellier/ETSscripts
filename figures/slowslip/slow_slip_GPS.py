import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

station = 'PGC5'
direction = 'lon'
dataset = 'raw'
filename = '../../data/PANGA/' + dataset + '/' + station + '.' + direction

data = np.loadtxt(filename, skiprows=26)
time = data[:, 0]
disp = data[:, 1]

time_ETS = [2000.9583, 2002.1250, 2003.1250, 2004.0417, 2004.5417, 2005.7083, 2007.0833, 2008.375, 2009.375, \
            2010.6667, 2011.6667, 2012.7083, 2013.7500, 2014.9167, 2016.0000, 2017.1667]

params = {'xtick.labelsize':16,
          'ytick.labelsize':16}
pylab.rcParams.update(params)

plt.figure(1, figsize=(10, 8))
for i in range(0, len(time_ETS)):
    plt.axvline(time_ETS[i], linewidth=2, color='grey', linestyle='--')
plt.plot(time, disp, 'ro', markersize=1)
plt.xlim(np.min(time), np.max(time))
plt.xlabel('Time (year)', fontsize=20)
plt.ylabel('Displacement (mm)', fontsize=20)
plt.title('GPS station ' + station + ' (' + direction + ' - ' + dataset + ')', fontsize=24)
plt.savefig('slow_slip_GPS.eps', format='eps')
