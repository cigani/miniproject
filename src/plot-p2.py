#!/Users/alorkowski/anaconda2/bin/python

# WARNING: Final answer may be incorrect by 0.1
print "WARNING: Final answer may be incorrect by 0.1"

import sys
import matplotlib.pyplot as plt
from neurodynex.hodgkin_huxley.HH import *
from brian2 import *
from operator import itemgetter

### Define functions here ###

def binarySearch(alist, item):
   first = 0
   last = len(alist)-1
   found = False
   listCurrent=[]
   listSpike=[]
   minCurrent=[]
   while first<=last and not found:
       midpoint = (first + last)//2
       I = inputCurrents[midpoint]
       print "Testing %.2f A ..." % I
       stateMonitor = HH_Step(I_tstart=20, I_tend=180, I_amp=I, tend=200, do_plot=False)
       nrOfSpikes = compute_spike_count(stateMonitor.vm, 0.01)
       if nrOfSpikes <= 10:
          first = midpoint+1
          listCurrent.append(I)
          listSpike.append(nrOfSpikes)
       else:
          last = midpoint-1
          listCurrent.append(I)
          listSpike.append(nrOfSpikes)
          minCurrent = I
   if minCurrent is not None:
      found = True
   if found == False:
      print "Minimum current to elicit repetitive firing not within the specified input."
      quit()
   return (minCurrent, listCurrent, listSpike)

def compute_spike_count(voltageStateMonitor, spikeAmplitude):
    allValues = voltageStateMonitor / volt # Strip values of its units.
    voltageChange = np.absolute(np.diff(allValues)) # Calculate the differences between successive values
    spikes = voltageChange[np.where( voltageChange >= spikeAmplitude )] # Filter for the voltage change greater than our preset tolerance.
    nrOfSpikes = len(spikes)
    return (nrOfSpikes) #return one values

### End defining functions ###

inputCurrents = np.arange(0.1, 20.0, 0.1)
minCurrent, CurrentArray, nrOfSpikes = binarySearch(inputCurrents, 1)

L = sorted(zip(CurrentArray, nrOfSpikes), key=itemgetter(0))

x, y = zip(*L)

print "Minimum Current to Elicit Repetitive Firing: %.2f" % minCurrent

plt.plot(x, y, '-ro')
plt.show()



