#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
from neurodynex.hodgkin_huxley.HH import *
from brian2 import *

### Define functions here ###

def binarySearch(alist, item):
   first = 0
   last = len(alist)-1
   found = False
   minCurrent=[]
   while first<=last and not found:
       midpoint = (first + last)//2
       I = inputCurrents[midpoint]
       print "Testing %.2f A ..." % I
       stateMonitor = HH_Ramp(I_tstart=30, I_tend=270, I_amp=I, tend=300, dt=0.1, do_plot=False)
       Spike = compute_spike_count(stateMonitor.vm, .06) # We will see if the voltage exceeds a predefined threshold
       if Spike == False:
          first = midpoint+1
       else:
          last = midpoint-1
          minCurrent = I
   if minCurrent is not None:
      found = True
   if found == False:
      print "Minimum current to elicit one spike not within the specified input."
      quit()
   return (minCurrent)

def compute_spike_count(voltageStateMonitor, spikeAmplitude):
    allValues = voltageStateMonitor / volt # Strip values of its units.
    voltageSpike = np.max(allValues) # Calculate the maximum spike amplitude
    if voltageSpike >= spikeAmplitude:
       Spike = True
    else:
       Spike = False
    return (Spike) #return one values

### End defining functions ###

inputCurrents = np.arange(0.1, 20.0, 0.1)
minCurrent = binarySearch(inputCurrents, 1)

print "Minimum Current to Elicit One Spike: %.2f" % minCurrent


