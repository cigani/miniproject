#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
from hhAdaptation import *
from brian2 import *
from operator import itemgetter

def compute_spike_event(voltageStateMonitor):
    allValues = voltageStateMonitor / volt
    maxi = max(allValues,key=itemgetter(1))[0]
    print maxi
    index = numpy.where(allValues == maxi)[0]
    return (index)

stateMonitor = HH_Step(I_tstart=20, I_tend=280, I_amp=20, tend=300, do_plot=True)

#firstSpike = compute_spike_event(stateMonitor.vm)

#print max(stateMonitor.vm/volt)
#print firstSpike

