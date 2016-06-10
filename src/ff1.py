

import sys
import matplotlib.pyplot as plt
from neurodynex.hodgkin_huxley.HH import *
from brian2 import *
from operator import itemgetter

def BinarySearch(alist, item):
    first = 0
    last = len(alist)-1
    found = False
    minCur = []
    listCurrent = []
    listSpike = []
    while first<= last and not found:
        midpoint = (first+last)//2
        I = inCur[midpoint]
        print 'Test %.2f A ...'  % I
        print 'point %.10f f ' % midpoint
        sMon = HH_Step(I_amp=I, do_plot =False)
        nSpike = compute_spike_count(sMon.vm, .01)
        if nSpike == 0:
            first = midpoint+1
            listCurrent.append(I)
            listSpike.append(nSpike)
        else:
            last = midpoint -1
            listCurrent.append(I)
            listSpike.append(nSpike)
            minCur=I
    if minCur is not None:
        found = True
    if found == False:
        print "Minimum current not within specificd input."
        quit
    return(minCur, listCurrent, listSpike)

def compute_spike_count(voltageStateMonitor, spikeAmplitude):
    vals = voltageStateMonitor / volt
    vdelta = np.absolute(np.diff(vals))
    spikes = vdelta[np.where( vdelta >= spikeAmplitude )]
    nSpike = len(spikes)
    return (nSpike)

inCur = np.arange(0.1, 20, 0.1)
minCur, CurrentArray, nSpike= BinarySearch(inCur, 1)

L = sorted(zip(CurrentArray, nSpike), key=itemgetter(0))

x, y = zip(*L)

print 'min current to elicit one spike: %.2f' % minCur


