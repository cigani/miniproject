from neurodynex.neuron_type.typeXY import *
from brian2 import *
import matplotlib.pyplot as plt
import numpy as np


n1 = NeuronX()
n2 = NeuronY()

curr = np.arange(0,1.4,.01)

cls  = n1.get_neuron_type().__name__
print cls
clm = n2.get_neuron_type().__name__
print clm



def BinarySearch(alist, item, sAmp=0.5):
# can optimize this by making it a kinda real binary search
# make it jump a few points ahead and then jump back by only 1 if it hits multiple spikes

    first = 0
    minCur = []
    found = False
    while not found:
        midpoint = first
        icur =alist[midpoint]
        t, v, w, I = item.step(I_amp=icur, I_tstart=100, I_tend=1000., t_end=1000., do_plot=False)
        print 'run %.2f  A ...' % icur
        nspikes = spike(t,v, sAmp)
        print nspikes
        print len(nspikes)
        if len(nspikes) >= 2: #need this to be atleast two because type2s spike at 0.15 non repeat
            minCur=icur
            found = True
        else:
            first = first+1
    return (minCur)

def spike(t,v, spikeAmp=0.5):
    dv = v > spikeAmp
    idx = np.nonzero((dv[:-1]==0) & (dv[1:] ==1)) # see test.py to understand how this works
    return t[idx[0]+1]

if cls == 'NeuronTypeTwo':
    minCur = BinarySearch(curr, n1)
    t, v, w, I = n1.step(I_amp=minCur, I_tstart=100, I_tend=1000, t_end=1000,do_plot=False)
    minCur = BinarySearch(curr, n2)
    t2, v2, w2, I2 = n2.step(I_amp=minCur, I_tstart=100, I_tend=1000, t_end=1000,do_plot=False)
else:
    minCur = BinarySearch(curr, n1)
    t, v, w, I = n1.step(I_amp=minCur, I_tstart=100, I_tend=1000, t_end=1000,do_plot=False)
    minCur = BinarySearch(curr, n2)
    t2, v2, w2, I2 = n2.step(I_amp=minCur, I_tstart=100, I_tend=1000, t_end=1000,do_plot=False)

plt.plot(t2,v2,'b')
plt.plot(t,v,'r')
plt.show()

