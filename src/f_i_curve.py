""" Calculate the spike rate against fixed input currents """

import numpy as np
import brian2 as b2
import hhBasic as hh
import matplotlib.pyplot as plt


curr = np.linspace(0.0,30.0,15)
nspike = []

for i in curr:
    hex = hh.hhStep(itEnd=120, tEnd=150,iAmp=i, doPlot=False)
    t,v,i_e,hinf,ninf,minf,tm,tn,th = hh.valTuple(hex)
    ns = hh.spikeRate(t,v,doPlot=False)
    nspike.append(ns)

plt.plot(curr, nspike, c='black',lw=2)
plt.ylabel('Spikecount [1/s]')
plt.xlabel('Current [uA]')
plt.suptitle('F-I curve')
plt.show()




