""" Calculate the spike rate against fixed input currents """

import numpy as np
import brian2 as b2
import hhBasic as hh
import matplotlib.pyplot as plt



def modCount(values, x):
    return {i for i in values if i % x ==0}

curr = np.arange(0,13,2)
n=0
nspike=[]
sPlts = len(modCount(curr,2))

for i in curr:
    hex = hh.hhStep(itEnd=120, tEnd=150,iAmp=i, doPlot=False)
    t,v,i_e,hinf,ninf,minf,tm,tn,th = hh.valTuple(hex)
    ns = hh.spikeRate(t,v,doPlot=False)
    nspike.append(ns)
    if i%2 == 0:
        n=n+1
        plt.subplot(sPlts,1,n)
        plt.plot(hex.t/b2.ms, hex.vm[0]/b2.mV, label=str(i))
        for s in hh.spikeGet(t,v):
             plt.plot([s,s],[np.min(v),np.max(v)],
             c ='red'
             )
        plt.ylabel('mV')
        plt.xlabel('ms')
        plt.legend(loc='upper right')
        plt.xticks(np.arange(min(v), max(v)+1, 10.0))
plt.show()

plt.plot(curr, nspike, c='black',lw=2)
plt.ylabel('Spikecount [1/s]')
plt.xlabel('Current [uA]')
plt.suptitle('F-I curve')
plt.show()
