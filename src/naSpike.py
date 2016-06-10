

import numpy as np
import brian2 as b2
import hhBasic as hh
import matplotlib.pyplot as plt
import spikeGet
import seaborn

def modCount(values,x):
    return {i for i in values if i % x ==0}
curr = np.arange(0,20,1)
n=0
nspike=[]
sPlts = len(modCount(curr,2))

for i in curr:
    hex = hh.hhStep(itEnd=600, tEnd=600,iAmp=i,ntype=3, doPlot=False)
    t,v = hh.valTuple(hex)[0:2]
    ns = hh.spikeRate(t,v,doPlot=False)
    print ns
    nspike.append(ns[0])
    if i%2 ==0:
        n=n+1
        plt.subplot(sPlts,1,n)
        plt.plot(t,v,label=str(i)+ 'uA')
        for s in hh.spikeGet(t,v):
            plt.plot([s,s],[np.min(v),np.max(v)],
            c='red')
        plt.ylabel('mv')
        plt.xlabel('ms')
        plt.legend(loc='upper right')
        plt.yticks(np.arange(min(v),max(v)+1, 50))
plt.suptitle('Spike rate with increasing input current')
plt.show()

plt.plot(curr,nspike,c='black',lw=2)
plt.ylabel('Spikecount [1/s]')
plt.xlabel('Current [uA]')
plt.suptitle('F-I curve')

plt.show()
