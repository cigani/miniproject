""" Calculate the spike rate against fixed input currents """


import numpy as np
import brian2 as b2
import hhBasic as hh
import matplotlib.pyplot as plt
import spikeGet
import seaborn

def modCount(values, x):
    return {i for i in values if i % x ==0}
vv,vv1,vv2,vv3,vv4,vv5,vv6,vv7,vv8 = spikeGet.spikeOptimize()
curr = np.arange(0.01,4,0.01)
n=0
nspike=[]
sPlts = len(modCount(curr,1))
print sPlts
for i in curr:
    hex = hh.hhStep(itEnd=400, tEnd=400,iAmp=i, var2=vv,
            doPlot=False, ntype=2,
            controlPar1=vv1,controlPar2=vv2,
            controlPar3=vv3,controlPar4=vv4,
            controlPar5=vv5,controlPar6=vv6,
            controlPar7=vv7,controlPar8=vv8)
    t,v = hh.valTuple(hex,ntype=2)[0:2]
    ns = hh.spikeRate(t,v,doPlot=False)[0]
    nspike.append(ns)
    if i%1 == 0:
        print 'true'
        n=n+1
        plt.subplot(sPlts,1,n)
        plt.plot(t, v, label=str(i)+' uA')
        for s in hh.spikeGet(t,v):
             plt.plot([s,s],[np.min(v),np.max(v)],
             c ='red')
        plt.ylabel('mV')
        plt.xlabel('ms')
        plt.legend(loc='upper right')
        plt.yticks(np.arange(min(v), max(v)+1, 50.0))
plt.suptitle('Spike Rate with increasing input current')
plt.show()
print 'length curr: '
print curr
print 'length nspike[0]: '
print nspike
plt.plot(curr, nspike, c='black',lw=2)
plt.ylabel('Spikecount [1/s]')
plt.xlabel('Current [uA]')
plt.suptitle('F-I curve')
plt.axis((0.5,4, min(nspike), max(nspike)))
plt.show()

