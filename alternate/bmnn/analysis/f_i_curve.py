""" Calculate the spike rate against fixed input currents """

import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt
import spikeDetector as sd
from ..HHmodel import hhNormal as hh
from ..HHmodel import hhAdaptation as hhA

def modCount(values, x):
    return {i for i in values if i % x ==0}

def f_i_curve(maxI, incr):
    curr = np.arange(0.0,maxI,incr)
    n=0
    nspike=[]
    sPlts = len(modCount(curr,3))
    print sPlts
    for i in curr:
        stateMonitor = hh.HH_Step(I_tstart=20, I_tend=480, I_amp=i, tend=500, do_plot=False)
        t = stateMonitor.t
        v = stateMonitor.vm
        ns = sd.spikeRate(t,v,doPlot=False)[0]
        nspike.append(ns)
        if i%3 == 0:
            print 'true'
            n=n+1
            plt.subplot(sPlts,1,n)
            plt.plot(t/b2.ms, v[0]/b2.mV, label=str(i)+' uA')
            for s in sd.spikeGet(t/b2.ms,v/b2.mV,vT=None):
                plt.plot([s,s],[np.min(v/b2.mV),np.max(v/b2.mV)],
                         c ='red')
            plt.ylabel('mV')
            plt.xlabel('ms')
            plt.legend(loc='upper right')
            plt.yticks(np.arange(min(v[0]/b2.mV), max(v[0]/b2.mV)+1, 50.0))
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
    plt.axis((0.0,12, min(nspike), max(nspike)))
    plt.show()

def f_i_curve_A(maxI, incr):
    curr = np.arange(0.0,maxI,incr)
    n=0
    nspike=[]
    sPlts = len(modCount(curr,3))
    print sPlts
    for i in curr:
        stateMonitor = hhA.HH_Step(I_tstart=20, I_tend=480, I_amp=i, tend=500, do_plot=False)
        t = stateMonitor.t
        v = stateMonitor.vm
        ns = sd.spikeRate(t,v,doPlot=False)[0]
        nspike.append(ns)
        if i%3 == 0:
            print 'true'
            n=n+1
            plt.subplot(sPlts,1,n)
            plt.plot(t/b2.ms, v[0]/b2.mV, label=str(i)+' uA')
            for s in sd.spikeGet(t/b2.ms,v/b2.mV,vT=None):
                plt.plot([s,s],[np.min(v/b2.mV),np.max(v/b2.mV)],
                         c ='red')
            plt.ylabel('mV')
            plt.xlabel('ms')
            plt.legend(loc='upper right')
            plt.yticks(np.arange(min(v[0]/b2.mV), max(v[0]/b2.mV)+1, 50.0))
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
    plt.axis((0.0,12, min(nspike), max(nspike)))
    plt.show()
