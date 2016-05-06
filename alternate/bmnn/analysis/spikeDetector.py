import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
from numpy import array

def spikeGet(t,v,vT=None):
    """  Extract spike time using boolean logic. Seperate array at T/F
         order offset one place, compare and detect.

    Args:
        t: numpy time array
        v: numpy voltage array
        vT = voltage threshold (optional)

    Returns:
        firing rate of neuron
    """
    #simple formula to get a sane vT
    if vT == None:
        vT = [(0.75*(np.max(v))), 10]
        vT = np.max(vT)
    vTF = v > vT

    # use numpy's index-a-index functionality to extract T/F shift point
    # this gives the point where false becomes true.
    idx = np.nonzero((vTF[0][:-1]==0) & (vTF[0][1:]==1))

    # we want the point one time point further though so index +1
    return t[idx[0]+1]


def spikeRate(t,v, vT=None, doPlot=False):
    """ finds spike rate
        if true plots spike times against voltage trace

    Args:
        t: valTuple output
        v: valTuple output
        vT : optional spike time

    Returns:
        spike rate
    """

    sr = spikeGet(t/b2.ms,v/b2.mV,vT=None)

    if doPlot:
        plt.plot(t/b2.ms, v[0]/b2.mV, c='blue', lw=2)
        for s in sr:
            print s
            print np.min(v[0]/b2.mV)
            print np.max(v[0]/b2.mV)
            plt.plot((s,s),
            (np.min(v[0]/b2.mV),np.max(v[0]/b2.mV)),
            c='red'
            )
        plt.ylabel('v [mV]')
        plt.xlabel('t [ms]')
        plt.suptitle('Voltage Trace and Spike points')
        plt.legend(('vm', 'spikes'))
        plt.grid()
        plt.show()

    # no spike or single spike detection
    if len(sr)<2:
        return (0.0,0.0) # a good reason

    # find innerspike interval
    srF =sr[1:]-sr[:-1]
    fstd = np.std(srF)
    # convert from ms to Hz
    f =1000.0/srF.mean()

    return (f,fstd)

