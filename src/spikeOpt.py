import brian2 as b2
import numpy as np
import hhBasic as hh
import matplotlib.pyplot as plt

paraSpace = [
    np.linspace(1e-5,5,3), \
    np.linspace(0.001,5,50), \
    np.linspace(1e-7,1,50), \
    np.linspace(1e-7,1,50), \
    np.linspace(0.001,5,50), \
    np.linspace(1e-4,5,50), \
    np.linspace(.1,100,300), \
    np.linspace(.001,100,100), \
    np.linspace(0.05,5,50)]
paraName = [
    'var2',
    'controlPar1',
    'contorlPar2',
    'controlPar3',
    'contorlPar4',
    'contorlPar5',
    'contorlPar6',
    'contorlPar7',
    'contorlPar8'
    ]
d={}
for x in range(len(paraSpace)):
        d[paraName[x]]=paraSpace[x]
print
print d[paraName[0]]
"""
paraVal = {}
spikeVal=[]
for paraIndex in np.arange(len(paraSpace)):
    for para in paraSpace[paraIndex]:
        paraVar = paraName[paraIndex]
        print paraVar
        neuron = hh.hhStep(itEnd=200, tEnd=200,
        iAmp=0.5, doPlot=False, ntype=2, var2 = para,
        controlPar1 = 1, controlPar2 = 1,
        controlPar3 = 1, controlPar4 = 1,
        controlPar5 = 1, controlPar6 = 1,
        controlPar7 = 1, controlPar8 = 1)
        t, v = hh.valTuple(neuron, ntype=2)[0:2]
        nspike = hh.spikeRate(t,v,doPlot=False)
        if nspike != 0:
            spikeVal.append(nspike[1])
            continue
        else:
            paraLoc = spikeVal.index(np.max(spikeVal))
            paraVal[paraIndex] = para
            break

"""
