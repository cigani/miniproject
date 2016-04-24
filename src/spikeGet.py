""" Intiate stepHH via hex. extract values as tuples. feed tuple values into
    nspike, plot results if necessary. returns spike frequecy

    optimizes the the engineered channel for maximum adaption

    Warning: Takes a very long time
"""

import brian2 as b2
import numpy as np
import hhBasic
import matplotlib.pyplot as plt

norp=[]
ved = np.linspace(1e-5,5,50)
for i in ved:
    hex = hhBasic.hhStep(itEnd=440, tEnd=440,
    iAmp=0.5,var2=i,doPlot=False, ntype=2)
    t,v, = hhBasic.valTuple(hex, ntype=2)[0:2]
    nspike = hhBasic.spikeRate(t,v, doPlot=False)
    if nspike == 0:
        break
    norp.append(nspike[1])
vxz = norp.index(np.max(norp))
vv = ved[vxz]
print 'Iteration 1 of 9 done'
norp1=[]
nspike1=[]
ved1=np.linspace(0.001,5,50)
for i in ved1:
    hex2 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=i)
    t2,v2 = hhBasic.valTuple(hex2,ntype=2)[0:2]
    nspike1= hhBasic.spikeRate(t2,v2,doPlot=False)
    if nspike1 == 0:
        break
    norp1.append(nspike1[1])
vxz = norp1.index(np.max(norp1))
vv1 = ved1[vxz]
print 'Iteration 2 of 9 done'
norp2=[]
nspike2=[]
ved2=np.linspace(0.001,5,50)
for i in ved2:
    hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=vv1,
           controlPar2=i)
    t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
    nspike2= hhBasic.spikeRate(t3,v3,doPlot=False)
    if nspike2 == 0:
        break
    norp2.append(nspike2[1])
vxz = norp2.index(np.max(norp2))
vv2 = ved2[vxz]
print 'Iteration 3 of 9 done'
norp2=[]
nspike2=[]
vv3=[]
ved2=np.linspace(0.0001,5,50)
for i in ved2:
    hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=vv1,
           controlPar2=vv2,controlPar3=i)
    t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
    nspike2= hhBasic.spikeRate(t3,v3,doPlot=False)
    if nspike2==0:
        break
    norp2.append(nspike2[1])
vxz = norp2.index(np.max(norp2))
vv3 = ved2[vxz]
print 'Iteration 4 of 9 done'
norp2=[]
nspike2=[]
vv4=[]
ved2=np.linspace(0.0001,5,50)
for i in ved2:
    hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=vv1,
           controlPar2=vv2,controlPar3=vv3,
           controlPar4=i)
    t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
    nspike2= hhBasic.spikeRate(t3,v3,doPlot=False)
    if nspike2==0:
        break
    norp2.append(nspike2[1])
vxz = norp2.index(np.max(norp2))
vv4 = ved2[vxz]
print 'Iteration 5 of 9 done'
norp2=[]
nspike2=[]
vv5=[]
ved2=np.linspace(0.0001,5,50)
for i in ved2:
    hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=vv1,
           controlPar2=vv2,controlPar3=vv3,
           controlPar4=vv4,controlPar5=i)
    t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
    nspike2= hhBasic.spikeRate(t3,v3,doPlot=False)
    if nspike2==0:
        break
    norp2.append(nspike2[1])
vxz = norp2.index(np.max(norp2))
vv5 = ved2[vxz]
print 'vv5'
print vv5
print 'Iteration 6 of 9 done'
norp2=[]
nspike2=[]
vv6=[]
ved2=np.linspace(0.1,100,55)
for i in ved2:
    hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=vv1,
           controlPar2=vv2,controlPar3=vv3,
           controlPar4=vv4,controlPar5=vv5,
           controlPar6=i)
    t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
    nspike2= hhBasic.spikeRate(t3,v3,doPlot=False)
    if nspike2==0:
        break
    norp2.append(nspike2[1])
print norp2
vxz = norp2.index(np.max(norp2))
vv6 = ved2[vxz]
print 'vv6'
print vv6
print 'Iteration 7 of 9 done'

norp2=[]
nspike2=[]
vv7=[]
ved2=np.linspace(0.01,100,50)
for i in ved2:
    hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=vv1,
           controlPar2=vv2,controlPar3=vv3,
           controlPar4=vv4,controlPar5=vv5,
           controlPar6=vv6, controlPar7=i)
    t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
    nspike2= hhBasic.spikeRate(t3,v3,doPlot=False)
    print 'nspike2'
    print nspike2
    print 'norp2'
    print norp2
    if nspike2==0:
        break
    norp2.append(nspike2[1])
print norp2
vxz = norp2.index(np.max(norp2))
vv7 = ved2[vxz]
print 'vv7'
print vv7
print 'Iteration 8 of 9 done'
norp2=[]
nspike2=[]
vv8=[]
ved2=np.linspace(.05,5,50)
for i in ved2:
    hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=vv1,
           controlPar2=vv2,controlPar3=vv3,
           controlPar4=vv4,controlPar5=vv5,
           controlPar6=vv6, controlPar7=vv7,
           controlPar8=i)
    t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
    nspike2= hhBasic.spikeRate(t3,v3,doPlot=False)
    if nspike2 ==0:
        break
    norp2.append(nspike2[1])
vxz = norp2.index(np.max(norp2))
vv8 = ved2[vxz]
print 'vv8'
print vv8
print 'Iteration 9 of 9 done'

hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
           doPlot=False, ntype=2,controlPar1=vv1,
           controlPar2=vv2,controlPar3=vv3,
           controlPar4=vv4,controlPar5=vv5,
           controlPar6=vv6, controlPar7=vv7,
           controlPar8=vv8)
t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
nspike2= hhBasic.spikeRate(t3,v3,doPlot=True)

#vv =0.2, vv8=2~, vv7=18
