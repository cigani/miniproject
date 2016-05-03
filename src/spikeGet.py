""" Intiate stepHH via hex. extract values as tuples. feed tuple values into
    nspike, plot results if necessary. returns spike frequecy

    optimizes the the engineered channel for maximum adaption

    Warning: Takes a very long time
"""

   #TODO: Turn the entire thing into a single function. The for loops are
   #     all repeats of each other. create a dictionary with appropriate
   #     variables names and linspaces to search for parameters.
   #TODO: Might also be interesting to set it to checks a variable using
   #     rng so that the order of parameters is randomized. might also want
   #     to look into making it run through the optimiatization again after
   #     finding a set of parameters to make sure were good.
   #TODO: Finally sometimes when optimizing it drops us down to a single
   #     spike. neeed to write a algorithim to shuffle indexes a bit when
   #     single/double spikes are detected.

import brian2 as b2
import numpy as np
import hhBasic
import matplotlib.pyplot as plt
import seaborn
def spikeOptimize():
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
    print 'pinf: '
    print hex.pinf
    print 'ninf: '
    print hex.ninf
    print 'minf: '
    print hex.minf
    print 'hinf: '
    print hex.hinf
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
    ved2=np.linspace(1e-7,1,50)
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
    ved2=np.linspace(1e-7,1,50)
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
    ved2=np.linspace(0.001,5,50)
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
    print 'Iteration 6 of 9 done'
    norp2=[]
    nspike2=[]
    vv6=[]
    ved2=np.linspace(0.1,300,100)
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
    vxz = norp2.index(np.max(norp2))
    vv6 = ved2[vxz]
    print 'Iteration 7 of 9 done'
    norp2=[]
    nspike2=[]
    vv7=[]
    ved2=np.linspace(0.01,300,100)
    for i in ved2:
        hex3 = hhBasic.hhStep(itEnd=440, tEnd=440,iAmp=0.5, var2=vv,
               doPlot=False, ntype=2,controlPar1=vv1,
               controlPar2=vv2,controlPar3=vv3,
               controlPar4=vv4,controlPar5=vv5,
               controlPar6=vv6, controlPar7=i)
        t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
        nspike2= hhBasic.spikeRate(t3,v3,doPlot=False)
        if nspike2==0:
            break
        norp2.append(nspike2[1])
    vxz = norp2.index(np.max(norp2))
    vv7 = ved2[vxz]
    print 'Iteration 8 of 9 done'
    norp2=[]
    nspike2=[]
    vv8=[]
    ved2=np.linspace(.05,5,50)
    for i in ved2:
        hex3 = hhBasic.hhStep(itEnd=1000, tEnd=1000,iAmp=0.5, var2=vv,
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
    print 'Iteration 9 of 9 done'

    hex3 = hhBasic.hhStep(itEnd=1000, tEnd=1000,iAmp=0.5, var2=vv,
               doPlot=False, ntype=2,controlPar1=vv1,
               controlPar2=vv2,controlPar3=vv3,
               controlPar4=vv4,controlPar5=vv5,
               controlPar6=vv6, controlPar7=vv7,
               controlPar8=vv8)
    t3,v3 = hhBasic.valTuple(hex3,ntype=2)[0:2]
    nspike2= hhBasic.spikeRate(t3,v3,doPlot=True)
    print("vv: %.10f \n vv1: %.10f \n vv2 :%.10f\n\
    vv3: %.10f\nvv4: %.10f\nvv5: %.10f\nvv6: \
    %.10f\n vv7: %.10f\n vv8: %.10f") % (vv, vv1, \
        vv2, vv3, vv4, vv5, vv6, vv7, vv8)
    print  hex.pinf
    return (vv,vv1,vv2,vv3,vv4,vv5,vv6,vv7,vv8)
""" optimized [vv: 0.2040912245
 vv1: 1.8373673469
  vv2 :0.0204082612
  vv3: 0.0000001000
  vv4: 0.0010000000
  vv5: 1.3266040816
  vv6: 18.2757575758
   vv7: 0.0100000000
    vv8: 1.8683673469
"""
