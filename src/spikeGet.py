""" Intiate stepHH via hex. extract values as tuples. feed tuple values into
    nspike, plot results if necessary. returns spike frequecy
"""

import brian2 as b2
import numpy as np
import hhBasic
import matplotlib.pyplot as plt



hex = hhBasic.hhStep(itEnd=120, tEnd=150, doPlot=False)
t, v, i_e, hinf, ninf, minf, tm, tn, th = hhBasic.valTuple(hex)
nspike = hhBasic.spikeRate(t,v, doPlot=True)

