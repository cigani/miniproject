#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
from brian2 import *
from operator import itemgetter
from bmnn.HHmodel import hhAdaptation as hh
from bmnn.HHmodel import hhNormal as hhN
from bmnn.analysis import mhn_plot as gt
from bmnn.analysis import mhn_inf_plot as gti
from bmnn.analysis import tau_plot as tp
from bmnn.analysis import spikeDetector as sd
from bmnn.analysis import f_i_curve as fi

#Call the normal Hodgkin-Huxley Model and store values to the stateMonitor
stateMonitor = hh.HH_Step(I_tstart=35, I_tend=235, I_amp=20, tend=300,
                          do_plot=False)
stateMonitor1 = hhN.HH_Step(I_tstart=35, I_tend=235, I_amp=20, tend=300,
                            do_plot=False)

# Plot data relevant to the spike times of the simulation


fig = plt.figure(figsize=(10,7))
fig.add_subplot(2,1,1)
sd.spikeRate(stateMonitor1.t,stateMonitor1.vm, vT=None, doPlot=True)[2]
plt.text(250, 60, r'No Adaption')
fig.add_subplot(2,1,2)
sd.spikeRate(stateMonitor.t, stateMonitor.vm, vT=None, doPlot=True)[2]
plt.text(250,60, r'Adaption')
title = "Adaption and No Adaption Voltrage Traces"
plt.suptitle('{}'.format(title))
plt.savefig('{}.eps'.format(''.join(title.split())))
## I might have broken the spikeRate to make this work.
## It returns a 3 tuple now.I messed with switchPlot really hard as well

plt.show()
