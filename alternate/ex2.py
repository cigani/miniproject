#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
from brian2 import *
from operator import itemgetter
from bmnn.HHmodel import hhAdaptation as hh
from bmnn.analysis import mhn_plot as gt
from bmnn.analysis import mhn_inf_plot as gti
from bmnn.analysis import tau_plot as tp
from bmnn.analysis import spikeDetector as sd
from bmnn.analysis import f_i_curve as fi

# Call the normal Hodgkin-Huxley Model and store values to the stateMonitor
stateMonitor = hh.HH_Step(I_tstart=20, I_tend=480, I_amp=20, tend=600, do_plot=False)

# Plot data relevant to the gating variables
gti.gatevar_A(stateMonitor)
# gt.gatevar(stateMonitor)

# Plot data relevant to the time constants
tp.tau_plot_A(stateMonitor)

# Plot data relevant to the spike times of the simulation
sd.spikeRate(stateMonitor.t,stateMonitor.vm, vT=None, doPlot=True)

# Plot data relevant to the F-I curve
fi.f_i_curve_A(20, 0.5)

