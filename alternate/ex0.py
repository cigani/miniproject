#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
import seaborn
from brian2 import *
from operator import itemgetter
from bmnn.HHmodel import hhNormal as hh
from bmnn.analysis import mhn_inf_plot as gt
from bmnn.analysis import tau_plot as tp
from bmnn.analysis import spikeDetector as sd
from bmnn.analysis import f_i_curve as fi

stateMonitor = hh.HH_Step(I_tstart=20, I_tend=480, I_amp=20, tend=600,
                          do_plot=True, writePlot=True)
#fig, ax = plt.subplots()
#plt.savefig('ex0.eps', format='eps',dpi=1000)
#fig.savefig('ex0.svg', format='svg', dpi=1200)
