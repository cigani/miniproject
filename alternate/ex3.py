#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
from brian2 import *
from operator import itemgetter
from HHmodel import hhAdaptation as hh

stateMonitor = hh.HH_Step(I_tstart=20, I_tend=980, I_amp=20, tend=1000, do_plot=True)


