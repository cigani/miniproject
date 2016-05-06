#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
from hhAdaptation import *
from brian2 import *
from operator import itemgetter

stateMonitor = HH_Step(I_tstart=20, I_tend=480, I_amp=20, tend=500, do_plot=True)


