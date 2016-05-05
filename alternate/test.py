#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
from hhtest import *
from brian2 import *

stateMonitor = HH_Step(I_tstart=20, I_tend=180, I_amp=20, tend=200, do_plot=True)

