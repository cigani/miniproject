#!/Users/alorkowski/anaconda2/bin/python

import sys
import matplotlib.pyplot as plt
from neurodynex.hodgkin_huxley.HH import *
from brian2 import *

HH_Step(I_tstart=20, I_tend=180, I_amp=-5, tend=200, do_plot=True)
HH_Step(I_tstart=20, I_tend=180, I_amp=-1, tend=200, do_plot=True)

