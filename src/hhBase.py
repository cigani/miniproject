"""
Base HH model for simulations. Import it this module for all other project
parts.

"""


import brian2 as b2
import sys
import matplotlib.pyplot as plt
import numpy as np

def hhNeuron(curr, simtime):
    """ hh model via brian2

    In:
    current: Injected current
    simtime: simulation time (seconds)

    Return:
        StateMonitor: brian2 StateMonitor with fields:
        ['vm', 'i_e', 'm', 'n', 'h']

    """

    # neuron paramaters from project file
    El = 10.6 * b2.mV
    EK = -12 * b2.mV
    ENa = 115 * b2.mV
    gl = 0.3 * b2.msiemens
    gK = 36 * b2.msiemens
    gNa = 1.5 * 120 * b2.msiemens # Edited to account for a 1.5 - fold incre    ase in Na channel density
    C = 1 * b2.ufarad







    # hh eqs from project file
    eqs = '''
    i_e = curr(t) : amp
    membane_im = i_e + gNa*m**3*h(ENa-vm) + \
        gl*(El-vm) + gK*n**4(EK-vm) : amp
     alphah = .07*exp(-.05*vm/mV)/ms    : Hz
     alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
     alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
     betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
     betam = 4*exp(-.0556*vm/mV)/ms : Hz
     betan = .125*exp(-.0125*vm/mV)/ms : Hz
     dh/dt = alphah*(1-h)-betah*h : 1
     dm/dt = alpham*(1-m)-betam*m : 1
     dn/dt = alphan*(1-n)-betan*n : 1
     dvm/dt = membrane_im/C : volt
     '''

    neuron = b2.NeuronGroup(1, eqs, method='exponential_euler')

    # parameter initial
    # TODO: find correct paramaters for m,h,n
    neuron.vm = 0
    neuron.m = 0.059
    neuron.h = 0.596
    neuron.n = 0.3176

    # track values
    rec = b2.StateMonitor(neuron, ['vm', 'i_e', 'm', 'n', 'h'], \
    record = True)

    # running simulatin
    b2.run(simtime)

    return rec

def hhStep(I_tstart=20, I_tend=180, I_amp=7,
            tend=200, do_plot=True):

    """ Step current for hh base model.

    Args:
        itStart (float): start of step [ms]
        itEnd(float): end of step [ms]
        iAmp (float): amplitude of step [uA]
        tEnd (float): end of sim [ms]
        sRate (float): sampling rate [ms]
        doPlot (binary): do plot [T/F]

    Return:
        StateMonitor: Brian 2 fields:
        ['vm', 'i_e', 'm', 'n', 'h']
    """

    # step sample rate
    tmp = np.zeros(tend) * b2.uamp
    tmp[int(I_tstart):int(I_tend)] = I_amp * b2.uamp
    curr = b2.TimedArray(tmp, dt=1.*b2.ms)
    rec = hhNeuron(curr, tend * b2.ms)

    #tmp = np.zeros(tEnd) * b2.uamp
    #tmp[int(itStart):int(itEnd)] = iAmp * b2.uamp
    #curr = b2.TimedArray(tmp, dt=1.*b2.ms)

    #tmp = np.zeros(itEnd)*b2.uamp
    #tmp[int(itStart):int(itEnd)] = iAmp*b2.uamp
    #curr = b2.TimedArray(tmp,dt=sRate*b2.ms)

    rec = hhNeuron(curr,tEnd*b2.ms)

    # TODO: Add plotting stuff
    return stateVals


