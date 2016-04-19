"""
Base HH model for simulations. Import it this module for all other project
parts.

"""


import brian2 as b2
import sys
import matplotlib.pyplot as plt
import numpy as np

def hhNeuron(current, simtime):
    """ hh model via brian2

    In: current: Injected current
    simtime: simulation time (seconds)

    Return:
        StateMonitor: brian2 StateMonitor with fields:
        ['vm', 'i_e', 'm', 'n', 'h']

    """

    # neuron paramaters from project file
    EL = 10.6 * b2.mV
    EK = -12 * b2.mV
    ENa = 115 * b2.mV
    gl = 0.3 * b2.msiemens
    gK = 36 * b2.msiemens
    gNa = 1.5 * 120 * b2.msiemens
    C = 1 * b2.ufarad

    # hh eqs from project file
    eqs = '''
    i_e = current(t) : amp
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
     dvm/dt = membrane_Im/C : volt
     '''

    neuron = b2.NeuronGroup(1,eqs,method='exponential_euler')

    # parameter initial
    # TODO: find correct paramaters for m,h,n
    neuron.vm = 0
    neuron.m = 0
    neuron.h = 0
    neuron.n = 0

    # track values
    stateVals = (b2.StateMonitor(neuron, ['vm', 'i_e', 'm', 'n', 'h'],
    record = True))

    # running simulatin
    b2.run(simtime)

    return stateVals

def hhStep(its=0,ite=150,iamp=0,te=200,sr=1, doPlot=False):

    """ Step current for hh base model.

    Args:
        its (float): start of step [ms]
        ite (float): end of step [ms]
        iamp (float): amplitude of step [uA]
        te (float): end of sim [ms]
        sr (float): sampling rate [ms]
        doPlot (binary): do plot [T/F]

    Return:
        StateMonitor: Brian 2 fields:
        ['vm', 'i_e', 'm', 'n', 'h']
    """

    # step sample rate
    tmp = np.zeros(te)*b2.uamp
    tmp[int(its):int(ite)] = iamp*b2.uamp
    current = b2.TimedArray(tmp,dt=sr*b2.ms)

    stateVals = hhNeuron(current,te*b2.ms)

    # TODO: Add plotting stuff

