"""
Hodgkin-Huxley (HH) model with Spike-Frequency Adaptaion (SFA)

Adaptation induced by an additional M-type current, i.e.  a slow voltage-dependent potassium current

Specify the type of input current to the system with the following:
Step Current       | HH_Step()
Sinusoidal Current | HH_Sinus()
Ramp Current       | HH_Ramp()
"""

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

def plot_data(rec, title=None):
    """Plots a TimedArray for values I and v

    Args:
        rec (TimedArray): the data to plot
        title (string, optional): plot title to display
    """

    plt.subplot(311)
    plt.plot(rec.t/b2.ms, rec.vm[0]/b2.mV, lw=2)

    plt.xlabel('t [ms]')
    plt.ylabel('v [mV]')
    plt.grid()

    # find max of activation and inactivation variables
    traceall = np.append(rec.m[0], [rec.n[0], rec.h[0]])
    nrmfactor = np.max(traceall)/b2.mV

    plt.subplot(312)

    plt.plot(rec.t/b2.ms, rec.m[0] / nrmfactor / b2.mV, 'black', lw=2)
    plt.plot(rec.t/b2.ms, rec.n[0] / nrmfactor / b2.mV, 'blue', lw=2)
    plt.plot(rec.t/b2.ms, rec.h[0] / nrmfactor / b2.mV, 'red', lw=2)
    plt.plot(rec.t/b2.ms, rec.w[0] / nrmfactor / b2.mV, 'green', lw=2)
    plt.xlabel('t (ms)')
    plt.ylabel('act./inact.')
    plt.legend(('m', 'n', 'h', 'w'))

    plt.grid()

    plt.subplot(313)
    plt.plot(rec.t/b2.ms, rec.I_e[0]/b2.uamp, lw=2)
    plt.axis((
        0,
        np.max(rec.t/b2.ms),
        min(rec.I_e[0]/b2.uamp)*1.1,
        max(rec.I_e[0]/b2.uamp)*1.1
    ))

    plt.xlabel('t (ms)')
    plt.ylabel('I (micro A)')

    if title is not None:
        plt.suptitle(title)

    plt.show()


def HH_Neuron(curr, simtime):

    """Hodgkin-Huxley neuron with Spike-Frequency Adaptation implemented in Brian2.

    Args:
        curr (TimedArray): Input current injected into the HH neuron
        simtime (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ['vm', 'I_e', 'm', 'n', 'h', 'tm', 'tn', 'th', 'tw', 'NaI', 'KI', 'MI']
    """

    # neuron parameters
    El = -67 * b2.mV
    EK = -100 * b2.mV
    ENa = 50 * b2.mV
    gl = 0.2 * b2.msiemens
    gK = 80 * b2.msiemens
    gNa = 100 * b2.msiemens
    C = 1 * b2.ufarad
    EM = -100 * b2.mV
    gM = 5.0 * b2.msiemens
    EA = 120 * b2.mV
    gA = 5.0 * b2.msiemens

    
    #membrane_Im = I_e + gNa*m**3*h*(ENa-vm) + gl*(El-vm) + gK*n**4*(EK-vm) + gM*w*(EM-vm) + gA*ca*(EK-vm)/(ca+1) - ica: amp
    # forming HH model with SFA with differential equations
    eqs = '''
    I_e = curr(t) : amp
    membrane_Im = I_e + gNa*m**3*h*(ENa-vm) + gl*(El-vm) + gK*n**4*(EK-vm) + gM*w*(EM-vm): amp
    alphah = .128*exp(-(50+vm/mV)/18)/ms    : Hz
    alpham = .32*(54*mV+vm)/(1-exp(-(vm/mV+54)/4))/mV/ms : Hz
    alphan = .032*(52*mV+vm)/(1-exp(-1*(vm/mV+52)/5))/mV/ms : Hz
    betah = 4./(1+exp(-(vm/mV+27)/5))/ms : Hz
    betam = 0.28*(vm/mV+27)/(exp((vm/mV+27)/5)-1)/ms : Hz
    betan = .5*exp(-(57+vm/mV)/40)/ms : Hz
    hi = alphah/(alphah+betah) : 1
    mi = alpham/(alpham+betam) : 1
    ni = alphan/(alphan+betan) : 1 
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dvm/dt = membrane_Im/C : volt
    wi = 1/(1+exp(-(35+vm/mV)/10)) : 1
    tauw = 100*ms : second
    tw = 100/(3.3*exp((vm/mV+35)/20)+exp(-(vm/mV+35)/20)) : second
    dw/dt = (wi-w)/tw : 1
    th = 1/(alphah+betah) : second
    tm = 1/(alpham+betam) : second
    tn = 1/(alphan+betan) : second
    NaI = gNa*m**3*h*(ENa-vm) : amp
    KI = gK*n**4*(EK-vm) : amp
    MI = gM*w*(EM-vm) : amp
    mica = 1/(1+exp(-(vm/mV+25)/2.5)) : 1
    ica = gA*mica*(vm-EA) : amp
    dca/dt = (-0.002*ica/amp - ca/80)/ms : 1
    '''

    neuron = b2.NeuronGroup(1, eqs, method='exponential_euler')

    # parameter initialization
    neuron.vm = 0
    neuron.m = 0.0529324852572
    neuron.h = 0.596120753508
    neuron.n = 0.317676914061

    # tracking parameters
    rec = b2.StateMonitor(neuron, ['vm', 'I_e', 'm', 'n', 'h', 'w', 'mi', 'ni', 'hi', 'wi', 'tm', 'tn', 'th', 'tw', 'NaI', 'KI', 'MI'], record=True)

    # running the simulation
    b2.run(simtime)

    return rec


def HH_Step(I_tstart=20, I_tend=180, I_amp=7,
            tend=200, do_plot=True):

    """
    Run the HH model with SFA for a step current input.

    Args:
        I_tstart (float, optional): start of current step [ms]
        I_tend (float, optional): start of end step [ms]
        I_amp (float, optional): amplitude of current step [uA]
        tend (float, optional): the simulation time of the model [ms]
        do_plot (bool, optional): plot the resulting simulation

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ['vm', 'I_e', 'm', 'n', 'h']
    """

    # 1ms sampled step current
    tmp = np.zeros(tend) * b2.uamp
    tmp[int(I_tstart):int(I_tend)] = I_amp * b2.uamp
    curr = b2.TimedArray(tmp, dt=1.*b2.ms)

    rec = HH_Neuron(curr, tend * b2.ms)

    if do_plot:
        plot_data(
            rec,
            title="Step current",
        )

    return rec


def HH_Sinus(I_freq=0.01, I_offset=0.5, I_amp=7.,
             tend=600, dt=.1, do_plot=True):
    """
    Run the HH model with SFA for a sinusoidal current

    Args:
        tend (float, optional): the simulation time of the model [ms]
        I_freq (float, optional): frequency of current sinusoidal [kHz]
        I_offset (float, optional): DC offset of current [nA]
        I_amp (float, optional): amplitude of sinusoidal [nA]
        do_plot (bool, optional): plot the resulting simulation

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
        voltage (V) recorded
    """

    # dt sampled sinusoidal function
    t = np.arange(0, tend, dt)
    tmp = (I_amp*np.sin(2.0*np.pi*I_freq*t)+I_offset) * b2.uamp
    curr = b2.TimedArray(tmp, dt=dt*b2.ms)

    rec = HH_Neuron(curr, tend * b2.ms)

    if do_plot:
        plot_data(
            rec,
            title="Sinusoidal current",
        )

    return rec


def HH_Ramp(I_tstart=30, I_tend=270, I_amp=20.,
            tend=300, dt=.1, do_plot=True):
    """
    Run the HH model with SFA for a sinusoidal current

    Args:
        tend (float, optional): the simulation time of the model [ms]
        I_tstart (float, optional): start of current ramp [ms]
        I_tend (float, optional): end of the current ramp [ms]
        I_amp (float, optional): final amplitude of current ramp [uA]
        do_plot (bool, optional): plot the resulting simulation

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
        voltage (V) recorded
    """

    # dt sampled sinusoidal function
    t = np.arange(0, tend, dt)
    tmp = np.zeros_like(t)
    index_start = np.searchsorted(t, I_tstart)
    index_end = np.searchsorted(t, I_tend)
    tmp[index_start:index_end] = np.arange(0, index_end-index_start, 1.0) \
        / (index_end-index_start) * I_amp
    curr = b2.TimedArray(tmp * b2.uamp, dt=dt*b2.ms)

    rec = HH_Neuron(curr, tend * b2.ms)

    if do_plot:
        plot_data(
            rec,
            title="Sinusoidal current",
        )

    return rec
