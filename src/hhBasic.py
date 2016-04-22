import sys
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

def plotData(valStat, title=None):
    """Plots a TimedArray for values I and v

    Args:
        valStat (TimedArray): the data to plot
        title (string, optional): plot title to display
    """

    plt.subplot(311)
    plt.plot(valStat.t/b2.ms, valStat.vm[0]/b2.mV, lw=2)

    plt.xlabel('t [ms]')
    plt.ylabel('v [mV]')
    plt.grid()

    # find max of activation and inactivation variables
    traceall = np.append(valStat.m[0], [valStat.n[0], valStat.h[0]])
    nrmfactor = np.max(traceall)/b2.mV

    plt.subplot(312)

    plt.plot(valStat.t/b2.ms, valStat.m[0] / nrmfactor / b2.mV,'black',lw=2)
    plt.plot(valStat.t/b2.ms, valStat.n[0] / nrmfactor / b2.mV,'blue',lw=2)
    plt.plot(valStat.t/b2.ms, valStat.h[0] / nrmfactor / b2.mV, 'red', lw=2)
    plt.xlabel('t (ms)')
    plt.ylabel('act./inact.')
    plt.legend(('m', 'n', 'h'))

    plt.grid()

    plt.subplot(313)
    plt.plot(valStat.t/b2.ms, valStat.i_e[0]/b2.uamp, lw=2)
    plt.axis((
        0,
        np.max(valStat.t/b2.ms),
        min(valStat.i_e[0]/b2.uamp)*1.1,
        max(valStat.i_e[0]/b2.uamp)*1.1
    ))

    plt.xlabel('t (ms)')
    plt.ylabel('I (micro A)')

    if title is not None:
        plt.suptitle(title)

    plt.show()


def hhNeuron(curr, simtime):

    """Simple Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        curr (TimedArray): Input current injected into the HH neuron
        simtime (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with valStat fields
        [vm', 'i_e', 'm', 'n','h', 'hinf','minf','ninf', 'tm',
        'th','tn']
    """

    # neuron parameters from project file
    El = 10.6 * b2.mV
    EK = -12 * b2.mV
    ENa = 115 * b2.mV
    gl = 0.3 * b2.msiemens
    gK = 36 * b2.msiemens
    gNa = 1.5*120 * b2.msiemens #*1.5
    C = 1 * b2.ufarad

    # forming HH model with differential equations
    # TODO: Figure out if Na/K currents are just gNa/k * volts
    eqs = '''
    i_e = curr(t) : amp
    membrane_Im = i_e + gNa*m**3*h*(ENa-vm) + \
        gl*(El-vm) + gK*n**4*(EK-vm) : amp
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
    gNa0 : siemens
    gK0 : siemens
    hinf = alphah/(alphah+betah) : 1
    minf = alpham/(alpham+betam) : 1
    ninf = alphan/(alphan+betan) : 1
    th = 1/(alphah+betah) : second
    tm = 1/(alpham+betam) : second
    tn = 1/(alphan+betan) : second
    '''

    neuron = b2.NeuronGroup(1, eqs, method='exponential_euler')

    # parameter initialization #TODO: Find paramter logic
        # Maybe apply stochastic methods from brian docs:
        # 'we can do this by using the symbol xi in differential equations'
    neuron.vm = 0
    neuron.m = 0.0529324852572
    neuron.h = 0.596120753508
    neuron.n = 0.317676914061

    # tracking parameters
    # TODO: add Na / K currents
    valStat = b2.StateMonitor(neuron, ['vm', 'i_e', 'm', 'n',
    'h', 'hinf','minf','ninf', 'tm','th','tn'], record=True)

    # running the simulation
    b2.run(simtime)
    return (valStat)


def hhStep(itStart=20, itEnd=180, iAmp=7,
            tEnd=200,dt = 1, doPlot=True):

    """Run the Hodgkin-Huley neuron for a step current input.

    Args:
        itStart (float, optional): start of current step [ms]
        itEnd (float, optional): start of end step [ms]
        iAmp (float, optional): amplitude of current step [uA]
        tEnd (float, optional): the simulation time of the model [ms]
        doPlot (bool, optional): plot the resulting simulation

    Returns:
        StateMonitor: Brian2 StateMonitor with valStatorded fields
        ['vm', 'i_e', 'm', 'n', 'h']
    """

    # dt sampled simulation
    tmp = np.zeros(tEnd) * b2.uamp
    tmp[int(itStart):int(itEnd)] = iAmp * b2.uamp
    curr = b2.TimedArray(tmp, dt=dt*b2.ms)

    valStat = hhNeuron(curr, tEnd * b2.ms)

    if doPlot:
        plotData(
            valStat,
            title="Step current",
        )

    return valStat


def hhSinus(iFreq=0.01, iOffset=0.5, iAmp=7.,
             tEnd=600, dt=.1, doPlot=True):
    """
    Run the HH model for a sinusoidal current

    Args:
        tEnd (float, optional): the simulation time of the model [ms]
        iFreq (float, optional): frequency of current sinusoidal [kHz]
        iOffset (float, optional): DC offset of current [nA]
        iAmp (float, optional): amplitude of sinusoidal [nA]
        doPlot (bool, optional): plot the resulting simulation

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
        voltage (V) valStatorded
    """

    # dt sampled sinusoidal function
    t = np.arange(0, tEnd, dt)
    tmp = (iAmp*np.sin(2.0*np.pi*iFreq*t)+iOffset) * b2.uamp
    curr = b2.TimedArray(tmp, dt=dt*b2.ms)

    valStat = hhNeuron(curr, tEnd * b2.ms)

    if doPlot:
        plotData(
            valStat,
            title="Sinusoidal current",
        )

    return valStat


def hhRamp(itStart=30, itEnd=270, iAmp=20.,
            tEnd=300, dt=.1, doPlot=True):
    """
    Run the HH model for a sinusoidal current

    Args:
        tEnd (float, optional): the simulation time of the model [ms]
        itStart (float, optional): start of current ramp [ms]
        itEnd (float, optional): end of the current ramp [ms]
        iAmp (float, optional): final amplitude of current ramp [uA]
        doPlot (bool, optional): plot the resulting simulation

    Returns:
        StateMonitor: Brian2 StateMonitor with input current (I) and
        voltage (V) valStatorded
    """

    # dt sampled sinusoidal function
    t = np.arange(0, tEnd, dt)
    tmp = np.zeros_like(t)
    index_start = np.searchsorted(t, itStart)
    index_end = np.searchsorted(t, itEnd)
    tmp[index_start:index_end] = np.arange(0, index_end-index_start, 1.0) \
        / (index_end-index_start) * iAmp
    curr = b2.TimedArray(tmp * b2.uamp, dt=dt*b2.ms)

    valStat = hhNeuron(curr, tEnd * b2.ms)

    if doPlot:
        plotData(
            valStat,
            title="Sinusoidal current",
        )

    return valStat

def valTuple(valStat):
    """ Extract our data from numpy arrays into tuple

    Args:
        valStat from StateMonitor

    Returns:
        [t, vm, i_e, hinf, ninf, minf, tm, tn, th] unit corrected
        normalizes acti/deacti parameters
    """
    fulltrace = np.append(valStat.minf[0], [valStat.ninf[0],
                valStat.hinf[0]])
    nrmfactor = np.max(fulltrace)/b2.mV
    t = valStat.t / b2.ms
    v = valStat.vm[0] / b2.mV
    i_e = valStat.i_e[0] /b2.pA # i is reserved for index, I breaks PEP
    hinf = valStat.hinf[0]/nrmfactor / b2.mV
    ninf = valStat.ninf[0]/nrmfactor / b2.mV
    minf = valStat.minf[0]/nrmfactor / b2.mV
    tm = valStat.tm[0] / b2.ms
    tn = valStat.tn[0] / b2.ms
    th = valStat.th[0] / b2.ms

    return (t,v,i_e,hinf,ninf,minf,tm,tn,th)

def spikeGet(t,v,vT=None):
    """  Extract spike time using boolean logic. Seperate array at T/F
         order offset one place, compare and detect.

    Args:
        t: numpy time array
        v: numpy voltage array
        vT = voltage threshold (optional)

    Returns:
        firing rate of neuron
    """

    if vT == None:
        vT = (0.85*(np.max(v)-np.sqrt(np.std(v))))
    vTF = v>vT
    # use numpy's index-a-index functionality to extract T/F shift point
    # this gives the point where false becomes true.
    idx = np.nonzero((vTF[:-1]==0) & (vTF[1:]==1))

    # we want the point one time point  further though so index +1
    return t[idx[0]+1]

def spikeRate(t,v, vT=None, doPlot=False):
    """ finds spike rate
        if true plots spike times against voltage trace

    Args:
        t: valTuple output
        v: valTuple output
        vT : optional spike time

    Returns:
        spike rate
    """

    sr = spikeGet(t,v,vT)

    if doPlot:
        plt.plot(t, v, c='blue', lw=2)
        for s in sr:
            plt.plot([s,s],
            [np.min(v),np.max(v)],
            c='red'
            )
        plt.ylabel('v [mV]')
        plt.xlabel('t [ms]')
        plt.suptitle('Voltage Trace and Spike points')
        plt.legend(('vm', 'spikes'))
        plt.grid()
        plt.show()

    # no spike or single spike detection
    if len(sr)<2:
        return 0.0

    # find innerspike interval
    srF =sr[1:]-sr[:-1]

    # convert from ms to Hz
    f =1000.0/srF.mean()

    return f

