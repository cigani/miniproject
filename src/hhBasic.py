import seaborn
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

    plt.plot(valStat.t/b2.ms,valStat.m[0] / nrmfactor / b2.mV,'black',lw=2)
    plt.plot(valStat.t/b2.ms,valStat.n[0] / nrmfactor / b2.mV,'blue',lw=2)
    plt.plot(valStat.t/b2.ms,valStat.h[0] / nrmfactor / b2.mV, 'red', lw=2)
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



def hhNeuronA(curr, simtime, var2,controlPar1,controlPar2,controlPar3,
            controlPar4,controlPar5,controlPar6, controlPar7,
            controlPar8):

    """Simple Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        curr  (TimedArray): Input current injected into the HH neuron
        simtime (float): Simulation time [seconds]

    Returns:
        StateM onitor: Brian2 StateMonitor with valStat fields
        [vm', 'i_e', 'm', 'n','h', 'hinf','minf','ninf', 'tm',
        'th','tn']
    """

    # neuron parameters from project file
    El = -10.6 * b2.mV
    EK = -12 * b2.mV
    ENa = 115 * b2.mV
    EIh = -22 * b2.mV
    gIh = var2 * b2.msiemens
    gl = 0.3 * b2.msiemens
    gK = 5 * b2.msiemens
    gNa = 1.5*120 * b2.msiemens #*1.5
    C = 1 * b2.ufarad
    tauKmax = 10*1000* b2.ms
    cPar1 = controlPar1 * b2.mV
    cPar2 = controlPar2 * b2.mV
    cPar3 = controlPar3 * b2.mV
    cPar4 = controlPar4 * b2.mV
    cPar5 = controlPar5 * b2.mV
    cPar6 = controlPar6 * b2.mV
    cPar7 = controlPar7 * b2.mV
    cPar8 = controlPar8 * b2.mV

    # forming HH model with differential equations
    eqs = '''
    i_e = curr(t) : amp
    membrane_Im = i_e + gNa*m**3*h*(ENa-vm) + \
        gl*(El-vm) + gK*n**4*(EK-vm) + gIh*p*(EIh-vm) : amp
    alphah = .07*exp(-.05*vm/mV)/ms    : Hz
    alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
    alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
    betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
    betam = 4*exp(-.0556*vm/mV)/ms : Hz
    betan = .125*exp(-.0125*vm/mV)/ms : Hz
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dp/dt = (pinf-p)/tp : 1
    dvm/dt = membrane_Im/C : volt
    hinf = alphah/(alphah+betah) : 1
    minf = alpham/(alpham+betam) : 1
    ninf = alphan/(alphan+betan) : 1
    pinf = cPar8/mV*1/(1+cPar5/mV*exp(-cPar1/mV*(cPar4/mV*vm/mV+35/10))) :1
    th = 1/(alphah+betah) : second
    tm = 1/(alpham+betam) : second
    tn = 1/(alphan+betan) : second
    tp = tauKmax/(3.3*exp(cPar2/mV*(vm/mV+35/20*cPar6/mV) + \
        exp(cPar3/mV*(-vm/mV+35/20*cPar7/mV)))) : second
    NaI = gNa*m**3*h*(ENa-vm) : amp
    KI = gK*n**4*(EK-vm) : amp
    KIslow = gIh*p*(EIh-vm) : amp
    '''


    neuron = b2.NeuronGroup(1, eqs, method='exponential_euler')

    # parameter initialization

    neuron.vm = 0
    neuron.m = 0.0529324852572
    neuron.h = 0.596120753508
    neuron.n = 0.317676914061

    # tracking parameters
    valStat = b2.StateMonitor(neuron,
                              ['vm', 'i_e', 'm', 'n',
                               'h', 'minf','ninf','hinf','pinf',
                               'tm','tn','th','NaI','KI','KIslow',
                               'tp'], record=True)

    # running the simulation
    b2.run(simtime)
    return (valStat)

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
    sod = gNa*m**3*h*(ENa-vm) : amp
    pot = gK*n**4*(EK-vm) : amp
    '''

    neuron = b2.NeuronGroup(1, eqs, method='exponential_euler')
    neuron.vm = 0
    neuron.m = 0.0529324852572
    neuron.h = 0.596120753508
    neuron.n = 0.317676914061

    # tracking parameters
    valStat = b2.StateMonitor(neuron, ['vm', 'i_e', 'm', 'n',
    'h', 'hinf','minf','ninf', 'tm','th','tn','sod', 'pot'], record=True)

    # running the simulation
    b2.run(simtime)
    return (valStat)


def hhNeuron3(curr, simtime):

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
    EKs = -12 * b2.mV
    gl = 0.3 * b2.msiemens
    gK = 36 * b2.msiemens
    gNa = 1.5*120 * b2.msiemens #*1.5
    gKs = .36 * b2.msiemens
    C = 1 * b2.ufarad


    # forming HH model with differential equations
    eqs = '''
    i_e = curr(t) : amp
    membrane_Im = i_e + gNa*m**3*h*(ENa-vm) + \
        gl*(El-vm) + gKs*p**2*h*(EKs-vm) + gK*n**4*(EK-vm) : amp
    alphah = .07*exp(-.05*vm/mV)/ms    : Hz
    alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
    alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
    alphap = .03*.01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
    betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
    betam = 4*exp(-.0556*vm/mV)/ms : Hz
    betan = .125*exp(-.0125*vm/mV)/ms : Hz
    betap = .15*.125*exp(-.0125*vm/mV)/ms : Hz
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dp/dt = alphap*(1-p)-betap*p : 1
    dvm/dt = membrane_Im/C : volt
    NaI = gNa*m**3*h*(ENa-vm) : amp
    KI = gK*n**4*(EK-vm) : amp
    KIslow = gKs*p*(EKs-vm) : amp
    hinf = alphah/(alphah+betah) : 1
    minf = alpham/(alpham+betam) : 1
    ninf = alphan/(alphan+betan) : 1
    pinf = alphap/(alphap+betap) : 1
    th = 1/(alphah+betah) : second
    tm = 1/(alpham+betam) : second
    tn = 1/(alphan+betan) : second
    tp = 1/(alphap+betap) : second
    '''
    #set betap coef to .15 for bursting behavior
    neuron = b2.NeuronGroup(1, eqs, method='exponential_euler')
    neuron.vm = 0
    neuron.m = 0.0529324852572
    neuron.h = 0.596120753508
    neuron.n = 0.317676914061
    neuron.p = 0.317
    # tracking parameters
    valStat = b2.StateMonitor(neuron, ['vm', 'i_e', 'm','p', 'n',
    'h','NaI','KI','KIslow', 'th','tp','tn','tm','pinf',
                            'minf','ninf','hinf'],record=True)

    # running the simulation
    b2.run(simtime)
    return (valStat)



def hhStep(itStart=0, itEnd=180, iAmp=7, tEnd=200,dt = 1, doPlot=True,
            ntype = 1,var2=.2,controlPar1=1,controlPar2=1,
            controlPar3=1,controlPar4=1,controlPar5=1,
            controlPar6=1, controlPar7=1, controlPar8=1):

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
    if ntype==1:
        valStat = hhNeuron(curr, tEnd * b2.ms)
    if ntype==3:
        valStat = hhNeuron3(curr,tEnd * b2.ms)
    else:
        valStat = hhNeuronA(curr,tEnd* b2.ms,var2,controlPar1,controlPar2,
                  controlPar3,controlPar4,controlPar5,controlPar6,
                  controlPar7, controlPar8)
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

def valTuple(valStat, ntype=1):
    """ Extract our data from numpy arrays into tuple

    Args:
        valStat from StateMonitor

    Returns:
        [t, vm, i_e, hinf, ninf, minf, tm, tn, th] unit corrected
        normalizes acti/deacti parameters
    """
    if ntype ==1:
        fulltrace = np.append(valStat.minf[0], [valStat.ninf[0],
                valStat.hinf[0]])
        nrmfactor = np.max(fulltrace)/b2.mV
        t = valStat.t / b2.ms
        v = valStat.vm[0] / b2.mV
        i_e = valStat.i_e[0] /b2.pA
        hinf = valStat.hinf[0]/nrmfactor / b2.mV
        ninf = valStat.ninf[0]/nrmfactor / b2.mV
        minf = valStat.minf[0]/nrmfactor / b2.mV
        tm = valStat.tm[0] / b2.ms
        tn = valStat.tn[0] / b2.ms
        th = valStat.th[0] / b2.ms
        return (t,v,i_e,hinf,ninf,minf,tm,tn,th)
    else:
        fulltrace = np.append(valStat.minf[0], [valStat.ninf[0],
                valStat.hinf[0]])
        nrmfactor = np.max(fulltrace)/b2.mV
        t = valStat.t / b2.ms
        v = valStat.vm[0] / b2.mV
        i_e = valStat.i_e[0] /b2.pA
        hinf = valStat.hinf[0]/nrmfactor / b2.mV
        ninf = valStat.ninf[0]/nrmfactor / b2.mV
        minf = valStat.minf[0]/nrmfactor / b2.mV
        pinf = valStat.pinf[0]/nrmfactor / b2.mV
        tm = valStat.tm[0] / b2.ms
        tn = valStat.tn[0] / b2.ms
        th = valStat.th[0] / b2.ms
        #tp = valStat.tp[0] / b2.ms
        ftrce = np.append(valStat.m[0], [valStat.n[0], valStat.h[0],
                          valStat.pinf[0]])
        nrmfac = np.max(ftrce)/b2.mV
        h = valStat.h[0]/nrmfac / b2.mV
        n = valStat.n[0]/nrmfac / b2.mV
        m = valStat.m[0]/nrmfac / b2.mV
        #p = valStat.p[0]/nrmfac / b2.mV
        return (t,v,i_e,hinf,ninf,minf,pinf,tm,tn,th,h,n,m)

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
    #simple formula to get a sane vT
    if vT == None:
        vT = [(0.75*(np.max(v)-np.sqrt(np.std(v)))), 10]
        vT = np.max(vT)
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
        return (0.0,0.0) # a good reason

    # find innerspike interval
    srF =sr[1:]-sr[:-1]
    fstd = np.std(srF)
    # convert from ms to Hz
    f =1000.0/srF.mean()

    return (f,fstd)

