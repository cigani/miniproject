def HH_Neuron(curr, simtime):
    # neuron parameters
    El = 10.6 * b2.mV
    EK = -12 * b2.mV
    ENa = 115 * b2.mV
    gl = 0.3 * b2.msiemens
    gK = 36 * b2.msiemens
    gNa = 1.5*120 * b2.msiemens
    C = 1 * b2.ufarad
    EM = -4 * b2.mV
    gM = 1.15 * b2.msiemens

    # forming HH model with SFA with differential equations
    eqs = '''
    I_e = curr(t) : amp
    membrane_Im = I_e + gNa*m**3*h*(ENa-vm) + gl*(El-vm) + \
            gK*n**4*(EK-vm) + gM*w*(EM-vm) : amp
    alphah = .07*exp(-.05*vm/mV)/ms    : Hz
    alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
    alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
    betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
    betam = 4*exp(-.0556*vm/mV)/ms : Hz
    betan = .125*exp(-.0125*vm/mV)/ms : Hz
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dw/dt = (wi-w)/tw : 1
    dvm/dt = membrane_Im/C : volt
    wi = 1/(1+exp(-3.5-.1*vm/mV)) : 1
    tauw = 1000*ms : second
    tw = (11.4*tauw)/(3.3*exp(1.75+0.05*vm/mV) + \
    exp(-1.75-0.05*vm/mV)) : second

    '''
