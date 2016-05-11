import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

def gatevar(hex):

    traceall = np.append(hex.m[0], [hex.n[0], hex.h[0]])
    nrmfactor = np.max(traceall)/b2.mV

    # Plot of activation variables (inf variants) vs time
    plt.subplot(411)
    plt.grid()
    plt.plot(hex.t/b2.ms, hex.vm[0]/b2.mV, lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel('v [mV]')
    
    plt.subplot(412)
    plt.plot(hex.t/b2.ms, hex.n[0] / nrmfactor / b2.mV, 'black',lw=2,label='n')
    plt.ylabel('act./inact.')
    plt.xlabel('t [ms]')
    
    plt.plot(hex.t/b2.ms, hex.m[0] / nrmfactor /b2.mV, 'blue', lw=2,label='m')
    plt.ylabel('act./inact.')
    plt.xlabel('t [ms]')
    
    plt.plot(hex.t/b2.ms, hex.h[0] / nrmfactor /b2.mV, 'red', lw=2,label='h')
    plt.ylabel('act./inact.')
    plt.xlabel('t [ms]')
    
    plt.legend(loc='upper right')
    
    plt.subplot(413)
    plt.plot(hex.t/b2.ms, hex.I_e[0]/b2.uamp,lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel('I [uA]')
    
    plt.subplot(414)
    plt.plot(hex.t/b2.ms, hex.NaI[0]/b2.uamp/1000,lw=2,label='Na')
    plt.plot(hex.t/b2.ms, hex.KI[0]/b2.uamp/1000,lw=2,label='K')
    plt.xlabel('t [ms]')
    plt.ylabel('I [uA]')
    plt.legend(loc='upper right')
    plt.suptitle('Current/Voltage dynamics')
    plt.show()
    
    plt.subplot(311)
    plt.plot(hex.vm[0]/b2.mV, hex.n[0] /nrmfactor /b2.mV, 'black', lw=2)
    plt.ylabel('A.U')
    plt.legend('ninf')
    plt.subplot(312)
    plt.plot(hex.vm[0]/b2.mV, hex.m[0] /nrmfactor /b2.mV, 'blue', lw=2)
    plt.ylabel('A.U')
    plt.legend('minf')
    plt.subplot(313)
    plt.plot(hex.vm[0]/b2.mV, hex.h[0] /nrmfactor /b2.mV, 'red', lw=2)
    plt.ylabel('A.U')
    plt.legend('hinf')
    plt.xlabel('v [mV]')
    plt.suptitle('Activation/Deactivation vs Voltage')
    plt.show()


def gatevar_A(hex):

    traceall = np.append(hex.m[0], [hex.n[0], hex.h[0]])
    nrmfactor = np.max(traceall)/b2.mV

    # Plot of activation variables (inf variants) vs time
    plt.subplot(411)
    plt.grid()
    plt.plot(hex.t/b2.ms, hex.vm[0]/b2.mV, lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel('v [mV]')
    
    plt.subplot(412)
    plt.plot(hex.t/b2.ms, hex.n[0] / nrmfactor / b2.mV, 'black',lw=2,label='n')
    plt.ylabel('act./inact.')
    plt.xlabel('t [ms]')
    
    plt.plot(hex.t/b2.ms, hex.m[0] / nrmfactor /b2.mV, 'blue', lw=2,label='m')
    plt.ylabel('act./inact.')
    plt.xlabel('t [ms]')
    
    plt.plot(hex.t/b2.ms, hex.h[0] / nrmfactor /b2.mV, 'red', lw=2,label='h')
    plt.ylabel('act./inact.')
    plt.xlabel('t [ms]')

    plt.plot(hex.t/b2.ms, hex.w[0] / nrmfactor /b2.mV, 'green', lw=2,label='w')
    plt.ylabel('act./inact.')
    plt.xlabel('t [ms]')
    
    plt.legend(loc='upper right')
    
    plt.subplot(413)
    plt.plot(hex.t/b2.ms, hex.I_e[0]/b2.uamp,lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel('I [uA]')
    
    plt.subplot(414)
    plt.plot(hex.t/b2.ms, hex.NaI[0]/b2.uamp/1000,lw=2,label='Na')
    plt.plot(hex.t/b2.ms, hex.KI[0]/b2.uamp/1000,lw=2,label='K')
    plt.plot(hex.t/b2.ms, hex.MI[0]/b2.uamp/1000,lw=2,label='Kslow')
    plt.xlabel('t [ms]')
    plt.ylabel('I [uA]')
    plt.legend(loc='upper right')
    plt.suptitle('Current/Voltage dynamics')
    plt.show()
    
    plt.subplot(411)
    plt.plot(hex.vm[0]/b2.mV, hex.n[0] /nrmfactor /b2.mV, 'black', lw=2)
    plt.ylabel('A.U')
    plt.legend(loc='upper right')
    plt.subplot(412)
    plt.plot(hex.vm[0]/b2.mV, hex.m[0] /nrmfactor /b2.mV, 'blue', lw=2)
    plt.ylabel('A.U')
    plt.legend(loc='upper right')
    plt.subplot(413)
    plt.plot(hex.vm[0]/b2.mV, hex.h[0] /nrmfactor /b2.mV, 'red', lw=2)
    plt.ylabel('A.U')
    plt.legend(loc='upper right')
    plt.subplot(414)
    plt.plot(hex.vm[0]/b2.mV, hex.w[0] /nrmfactor /b2.mV, 'green', lw=2)
    plt.ylabel('A.U')
    plt.legend(loc='upper right')
    plt.xlabel('v [mV]')
    plt.suptitle('Activation/Deactivation vs Voltage')
    plt.show()

