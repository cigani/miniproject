import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

def gatevar(hex):

    traceall = np.append(hex.mi[0], [hex.ni[0], hex.hi[0]])
    nrmfactor = np.max(traceall)/b2.mV


    # Plot of activation variables (inf variants) vs time
    plt.subplot(411)
    plt.grid()
    plt.plot(hex.t/b2.ms, hex.vm[0]/b2.mV, lw=2)
    plt.ylabel('v [mV]')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplot(412)
    plt.plot(hex.t/b2.ms, hex.ni[0] / nrmfactor / b2.mV,
             'black',lw=2,label='ninf')
    plt.ylabel('act./inact.')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.plot(hex.t/b2.ms, hex.mi[0] / nrmfactor /b2.mV,
             'blue', lw=2,label='minf')
    plt.ylabel('act./inact.')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.plot(hex.t/b2.ms, hex.hi[0] / nrmfactor /b2.mV,
             'red', lw=2,label='hinf')
    plt.ylabel('act./inact.')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.legend(loc='upper right')
    plt.subplot(413)
    plt.plot(hex.t/b2.ms, hex.I_e[0]/b2.uamp,lw=2)
    plt.ylim([0, np.max(hex.I_e[0]/b2.uamp)+1])
    plt.ylabel('I [uA]')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplot(414)
    plt.plot(hex.t/b2.ms, hex.NaI[0]/b2.uamp/1000,lw=2,label='Na')
    plt.plot(hex.t/b2.ms, hex.KI[0]/b2.uamp/1000,lw=2,label='K')
    plt.xlabel('t [ms]')
    plt.ylabel('I [uA]')
    plt.legend(loc='upper right')
    plt.suptitle('Current/Voltage dynamics')
    plt.savefig('current_voltage_dyn.eps', format='eps',dpi=1200)
    plt.show()

    plt.subplot(311)
    plt.plot(hex.vm[0]/b2.mV, hex.ni[0] /nrmfactor /b2.mV,
             'black', lw=2, label='ninf')
    plt.ylabel('act.deact')
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplot(312)
    plt.plot(hex.vm[0]/b2.mV, hex.mi[0] /nrmfactor /b2.mV,
             'blue', lw=2, label='minf')
    plt.ylabel('act.deact')
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplot(313)
    plt.plot(hex.vm[0]/b2.mV, hex.hi[0] /nrmfactor /b2.mV,
             'red', lw=2, label='hinf')
    plt.ylabel('act.deact')
    plt.legend(loc='upper right')
    plt.xlabel('v [mV]')
    plt.suptitle('Activation/Deactivation vs Voltage')
    plt.savefig('act_deact_vs_voltage.eps', format='eps',dpi=1200)
    plt.show()


def gatevar_A(hex):

    traceall = np.append(hex.mi[0], [hex.ni[0], hex.hi[0]])
    nrmfactor = np.max(traceall)/b2.mV


    plt.subplot(411)
    plt.grid()
    plt.plot(hex.t/b2.ms, hex.vm[0]/b2.mV, lw=2)

    plt.ylabel('v [mV]')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplot(412)
    plt.plot(hex.t/b2.ms, hex.n[0] / nrmfactor / b2.mV,
             'black',lw=2,label='n')
    plt.ylabel('act./inact.')


    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.plot(hex.t/b2.ms, hex.m[0] / nrmfactor /b2.mV, 'blue',
             lw=2,label='m')
    plt.ylabel('act./inact.')


    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.plot(hex.t/b2.ms, hex.h[0] / nrmfactor /b2.mV, 'red',
             lw=2,label='h')
    plt.ylabel('act./inact.')


    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.plot(hex.t/b2.ms, hex.w[0] / nrmfactor /b2.mV, 'green',
             lw=2,label='w')
    plt.ylabel('act./inact.')


    plt.legend(loc='upper right')

    plt.subplot(413)
    plt.plot(hex.t/b2.ms, hex.I_e[0]/b2.uamp,lw=2)

    plt.ylabel('I [uA]')

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplot(414)
    plt.plot(hex.t/b2.ms, hex.NaI[0]/b2.uamp/1000,lw=2,label='Na')
    plt.plot(hex.t/b2.ms, hex.KI[0]/b2.uamp/1000,lw=2,label='K')
    plt.plot(hex.t/b2.ms, hex.MI[0]/b2.uamp/1000,lw=2,label='Kslow')
    plt.xlabel('t [ms]')
    plt.ylabel('I [uA]')
    plt.legend(loc='upper right')
    plt.suptitle('Current/Voltage dynamics')
    plt.savefig('current_voltage_dyn_a.eps', format='eps', dpi=1200)
    plt.show()

    plt.subplot(411)
    plt.plot(hex.vm[0]/b2.mV, hex.ni[0] /nrmfactor /b2.mV, 'black', lw=2, label='ninf')
    plt.ylabel('act.deact')
    plt.legend(loc='upper right')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.subplot(412)
    plt.plot(hex.vm[0]/b2.mV, hex.mi[0] /nrmfactor /b2.mV, 'blue', lw=2, label='minf')
    plt.ylabel('act.deact')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.legend(loc='upper right')
    plt.subplot(413)
    plt.plot(hex.vm[0]/b2.mV, hex.hi[0] /nrmfactor /b2.mV, 'red', lw=2, label='hinf')
    plt.ylabel('act.deact')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.legend(loc='upper right')
    plt.subplot(414)
    plt.plot(hex.vm[0]/b2.mV, hex.wi[0] /nrmfactor /b2.mV, 'green', lw=2, label='winf')
    plt.ylabel('act.deact')
    plt.legend(loc='upper right')
    plt.xlabel('v [mV]')
    plt.suptitle('Activation/Deactivation vs Voltage')
    plt.savefig('act_deact_vs_voltage_a.eps', format='eps',dpi=1200)
    plt.show()

