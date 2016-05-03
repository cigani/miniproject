import hhBasic
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

hex = hhBasic.hhStep(iAmp=3,ntype=3,itStart=20, itEnd=200, tEnd=250, doPlot=False)

traceall = np.append(hex.minf[0], [hex.ninf[0], hex.hinf[0]])
nrmfactor = np.max(traceall)/b2.mV

# Plot of activation variables (inf variants) vs time
plt.subplot(411)
plt.grid()
plt.plot(hex.t/b2.ms, hex.vm[0]/b2.mV, lw=2)
plt.xlabel('t [ms]')
plt.ylabel('v [mV]')
#plt.axis((15,35, min(hex.vm[0]/b2.mV)*1.1,max(hex.vm[0]/b2.mV)*1.1))


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

plt.plot(hex.t/b2.ms, hex.p[0] / nrmfactor /b2.mV, 'green', lw=2,label='p')
plt.ylabel('act./inact.')
plt.xlabel('t [ms]')

plt.legend(loc='upper right')
#plt.axis((15,35, 0,max(hex.m[0])*1.5))

plt.subplot(413)
plt.plot(hex.t/b2.ms, hex.i_e[0]/b2.uamp,lw=2)
plt.xlabel('t [ms]')
plt.ylabel('I [uA]')
#plt.axis((
#    15,
#    35,
#    min(hex.i_e[0]/b2.uamp)*1.1,
#    max(hex.i_e[0]/b2.uamp)*1.1
#))

plt.subplot(414)
plt.plot(hex.t/b2.ms, hex.NaI[0]/b2.uamp/1000,lw=2,label='Na')
plt.plot(hex.t/b2.ms, hex.KI[0]/b2.uamp/1000,lw=2,label='K')
plt.plot(hex.t/b2.ms, hex.KIslow[0]/b2.uamp/1000,lw=2,label='Kslow')
plt.xlabel('t [ms]')
plt.ylabel('I [uA]')
#plt.axis((15,35,
#         min(hex.KI[0]/b2.uamp/1000)*1.3,
#         max(hex.NaI[0]/b2.uamp/1000)*1.3))
plt.legend(loc='upper right')
plt.suptitle('Current/Voltage dynamics')
plt.show()


# Keeping this guy just to remind me how to set the limits like this

'''
plt.plot(hex.t/b2.ms, hex.i_e[0]/b2.uamp, lw=2)
plt.axis((
    0,
    np.max(hex.t/b2.ms),
    min(hex.i_e[0]/b2.uamp)*1.1,
    max(hex.i_e[0]/b2.uamp)*1.1
))
plt.xlabel('t [ms]')
plt.ylabel('I [uA]')
plt.show()
'''

# Plot of the activtion variables [inf variants] vs time i think it works
# TODO: find units of n/m/h (Arbitrary units i think).
plt.subplot(311)
plt.plot(hex.vm[0]/b2.mV, hex.ninf[0] /nrmfactor /b2.mV, 'red', lw=2)
plt.ylabel('A.U')
plt.legend('ninf')
plt.subplot(312)
plt.plot(hex.vm[0]/b2.mV, hex.minf[0] /nrmfactor /b2.mV, 'black', lw=2)
plt.ylabel('A.U')
plt.legend('minf')
plt.subplot(313)
plt.plot(hex.vm[0]/b2.mV, hex.hinf[0] /nrmfactor /b2.mV, 'blue', lw=2)
plt.ylabel('A.U')
plt.legend('hinf')
plt.xlabel('v [mV]')
plt.suptitle('Activation/Deactivation vs Voltage')
plt.show()


