import hhBasic
import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

hex = hhBasic.hhStep(itStart=20, itEnd=35, tEnd=35, doPlot=False)

traceall = np.append(hex.minf[0], [hex.ninf[0], hex.hinf[0]])
nrmfactor = np.max(traceall)/b2.mV

# Plot of activation variables (inf variants) vs time
plt.subplot(411)
plt.plot(hex.t/b2.ms, hex.vm[0]/b2.mV, lw=2)
plt.xlabel('t [ms]')
plt.ylabel('v [mV]')
plt.grid()

plt.subplot(412)
plt.plot(hex.t/b2.ms, hex.n[0] / nrmfactor / b2.mV, 'black', lw=2)
plt.ylabel('act./inact.')
plt.xlabel('t [ms]')
plt.legend('n')

plt.subplot(413)
plt.plot(hex.t/b2.ms, hex.m[0] / nrmfactor /b2.mV, 'blue', lw=2)
plt.ylabel('act./inact.')
plt.xlabel('t [ms]')
plt.legend('m')

plt.subplot(414)
plt.plot(hex.t/b2.ms, hex.h[0] / nrmfactor /b2.mV, 'red', lw=2)
plt.ylabel('act./inact.')
plt.xlabel('t [ms]')
plt.legend('h')

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

plt.show()


