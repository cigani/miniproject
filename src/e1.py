import hhBasic
import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

hex = hhBasic.hhStep(doPlot=False)

print hex.vm[0]
print hex.ninf[0]
print hex.minf[0]
print hex.hinf[0]


traceall = np.append(hex.minf, [hex.ninf[0], hex.hinf[0]])
nrmfactor = np.max(traceall)/b2.mV


plt.subplot(311)
plt.plot(hex.t/b2.ms, hex.ninf[0] / nrmfactor / b2.mV, 'black', lw=2)
plt.ylabel('n activation')
plt.subplot(312)
plt.plot(hex.t/b2.ms, hex.minf[0] / nrmfactor /b2.mV, 'blue', lw=2)
plt.ylabel('m activation')
plt.subplot(313)
plt.plot(hex.t/b2.ms, hex.hinf[0] / nrmfactor /b2.mV, 'red', lw=2)
plt.ylabel('h activation')


plt.show()
