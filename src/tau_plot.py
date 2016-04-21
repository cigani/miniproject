import brian2 as b2
import hhBasic
import numpy as np
import matplotlib.pyplot as plt


hex = hhBasic.hhStep(doPlot=False)

print hex.tm[0]

traceall = np.append(hex.minf[0], [hex.ninf[0], hex.hinf[0]])
nrmfactor = np.max(traceall)/b2.mV


plt.plot(hex.vm[0]/b2.mV, hex.tm[0]/b2.ms, 'blue', lw=2)
plt.show()
