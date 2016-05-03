import brian2 as b2
import hhBasic
import numpy as np
import matplotlib.pyplot as plt
import seaborn

hex = hhBasic.hhStep(doPlot=False,ntype=3)

plt.subplot(411)
plt.plot(hex.vm[0]/b2.mV, hex.tn[0]/b2.ms, 'blue', lw=2)
plt.ylabel('t [1/s]')
plt.legend('n')

plt.subplot(412)
plt.plot(hex.vm[0]/b2.mV, hex.tm[0]/b2.ms, 'black', lw=2)
plt.ylabel('t [1/s]')
plt.legend('m')

plt.subplot(413)
plt.plot(hex.vm[0]/b2.mV, hex.th[0]/b2.ms, 'red', lw=2)
plt.ylabel('t [1/s]')
plt.legend('h')

plt.subplot(414)
plt.plot(hex.vm[0]/b2.mV,hex.tp[0]/b2.ms, 'green', lw=2)

'''plt.plot(hex.vm[0]/b2.mV, hex.tm[0]/b2.ms, 'black', lw=2)
plt.ylabel('t [1/s]')
plt.plot(hex.vm[0]/b2.mV, hex.th[0]/b2.ms, 'red', lw=2)
plt.ylabel('t [1/s]')
plt.plot(hex.vm[0]/b2.mV, hex.tn[0]/b2.ms, 'blue', lw=2)
plt.ylabel('t [1/s]')'''

plt.xlabel('v [mV]')
plt.suptitle('Activation/Deactivation time constants')
plt.show()

