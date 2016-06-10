import hhBasic as hh
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import brian2 as b2

hex = hh.hhStep(ntype=3, doPlot=True)

plt.plot(hex.t/b2.ms, hex.NaI[0]/b2.uamp,lw=2,label='Na')
plt.plot(hex.t/b2.ms, hex.KI[0]/b2.uamp,lw=2,label='K')
plt.plot(hex.t/b2.ms, hex.KIslow[0]/b2.uamp,lw=2,label='Kslow')
plt.legend(loc='upper right')
plt.show()
