import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt

def tau_plot(hex, do_plot=False):
    plt.subplot(311)
    plt.plot(hex.vm[0]/b2.mV, hex.tn[0]/b2.ms, 'black', lw=2)
    plt.ylabel('t [1/s]')
    plt.legend('n')
    
    plt.subplot(312)
    plt.plot(hex.vm[0]/b2.mV, hex.tm[0]/b2.ms, 'blue', lw=2)
    plt.ylabel('t [1/s]')
    plt.legend('m')
    
    plt.subplot(313)
    plt.plot(hex.vm[0]/b2.mV, hex.th[0]/b2.ms, 'red', lw=2)
    plt.ylabel('t [1/s]')
    plt.legend('h')

    plt.xlabel('v [mV]')
    plt.suptitle('Activation/Deactivation time constants')
    plt.show()

def tau_plot_A(hex, do_plot=False):
    plt.subplot(411)
    plt.plot(hex.vm[0]/b2.mV, hex.tn[0]/b2.ms, 'black', lw=2)
    plt.ylabel('t [1/s]')
    plt.legend('n')
    
    plt.subplot(412)
    plt.plot(hex.vm[0]/b2.mV, hex.tm[0]/b2.ms, 'blue', lw=2)
    plt.ylabel('t [1/s]')
    plt.legend('m')
    
    plt.subplot(413)
    plt.plot(hex.vm[0]/b2.mV, hex.th[0]/b2.ms, 'red', lw=2)
    plt.ylabel('t [1/s]')
    plt.legend('h')
    
    plt.subplot(414)
    plt.plot(hex.vm[0]/b2.mV,hex.tw[0]/b2.ms, 'green', lw=2)
    plt.ylabel('t [1/s]')
    plt.legend('w')
    plt.xlabel('v [mV]')
    plt.suptitle('Activation/Deactivation time constants')
    plt.show()
