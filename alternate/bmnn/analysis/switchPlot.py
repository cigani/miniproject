import seaborn
import matplotlib.pyplot as plt
import sys
import brian2 as b2
import numpy as np
class switchPlot(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def unitPlot(string):
    # Turns out this is a pita to make work. Just feed corrected values
    units = {
        'Voltage' : 'b2.mV',
        'Time' : 'b2.ms',
        'Amplitude' : 'b2.uamp'
            }
    for key, value in units.iteritems():
        if any(s in string for s in (value,key)):
            return value


def casePlot(title, xval, yval, ylbl, xlbl, lgnd, plotType, args):
    for case in switchPlot(plotType):
        if case('sd'):
            epsName = '{}.eps'.format(''.join(title.split()))
            svgName = '{}.svg'.format(''.join(title.split()))
            plt.plot(xval,
                     yval,
                     c='blue',lw=2)
            plt.plot((args,args),
                     (np.min(yval),
                      np.max(yval)),
                     c='red')
            plt.ylabel(ylbl)
            plt.xlabel(xlbl)
            plt.legend((lgnd))
            plt.grid()
            plt.savefig(epsName, format='eps', dpi=1200)
            plt.savefig(svgName, format='svg', dpi=1200)

'''
assert unitPlot('v [mV]')==b2.mV
assert unitPlot('t [ms]')==b2.ms
assert unitPlot('Trace Voltage and Spike')==b2.mV
assert unitPlot('amp')==b2.uamp
'''



