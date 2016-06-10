import re
units = {'Voltage' : mv, 'Time' : 'ms'}
#print units
#print units['Voltage']
#print units.items()

string = "Voltage SPike Timing"

for key, value in units.items():
    #print 'key ' + key
    #print  value
    if key in string:
        print value




