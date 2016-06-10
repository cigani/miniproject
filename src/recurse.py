import numpy as np

def recursive_sum(nested_num_list):
    the_sum = 0
    for element in nested_num_list:
        if type(element) == list:
            the_sum = the_sum + recursive_sum(element)
        else:
            the_sum = the_sum + element
    return the_sum

paraSpace = [
np.linspace(1e-5,5,50),
np.linspace(0.001,5,50),
np.linspace(1e-7,1,50),
np.linspace(1e-7,1,50),
np.linspace(0.001,5,50),
np.linspace(1e-4,5,50),
np.linspace(.1,100,50),
np.linspace(.001,100,50),
np.linspace(0.05,5,50)]

print recursive_sum(paraSpace)

print len(recursive_sum(paraSpace))
