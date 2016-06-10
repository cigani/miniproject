def spikeGet(t,v,vT=None):
    #simple formula to get a sane vT
    if vT == None:
        vT = [(0.75*(np.max(v))), 10]
        vT = np.max(vT)
    vTF = v > vT

    # use numpy's index-a-index functionality to extract T/F shift point
    # this gives the point where false becomes true.
    idx = np.nonzero((vTF[0][:-1]==0) & (vTF[0][1:]==1))

    # we want the point one time point further though so index +1
    return t[idx[0]+1]
