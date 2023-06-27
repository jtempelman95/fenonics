import numpy as np
import matplotlib.pyplot as plt

def getbands(bands):
    '''
        Getting the band-gaps from a dispersion diagram
    '''
    #################################################################
    #               Idenitfy the band gaps                          #
    #################################################################
    evals_disp = bands
    nvec = bands.shape[1]
    nK = bands.shape[0]
    eigfqs      = np.array(evals_disp)
    ef_vec      = eigfqs.reshape((1,nvec*nK))
    evals_all   = np.sort(ef_vec).T
    deval   =   np.diff(evals_all.T).T
    args    =   np.flip(np.argsort(deval.T)).T
    lowlim  = []
    uplim   = []
    bgs     = []

    # =======================
    # Finding the boundaries of the pass bands
    # =======================
    lowb      = []; uppb      = []
    for k in range(nvec):
        lowb.append(np.min(eigfqs.T[k]))
        uppb.append(np.max(eigfqs.T[k]))
    for k in range(nvec):
        LowerLim =  np.max(eigfqs[:,k])
        if k < nvec-1:
            UpperLim =  np.min(eigfqs[:,k+1])
        else:
            UpperLim =  np.min(eigfqs[:,k])
        # =======================
        # Check if these limits fall in a pass band
        # =======================
        overlap = False
        for j in range(nvec):
            if LowerLim > lowb[j] and LowerLim <uppb[j]:
                overlap = True            
            if UpperLim > lowb[j] and UpperLim <uppb[j]:
                overlap = True
        if overlap == False:
            # print('isbg')
            lowlim.append(LowerLim)
            uplim.append(UpperLim)
            bgs.append(UpperLim - LowerLim)
            
    # Filter band gaps
    maxfq           = np.max(eigfqs[:])
    isgap           = [i for i,v in enumerate(bgs) if v > np.median(deval)] 
    gaps            = np.array(bgs)
    lower           = np.array(lowlim)
    higher          = np.array(uplim)
    gapwidths       = gaps[isgap]
    lowbounds       = lower[isgap]
    highbounds      = higher[isgap]
    BG_normalized   = gapwidths/(.5*lowbounds  + .5*highbounds)
    
    
    
    return BG_normalized, gapwidths, gaps, lowbounds, highbounds



def objfun(BGdata,gaploc1,gaploc2):
    '''
        Process the loaded data
    '''
    #################################################
    #           COMPUTE OBJECTIVE FUNCTION
    #################################################
    optvec      = np.zeros(nfile)
    optvec2     = np.zeros(nfile)
    xvec        = np.zeros((nfile,SamplrDim))
    for k in range(len(BGdata)):
        bands = Bands[k]
        bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(bands)
        optfnc = []
        if not np.any(optfnc):
            optfnc = 0
        else:
            # optfnc = optfnc[0]
            optfnc =  optfnc[-1] + optfnc[0] 
        
        optfnc = []
        gaploc = 4.5e3
        if any(lowbounds):
            for j,lb in enumerate(lowbounds):
                if lowbounds[j] < gaploc1 and highbounds[j]>gaploc1:
                    optfnc = gapwidths[j]  - np.abs( gaploc1- (highbounds[j]+lowbounds[j])/2)
            # optfnc = np.max(bgnrm)
        if not isinstance(optfnc,float):
            if not any(optfnc):
                optfnc = 0
        optvec[k]       = optfnc
        
        optfnc = []
        gaploc = 2e3
        if any(lowbounds):
            for j,lb in enumerate(lowbounds):
                if lowbounds[j] < gaploc2 and highbounds[j]>gaploc2:
                    optfnc = gapwidths[j]  - np.abs( gaploc2- (highbounds[j]+lowbounds[j])/2)
        if not isinstance(optfnc,float):
            if not any(optfnc):
                optfnc = 0
        optvec2[k]  = optfnc
        
        # optvec2[k]  = np.max(np.abs(np.diff(bands[:,0])))
        # optvec2[k]  = np.max(np.diff(bands.reshape(60*20,)))
        xvec[k,0]   = dvecs[k][0]
        xvec[k,1]   = dvecs[k][1]
        if len(xvec[0,:]) > 2:
            xvec[k,2]   = dvecs[k][2]
            
            
            
        
        
 #################################################################
#            FIG : Plotting the dispersion                     #
#################################################################    
def plotbands(bands, figsize = (5,4)):
    """
    Plot the disprsion bands with bandgaps highlighted
    """ 
    bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(bands)

    from matplotlib.patches import Rectangle
    plt.figure(figsize=figsize)
    np1 = 20; np2 = 20; np3 = 20
    x1 = np.linspace(0,1-1/np1,np1)
    x2 = np.linspace(1,2-1/np1,np2)
    x3 = np.linspace(2,2+np.sqrt(2),np3)
    xx = np.concatenate((x1,x2,x3))
    nvec = 20
    maxfq = 5e3
    # PLOT THE DISPERSION BANDS
    for n in range(nvec):
        ev = bands[:,n]
        if n == 0:
            plt.plot( xx,(ev),'b.-',markersize = 3, label = 'Bands')
        else:
            plt.plot( xx,(ev),'b.-',markersize = 3)
    plt.grid(color='gray', linestyle='-', linewidth=0.2)
    plt.xticks(np.array([0,1,2,2+np.sqrt(2)]), ['$\Gamma$', 'X', 'M', '$\Gamma$'],fontsize=18)
    plt.xlabel(r'Wave Vector ',fontsize=18)
    plt.ylabel('$\omega$ [rad/s]',fontsize=18)
    plt.title('Dispersion Diagram',fontsize = 18)
    plt.xlim((0,np.max(xx)))
    plt.ylim((0,np.max(maxfq)))
    currentAxis = plt.gca()
    for j in range(len(gapwidths)):
        lb = lowbounds[j]
        ub = highbounds[j]
        if j == 0:
            currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="g" ,ec='none', alpha =.3,label='bangap'))
        else:
            currentAxis.add_patch( Rectangle((np.min(xx),lb), np.max(xx), ub-lb,  facecolor="g" ,ec='none', alpha =.3))
    plt.legend()
    return plt
    ###########################################################
    