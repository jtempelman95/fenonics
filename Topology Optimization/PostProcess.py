import numpy as np
import matplotlib.pyplot as plt




def getbands(bands):
    '''
        Getting the band-gaps from a dispersion diagram
    '''
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
        Compute an objective function over a diagram
        
        nfile:      Number of training dispersion diagrams
        SamplrDim:  Dimension of LHS sampler (number of design variables)
        BGdata:     a list with each entry containing a different training dispersion diagram
        gaploc1:    The first gap location of interest
        gaploc2:
    
    '''
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
    
    
def plotvecs(bands, figsize = (5,4)):
    """
    Plot the eigenvectors of the disperison
    """
    from matplotlib.patches import Rectangle
    plt.figure(figsize=figsize)
    np1 = 20; np2 = 20; np3 = 20
    x1 = np.linspace(0,1-1/np1,np1)
    x2 = np.linspace(1,2-1/np1,np2)
    x3 = np.linspace(2,2+np.sqrt(2),np3)
    xx = np.concatenate((x1,x2,x3))
    nvec = 20
    maxfq = 5e3
    
    
    
    
#################################################
#            DATA PROECESSING                   #
################################################# 
def load_data(MshRs,Xdim,refinement_level, Nquads, SamplrDim, Nsamp, LHS_Seed, path):
    """
    Simple function for loading in training data
    """
    datapath = (path + '//SamplrSeed '  + str(LHS_Seed) +' SamplrDim '+  str(SamplrDim)   +' SamplrNgen '+  str(Nsamp)   
                                                    + '//Quads_' + str(Nquads) + ' Xdim_' 
                                                    + str(Xdim)    +  ' MshRs_'+ str(MshRs)
                                                    + ' rfnmt_' +  str(refinement_level) )
    os.listdir(datapath)
    dirs    = os.listdir(datapath)
    files   = os.listdir((datapath+'//'+dirs[0]))
    nfile   = len(files)
    
    #  Load in the data
    dvecs, splines,BGdata, Bands,Rads = [],[],[],[],[]
    for j in range(nfile):
        fname = str(j)+'.csv'
        dvec_dat = np.loadtxt(datapath+'//Dvecdata//'+fname)
        splne_dat = np.loadtxt(datapath+'//Splinecurves//'+fname, delimiter=',')
        BG_dat   = np.loadtxt(datapath+'//BGdata//'+fname, delimiter=',')
        Band_dat = np.loadtxt(datapath+'//dispersiondata//'+fname, delimiter=',')
        dvecs.append(dvec_dat)
        splines.append(splne_dat)
        BGdata.append(BG_dat)
        Bands.append(Band_dat)
        ns = splne_dat.shape[0]
        angles = np.linspace(0,2*np.pi,ns)
        rads   = np.sqrt(splne_dat[:,0]**2 + splne_dat[:,1]**2)
        rads_interp = np.interp(np.linspace(0,2*np.pi,200), angles, rads)
        Rads.append(rads_interp)
    return dvecs,splines,BGdata,Bands, Rads, datapath
      
def objfnc_bgloc(lowbounds,highbounds,quantity,gaploc = 1e3, centered = False):
    '''
    # ==============================
    # To check for band gap at a center frequnecy
    # ==============================
    '''
    optfnc = []
    if any(lowbounds):
        for j,lb in enumerate(lowbounds):
            if lowbounds[j] < gaploc and highbounds[j]>gaploc:
                if not centered:
                    optfnc = quantity[j] # - np.abs( gaploc- (highbounds[j]+lowbounds[j])/2)
                else:
                    optfnc = quantity[j] - np.abs( gaploc- (highbounds[j]+lowbounds[j])/2)
    if not isinstance(optfnc,float):
        if not any(optfnc):
            optfnc = 0
    return optfnc

def objfnc_bgHighLow(lowbounds,highbounds,quantity , gaploc = 1e3,  below = True):
    '''
    # ==============================
    # To check if above/below range
    # ==============================
    '''
    optfnc = []
    if any(lowbounds):
        if below:
            if np.min(lowbounds)<gaploc:
                optfnc = np.max(quantity[lowbounds<gaploc])
            else: optfnc = 0
        else:
            if np.max(highbounds)>gaploc:
                optfnc = np.max(quantity[highbounds>gaploc])
            else: optfnc = 0
    else: optfnc = 0
    return optfnc

def objfnc_bgrng(lowbounds,highbounds,gaploc_lower = 1e3,gaploc_upper = 4e3,  max = False):
    '''
    # ==============================
    # To check for band gap between two fequencies
    # ==============================
    '''
    optfnc = []
    if any(lowbounds):
        vals = []
        for j,lb in enumerate(lowbounds):
            # Full BG within bounds
            if lowbounds[j] > gaploc_lower and highbounds[j] < gaploc_upper :
                vals.append(highbounds[j] -lowbounds[j] )
            # BG clipped by top bound
            elif    lowbounds[j] > gaploc_lower and highbounds[j] > gaploc_upper and lowbounds[j] < gaploc_upper:
                vals.append(gaploc_upper-lowbounds[j] )
            # BG clipped by lower bound
            elif    lowbounds[j] < gaploc_lower and highbounds[j] < gaploc_upper and highbounds[j] > gaploc_lower:
                vals.append(highbounds[j] - gaploc_lower)
        if max:
            if np.array(vals).shape[0] == 0:
                optfnc = np.array(vals)
            else:
                optfnc = np.max(np.array(vals))
        else:
            optfnc = np.sum(np.array(vals))
    if not isinstance(optfnc,float):
        if not any(optfnc):
            optfnc = 0
    return optfnc


def objfun(bands, BG_op = 5):
    '''
    Computing an objective function from a band structure
    '''
    bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(bands)
    # ==============================
    # To check if band gap is in range
    # ==============================
    # quantity    =  gapwidths
    # opt1   =  0*objfnc_bgloc(lowbounds,highbounds,quantity,gaploc = 1e3, centered = True)*1
    # opt2   =  objfnc_bgloc(lowbounds,highbounds,quantity,gaploc = 2.5e3, centered = True)
    # ==============================
    # To check if bandgap is above/below 
    # ==============================
    # quantity    =  bgnrm
    # opt1   =  objfnc_bgHighLow(lowbounds,highbounds, quantity , gaploc = 1.5e3,   below = True)
    # opt2   =  objfnc_bgHighLow(lowbounds,highbounds, quantity , gaploc = 3.5e3,    below = False)
    # ==============================
    # To check band gaps within freq ranges
    # ==============================
    # optvec[k]  =  -objfnc_bgrng(lowbounds,highbounds,gaploc_lower = 1e2,gaploc_upper = 1e3,  max = False )
    # opt1  =   objfnc_bgrng(lowbounds,highbounds,gaploc_lower = 1e3, gaploc_upper = 1.5e3,    max = False)
    # opt2 =    objfnc_bgrng(lowbounds,highbounds,gaploc_lower = 3e3, gaploc_upper = 4e3,      max = False)
    # ==============================
    # Adhoc definition
    # ==============================
    # if any(lowbounds):
    #     optvec[k]  = np.sum(bgnrm)
    #     optvec2[k] = np.max(lowbounds)
    # else:
    #     optvec[k]  = 0
    #     optvec2[k] = 0
    
    B1   = BG_op
    B2   = 11
    # opt1 = np.max((np.min(bands[:,B1]) - np.max(bands[:,B1-1]) , 0 )) / np.max(bands[:,B1-1]) 
    opt2 = np.max((np.min(bands[:,B2]) - np.max(bands[:,B2-1]) , 0 )) / np.max(bands[:,B2-1])*0
    opt1 = (np.min(bands[:,B1]) - np.max(bands[:,B1-1]) ) / np.max(bands[:,B1-1]) 
    # optvec[k]  =  np.max(np.diff(bands[:, 4]))
    # optvec2[k]  = np.max(np.diff(bands[:, 4]))
    return opt1 + opt2

def multi_objfun(bands, BG_op = 5, BG_op2 = 5):
    '''
    Computing an objective function from a band structure
    '''
    bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(bands)
    # ==============================
    # To check if band gap is in range
    # ==============================
    # quantity    =  gapwidths
    # opt1   =  0*objfnc_bgloc(lowbounds,highbounds,quantity,gaploc = 1e3, centered = True)*1
    # opt2   =  objfnc_bgloc(lowbounds,highbounds,quantity,gaploc = 2.5e3, centered = True)
    # ==============================
    # To check if bandgap is above/below 
    # ==============================
    # quantity    =  bgnrm
    # opt1   =  objfnc_bgHighLow(lowbounds,highbounds, quantity , gaploc = 1.5e3,   below = True)
    # opt2   =  objfnc_bgHighLow(lowbounds,highbounds, quantity , gaploc = 3.5e3,    below = False)
    # ==============================
    # To check band gaps within freq ranges
    # ==============================
    # optvec[k]  =  -objfnc_bgrng(lowbounds,highbounds,gaploc_lower = 1e2,gaploc_upper = 1e3,  max = False )
    # opt1  =   objfnc_bgrng(lowbounds,highbounds,gaploc_lower = 1e3, gaploc_upper = 1.5e3,    max = False)
    # opt2 =    objfnc_bgrng(lowbounds,highbounds,gaploc_lower = 3e3, gaploc_upper = 4e3,      max = False)
    # ==============================
    # Adhoc definition
    # ==============================
    # if any(lowbounds):
    #     optvec[k]  = np.sum(bgnrm)
    #     optvec2[k] = np.max(lowbounds)
    # else:
    #     optvec[k]  = 0
    #     optvec2[k] = 0
    
    B1   = BG_op
    B2   = BG_op2
    #opt1 = np.max((np.min(bands[:,B1]) - np.max(bands[:,B1-1]) , 0 )) / np.max(bands[:,B1-1]) 
    opt2 = np.max((np.min(bands[:,B2]) - np.max(bands[:,B2-1]) , 0 )) / np.max(bands[:,B2-1])
    opt1 = (np.min(bands[:,B1]) - np.max(bands[:,B1-1]) ) / np.max(bands[:,B1-1]) 
    # opt1 = np.max(np.diff(bands[:, B1]))
    # optvec[k]  =  np.max(np.diff(bands[:, 4]))
    # optvec2[k]  = np.max(np.diff(bands[:, 4]))
    return opt1*B1 + opt2*B2


#################################################
#             POST PROESSES                     #
#################################################
def GetSpline(gmsh):   
    '''
    # ==============================
    # Get spline geometry from gmsh mesh
    # ==============================
    '''
    node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)[1]
    x_int       = node_interior[0::3]
    y_int       = node_interior[1::3]
    x_int       = np.concatenate([x_int,[x_int[0]]])
    y_int       = np.concatenate([y_int,[y_int[0]]])
    xi          = x_int - a_len/2
    yi          = y_int - a_len/2
    lxi         = len(xi)
    lxp         = len(xpt)
    xptnd       = np.array(xpt)
    yptnd       = np.array(ypt)
    lxp         = len(xptnd)
    xsv         = np.empty(xi.shape)
    SplineDat   = np.hstack( (xi.reshape(lxi,1), yi.reshape(lxi,1) ))  
    SplinePtDat = np.hstack( (xptnd.reshape(lxp,1), yptnd.reshape(lxp,1) ))  
    return SplinePtDat, SplineDat


#################################################
#             VISUALIZATION                     #
#################################################
def plot_regressor(regressor, xvec, features,savefigs):
    """
    Simple function for plotting the SVR over the data
    """
    #  DEFINE TEST GRID
    if features.shape[1] == 3:
        Ntest  = 10
        xtest  = np.linspace(0,.9,Ntest)
        ytest  = np.linspace(0,.9,Ntest)
        ztest  = np.linspace(0,.9,Ntest)
        Xtest,Ytest,Ztest = np.meshgrid(xtest,ytest,ztest)
        Xtst    = Xtest.reshape(Ntest**3,1)
        Ytst    = Ytest.reshape(Ntest**3,1)
        Ztst    = Ztest.reshape(Ntest**3,1)
        testdat = np.hstack( ( Xtst,Ytst,Ztst ))
        testpt  = y_pred = regressor.predict(testdat )
        origshape = Xtest.shape
    elif features.shape[1] == 2:
        Ntest  = 100
        xtest  = np.linspace(0,1,Ntest)
        ytest  = np.linspace(0,1,Ntest)
        Xtest,Ytest = np.meshgrid(xtest,ytest)
        Xtst    = Xtest.reshape(Ntest**2,1)
        Ytst    = Ytest.reshape(Ntest**2,1)
        testdat = np.hstack( ( Xtst,Ytst))
        testpt  = y_pred = regressor.predict(testdat )
        origshape = Xtest.shape
        
    # PLOT THE REGRESSOR FUNCTION
    if features.shape[1] == 3:
        fig     = plt.figure()
        ax      = plt.axes(projection='3d')
        ax.scatter(xvec[:,0],xvec[:,1],xvec[:,2], c = optvec, cmap = "jet")
        ax.view_init(elev=25, azim=-35, roll=0)    
        plt.show()
        fig = plt.figure()
        ax  = plt.axes(projection='3d')
        s1  = ax.scatter(Xtst,Ytst,Ztst, c = testpt, cmap = "jet")
        plt.colorbar(s1)
        ax.view_init(elev=25, azim=-35, roll=0)  
        plt.show()  
    elif features.shape[1] == 2:
        plt.figure(figsize=(5,5))
        plt.imshow(testpt.reshape(origshape),cmap = 'twilight_shifted', origin='lower', extent =(0,1,0,1) )
        plt.scatter(xvec[:,0],xvec[:,1], c = targets, cmap = 'twilight_shifted', edgecolors='k', s = 20)
        plt.xlabel('Design Feature 1')
        plt.ylabel('Design Feature 2')
        plt.colorbar()
        plt.title('SV Regressor')
        if savefigs: plt.savefig(figpath+'//Regressor.pdf', bbox_inches = 'tight')
        plt.show()
        
def riseplot(targets,predictions):
    """
    Simple function for the rise plot
    """
    x1 = targets
    x2 = predictions
    args = x1.argsort()
    plt.figure(figsize =(3,3))
    plt.plot(x2[args],'.', color = (0,  .7, 0), label = 'Prediction' )
    plt.plot(x1[args],'r-',label = 'Target')
    plt.legend()
    plt.xlabel('Sort Idx')
    plt.ylabel('Target Val')
    plt.title('Rise Plot')
    return plt

def targetplot(targets,predictions):
    """
    Simple function for the target
    """
    x1 = targets
    x2 = predictions
    args = x1.argsort()
    plt.figure(figsize =(3,3))
    plt.plot(x1[args],x2[args],'.', color = (0,.7,0), label = 'Prediction' )
    plt.plot(x1[args],x1[args],'r-',label = 'Target')
    plt.xlabel('Target Val')
    plt.ylabel('Target Val')
    plt.title('Target Plot')
    plt.legend()
    score = metrics.mean_squared_error(x1,x2)
    plt.title("Final score (MSE): {}".format(np.round(score,4)))
    return plt

def learningplot(train_loss, val_loss):
    """
    Simple function for plotting model leanrning
    """
    plt.figure(figsize =(3,3))
    plt.plot(train_loss,'b-',label = 'Training')
    plt.plot(val_loss, 'r-',label = 'Validation')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('lr = ' + str(optimizer.param_groups[0]['initial_lr'])+ ' $n_b$ = '+str(BATCH_SIZE))
    return plt

def plot_splines(splines,targets, nrow = 6, ncol = 6, is_rand = True):
    """
    Function for plotting the spline curves
    """
    # cm      = plt.get_cmap('twilight_shifted', 500)   # 11 discrete colors
    arg     = np.argsort(targets)
    cdata   = targets
    cdata   = pre.MinMaxScaler().fit_transform(cdata.reshape(-1,1))
    plt.figure(figsize=(5,5))
    plt.subplots_adjust(left    =0.1,bottom  =0.1,right   =0.9,top     =0.9, wspace  =0.05, hspace  =0.02)
    for idx in range(nrow*ncol):
        ax = plt.subplot(nrow,ncol, idx+1)
        if is_rand:
            idx = int(np.random.rand()*len(features))
        maxarg=arg[-idx]
        ax.plot(splines[maxarg][:,0],splines[maxarg][:,1] ,c=cm.twilight_shifted(cdata[maxarg]   ) ); 
        ax.axis('equal'); plt.axis('image')
        ax.set_xlim((-.05,.05))
        ax.set_ylim((-.05,.05))
        ax.set_xticks([])
        ax.set_yticks([])
    return plt

#################################################################
#                      MACHINE LEARNING                         #
#################################################################
class EarlyStopping():
    '''
    Class to determine if stopping criteria met in back propogation
    '''
    def __init__(self,model, patience = 50, min_delta = 1e-6, restore_best_weights = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model     = None
        self.best_loss  = None
        self.counter    = 0
        self.status     = ""
        
    def __call__(self, model, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
        if self.counter >= self.patience:
            self.status = f"Stopped on {self.counter}"
            if self.restore_best_weights:
                model.load_state_dict(self.best_model.state_dict())
            return True
        self.status = f"{self.counter}/{self.patience}"
        return False
    
    
def trainingloop(model,optimizer,x_train,y_train,x_test,y_test,dataloader_train, 
                    fair = True, patience = 25, min_delta = 1e-6, restore_best_weights = True):
        '''
        Function to excecute backpropagation with early stopping
        '''
        es          = EarlyStopping(model, patience = patience, min_delta = min_delta, 
                                    restore_best_weights = restore_best_weights)
        epoch       = 0
        done        = False
        history,  historyt   = [], []
        while epoch<300 and not done:
            epoch   += 1
            steps   = list(enumerate(dataloader_train))
            pbar    = tqdm.tqdm(steps)
            model.train()
            for i, (x_batch, y_batch) in pbar:
                y_batch_pred    = model(x_batch.to(device)).flatten()
                loss            = loss_fn(y_batch_pred , y_batch.flatten().to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss, current   = loss.item(), (i + 1)* len(x_batch)
                if i == len(steps)-1:
                    model.eval()
                    if not fair:
                        pred    = model(x_train).flatten()
                        vloss   = loss_fn(pred, y_train.flatten())
                    else:
                        pred    = model(x_test).flatten()
                        vloss   = loss_fn(pred, y_test.flatten())
                    history.append(float(vloss))
                    pred    = model(x_train).flatten()
                    tloss   = loss_fn(pred, y_train.flatten())
                    historyt.append(float(tloss))
                    if es(model,vloss): done = True
                    pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, EStop:[{es.status}]")
                else:
                    pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
            y_pred = model(x_train).flatten()
            mse = loss_fn(y_pred,y_train.flatten())
        return model, history, historyt
        # scheduler1.step()
    
class NetTanh(nn.Module):
    def __init__(self, D_in, H, D, D_out):
        """
        In the constructor, instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(NetTanh, self).__init__()
        self.inputlayer         = nn.Linear(D_in, H)
        self.middle             = nn.Linear(H, H)
        self.lasthiddenlayer    = nn.Linear(H, D)
        self.outputlayer        = nn.Linear(D, D_out)
    def forward(self, x):
        """
        In the forward function, accept a variable of input data and return
        a variable of output data. Use modules defined in the constructor, as
        well as arbitrary operators on variables.
        """
        y_pred = self.outputlayer(self.PHI(x))
        return y_pred
        
    def PHI(self, x):
        h_relu = self.inputlayer(x).tanh()
        for i in range(3):
            h_relu = self.middle(h_relu).tanh()
            # x = F.relu(x)
        phi = self.lasthiddenlayer(h_relu)
        return phi
    

class Net(nn.Module):
    """
    Define a simple fully connected model
    """
    def __init__(self, in_count, out_count, First = 128 , deep  = 32, Ndeep = 4, isrelu = True):
        super(Net,self).__init__()
        d2 = deep
        self.Ndeep  = Ndeep
        self.fc1    = nn.Linear(in_count,First,bias = True)
        self.fc2    = nn.Linear(First,d2,bias = True)
        self.fc3    = nn.Linear(d2,d2,bias = True)
        self.fc4    = nn.Linear(d2,d2,bias = True)
        self.do     = nn.Dropout(.25)
        self.fcend  = nn.Linear(d2,out_count)
        self.tanh   = nn.Tanh()
        if isrelu:
            self.relu   = nn.ReLU()
        else:
            self.relu   = nn.Tanh()
            
        self.seq    = nn.Sequential(
                    nn.Linear(in_count, 24),
                    nn.ReLU(),
                    nn.Linear(24, 12),
                    nn.ReLU(),
                    nn.Linear(12, 6),
                    nn.ReLU(),
                    nn.Linear(6, 1)
                )
    def forward(self, x):
        # x = self.seq(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        for i in range(self.Ndeep):
            x = self.relu(self.fc4(x)) 
        return self.fcend(x)
    
class Netc(nn.Module):
    """
    Define a simple 1D CNN
    """
    def __init__(self):
        super(Netc, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.linear(x)
    
class Simple1DCNN(torch.nn.Module):
    """
    Define a simple 1D CNN
    """
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.fc1 = torch.nn.Linear(3,7)
        self.layer1 = torch.nn.Conv1d(in_channels=32, out_channels=20, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)
        self.fc2 = torch.nn.Linear(20,1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.fc2(x)
        return F.limear(x)