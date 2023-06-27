"""
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
             POST PROCESSING/OPTIMIZING SIMULATION RESULTS
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
    Josh Tempelman
    Universrity of Illinois
    jrt7@illinois.edu
    
    Originated      MARCH 10, 2023
    Last Modified   MARCH 24, 2023
    
    ! ! Do not distribute ! !
    
    
    About:
        This program loads in dispersion data from FEM
        and fits surrogate models to the band diagram objective functions.
        Optimization is performed over the surrogate model and then re-evalated
        by an FEM script (imported form FEM_Functions file).
    
        
"""

#%%
''' 
======================================================================
                
                IMPORT DEPENDENCIES AND INITIALIZE FCNS
                
======================================================================
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage        import measure
from sklearn.svm    import SVR
from sklearn.tree   import DecisionTreeRegressor
from sklearn        import metrics
from sklearn.preprocessing import MinMaxScaler  
from matplotlib     import cm
from sklearn        import preprocessing as pre
from PostProcess    import *
# from GenerateData import *
from FEM_Functions  import *
os.sys.path.append("./../") 

import torch.nn as nn
import copy
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import tqdm
import torch

device = "mps" if getattr(torch,'has_mps',False) \
    else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

'''      
        *******************************         
        ***    SUPPORT FUNCTIONS    ***
        *******************************          
'''

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
    # opt1   = 0* objfnc_bgHighLow(lowbounds,highbounds, quantity , gaploc = 2e3,   below = True)
    # opt2   =  objfnc_bgHighLow(lowbounds,highbounds, quantity , gaploc = 3e3,    below = False)
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
    opt1 = np.max((np.min(bands[:,B1]) - np.max(bands[:,B1-1]) , 0 )) / np.max(bands[:,B1-1]) 
    opt2 = np.max((np.min(bands[:,B2]) - np.max(bands[:,B2-1]) , 0 )) / np.max(bands[:,B2-1])*0
    opt1 = (np.min(bands[:,B1]) - np.max(bands[:,B1-1]) )/ np.max(bands[:,B1-1]) 
    # optvec[k]  =  np.max(np.diff(bands[:, 4]))
    # optvec2[k]  = np.max(np.diff(bands[:, 4]))
    return opt1 + opt2


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
    while epoch<500 and not done:
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
        y_pred = model(x_train)
        mse = loss_fn(y_pred,y_train)
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
    
# Define the model, loss function and optimizer
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
    
    

''' ======================================================================
                  
                  LOAD DATA AND FIT REGRESSOR
                  
====================================================================== '''

savefigs = False
if not savefigs:
    plt.style.use('dark_background')
else:
    plt.style.use('default')
#################################################
#                   PARAMETERS
#################################################
MshRs       = float(25)
Xdim        = 3
rf_level    = 6
Nquads      = 8
SamplrDim   = Xdim
Nsamp       = int(1e6)
LHS_Seed    = 4
path        = "..//data//TrainingData"
figpath     = ('..//figures//WeeklyMeetings//42723//Quads_'    + str(Nquads) + ' Xdim_' 
                                                + str(Xdim)    +  ' MshRs_'+ str(MshRs)
                                                + ' rfnmt_' +  str(rf_level) )
isExist = os.path.exists(figpath)
if not isExist: os.makedirs(figpath)

#################################################
#        Load in the data
#################################################
if not 'dvecs' in locals() and not 'Xdimcurrent' in locals():
    Xdimcurrent = Xdim
    dvecs,splines,BGdata,Bands, Rads, datapath = load_data(MshRs,         Xdim,   rf_level,   Nquads,    
                                                SamplrDim,     Nsamp,  LHS_Seed,   path)
elif Xdimcurrent != Xdim:
    dvecs,splines,BGdata,Bands, Rads, datapath = load_data(MshRs,         Xdim,   rf_level,   Nquads,    
                                                SamplrDim,     Nsamp,  LHS_Seed,   path)
    
#################################################
#           COMPUTE OBJECTIVE FUNCTION
#################################################
BG_op       = 6
optvec      = np.zeros(len(dvecs))
xvec        = np.zeros((len(dvecs),SamplrDim))
for k in range(len(BGdata)):
    '''
    Computing some parameters from the band diagrams
    '''
    optvec[k]   = objfun(Bands[k], BG_op = BG_op)
    xvec[k,0]   = dvecs[k][0]
    xvec[k,1]   = dvecs[k][1]
    if len(xvec[0,:]) > 2:
        xvec[k,2]   = dvecs[k][2]

def normalize(targ, targ_raw):
    return (targ - np.min(targ_raw)) /(np.max(targ_raw)-np.min(targ_raw))

#################################################
#        DEFINE FEATURES AND TARGETS
#################################################
features    = np.array(dvecs);
targ_raw    = optvec
targets     = normalize(targ_raw, targ_raw)
len_feat    = features.shape[1] 
if np.isclose(np.sum(features[:,len_feat-1]),0): features = np.delete(features,len_feat-1,1)
    
#################################################
#        FIT A SIMPLE REGRESSOR
#################################################  
regressor       = SVR(kernel = 'rbf')
regressor.fit(features,targets)
plot_regressor(regressor, xvec, features,savefigs)

#################################################
#        Plot some splines
#################################################
plt = plot_splines(splines,targets, nrow = 6, ncol = 6,is_rand = True)
if savefigs: plt.savefig(figpath+'//candidates.pdf')
plt.show()

#################################################
#       Rise/target plots for SVR
#################################################
predictions = regressor.predict(features)
plt = riseplot(targets,predictions)
if savefigs: plt.savefig(figpath+'//RisePlotSVR.pdf', bbox_inches = 'tight')
plt.show()
plt = targetplot(targets,predictions)
if savefigs: plt.savefig(figpath+'//TargetPlotSVR.pdf', bbox_inches = 'tight')
plt.show()

#################################################
#       Plot the ad-hoc optimal
#################################################
plt = plotbands(np.array(Bands[targets.argsort()[-1]]))
plt.show()

plt.plot(splines[targets.argsort()[-1]][:,0],splines[targets.argsort()[-1]][:,1] );
plt.grid()
plt.show()


# %%

#%%
''' 
=================================================================================================
                
                                FIT NEURAL NETWORK MODEL
                
=================================================================================================
'''

#################################################
#               SELECT DATA SETS
#################################################
scaler  = MinMaxScaler()
x       = features
y       = targets
dimx    = x.shape[1]
dimy    = 1 if len(y.shape) == 1 else y.shape

#################################################
#       DEFINE TRAINING/TESTING DAT
#################################################
BATCH_SIZE  = 32           # int(len(features)/8
BATCH_SIZE  = int(BATCH_SIZE)
fair        = True
#################################################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
x_train     = torch.Tensor(x_train).float()
y_train     = torch.Tensor(y_train).float()
if not fair:
    x_train         = torch.Tensor(x).float()
    y_train         = torch.Tensor(y).float()
x_test              = torch.Tensor(x_test).float().to(device)
y_test              = torch.Tensor(y_test).float().to(device)
dataset_train       = TensorDataset(x_train, y_train)
dataloader_train    = DataLoader(dataset_train,batch_size=BATCH_SIZE, shuffle=True)
dataset_test        = TensorDataset(x_test, y_test)
dataloader_test     = DataLoader(dataset_test,batch_size=BATCH_SIZE, shuffle=True)

#################################################
#       DEFINE CUSTOM LOSS FCN
#################################################
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
    def forward(self, output, target):
        # target = torch.LongTensor(target)
        criterion   = nn.MSELoss()
        loss        = criterion(output, target)
        mask        = target > .75
        # high_cost = (loss * mask.float()).mean()
        
        loss2 = criterion(output*mask, target*mask)
        return loss + loss2*3

#################################################
#       MODEL INITIALIZATION
#################################################
model       = Net(dimx, dimy, First = 128 , deep  = 64, Ndeep = 4, isrelu = True) .to(device)
loss_fn     = CustomLoss()
optimizer   = torch.optim.Adam(model.parameters())
lr          = 0.0005 
optimizer   = torch.optim.Adam(model.parameters(), lr = lr)
scheduler1  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
Nparams     = sum(p.numel() for p in model.parameters() if p.requires_grad)
#################################################
#       OPTIMIZATION LOOP
#################################################
model, history, historyt =   trainingloop(model,optimizer,x_train,y_train,x_test,y_test,dataloader_train, 
                                          fair = fair,  patience = 25, min_delta = 1e-6, restore_best_weights = True)
#%%

''' =================================================================================================                

                                   EVALUATE NN FIT
            
================================================================================================= '''
x_train = torch.Tensor(x).float()
y_train = torch.Tensor(y).float()
#################################################
# Predict over trianing data and get MSE
#################################################
from sklearn import metrics
pred = model(x_train)
score = metrics.mean_squared_error(pred.cpu().detach(),y_train.cpu())
print("Final score (MSE): {}".format(score))

#################################################
# Plot the target vs predicted values
#################################################
x1 = y_train.cpu().detach()
x2 = pred.flatten().cpu().detach()
args = np.argsort(x1)

#################################################
#       Rise/target plots for NN
#################################################
plt = learningplot(historyt,history)
if savefigs: plt.savefig(figpath+'//Error.pdf', bbox_inches = 'tight')
plt.show()
predictions = model(x_train)
plt = riseplot(targets.flatten(),predictions.detach().flatten())
if savefigs: plt.savefig(figpath+'//RisePlotNN.pdf', bbox_inches = 'tight')
plt.show()
plt = targetplot(targets.flatten(),predictions.detach().flatten())
if savefigs: plt.savefig(figpath+'//TargetPlotNN.pdf', bbox_inches = 'tight')
plt.show()


#################################################
# Set grid to view NN model over
#################################################
if features.shape[1] == 3:
    Ntest  = 15
    xtest  = np.linspace(0,1,Ntest)
    ytest  = np.linspace(0,1,Ntest)
    ztest  = np.linspace(0,1,Ntest)
    Xtest,Ytest,Ztest = np.meshgrid(xtest,ytest,ztest)
    Xtst    = Xtest.reshape(Ntest**3,1)
    Ytst    = Ytest.reshape(Ntest**3,1)
    Ztst    = Ztest.reshape(Ntest**3,1)
    testdat = np.hstack( ( Xtst,Ytst,Ztst ))
    origshape = Xtest.shape
elif features.shape[1] == 2:
    Ntest  = 100
    xtest  = np.linspace(0,1,Ntest)
    ytest  = np.linspace(0,1,Ntest)
    Xtest,Ytest = np.meshgrid(xtest,ytest)
    Xtst    = Xtest.reshape(Ntest**2,1)
    Ytst    = Ytest.reshape(Ntest**2,1)
    testdat = np.hstack( ( Xtst,Ytst))
    origshape = Xtest.shape
    
#################################################
# Predict NN on Grid and Plot
#################################################
if x_train.shape[1] == 3:
    pred_grid = model(torch.Tensor(testdat))
    fig     = plt.figure(figsize = (5,4))
    ax      = plt.axes(projection='3d')
    s1      = ax.scatter(x_train[:,0],x_train[:,1],x_train[:,2], c = y_train, cmap =  'twilight_shifted', vmin = torch.min(y_train).numpy(), vmax = torch.max(y_train).numpy() )
    ax.view_init(elev=20, azim=80, roll=0)      
    ax.set_title('Training Data Samples (FEM)')
    plt.colorbar(s1)
    if savefigs: plt.savefig(figpath+'//Targets.pdf', bbox_inches = 'tight')
    plt.show()
    fig     = plt.figure(figsize = (5,4))
    ax      = plt.axes(projection='3d')
    s1      = ax.scatter(x_train[:,0],x_train[:,1],x_train[:,2], c = pred.flatten().cpu().detach(), cmap =  'twilight_shifted', vmin = torch.min(y_train).numpy(), vmax = torch.max(y_train).numpy() )
    ax.view_init(elev=20, azim=80, roll=0)       
    ax.set_title('Neural Network on Samples')
    plt.colorbar(s1)
    if savefigs: plt.savefig(figpath+'//FeatPred_NN.pdf', bbox_inches = 'tight')
    plt.show()
    fig     =   plt.figure(figsize = (5,4))
    ax      =   plt.axes(projection='3d')
    s0      =   ax.scatter(x_train[:,0],x_train[:,1],x_train[:,2], c = pred.flatten().cpu().detach(), cmap =  'twilight_shifted', vmin = torch.min(y_train).numpy(), vmax = torch.max(y_train).numpy() )
    s1      =   ax.scatter(testdat[:,0],testdat[:,1],testdat[:,2],alpha = .1, c = pred_grid.flatten().cpu().detach(),  cmap = 'twilight_shifted')
    ax.view_init(elev=20, azim=80, roll=0)    
    plt.colorbar(s1)
    ax.set_title('Neural Network Over Grid')
    if savefigs: plt.savefig(figpath+'//GridPred_NN.pdf', bbox_inches = 'tight')
    plt.show()

elif x_train.shape[1] == 2:
    pred_grid = model(torch.Tensor(testdat))
    pltd = pred_grid.flatten().cpu().detach().numpy()
    cm ='twilight_shifted'
    cm = plt.get_cmap('twilight_shifted', 500)   # 11 discrete colors
    plt.figure(figsize =(5,5))
    plt.imshow((pltd.reshape(origshape)),cmap = cm, origin='lower', extent =(0,1,0,1) ,label='Surrogate NN')
    plt.scatter(x_train[:,0],x_train[:,1], c = (y_train), s =20, cmap = cm, edgecolors='k', label = 'FEM Result')
    plt.colorbar()
    plt.title('NN Vs Targets')
    plt.xlabel('Design Feature 1')
    plt.ylabel('Design Feature 2')
    if savefigs: plt.savefig(figpath+'//ContourNN.pdf', bbox_inches = 'tight')
    plt.show()

#%%

''' 
=================================================================================================                

                                   OPTIMIZATION
                
=================================================================================================
'''
from scipy.optimize import differential_evolution
from scipy.optimize import minimize

def optfun(x):
    feval  = model(torch.Tensor(x)).detach()
    return -(feval)
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})

#################################################
#       PLOT THE OPTIMIZATION RESULTS
#################################################
if features.shape[1] ==3:
    bnds = ((0.1, .95), (0.1, .95), (0.1, .95))
    fig     = plt.figure()
    ax      = plt.axes(projection='3d')
    cdata = (pred_grid.flatten().cpu().detach())
    s1=ax.scatter(testdat[:,0],testdat[:,1],testdat[:,2],alpha = .025, c = cdata, cmap = 'twilight_shifted')
    ax.view_init(elev=20, azim=80, roll=0)    
    plt.colorbar(s1)
    ax.set_title('Neural Network Over Grid')
    
    fstar = 1e5
    xoptall = []
    foptall = []
    for k in range(10):
        x0 = np.random.rand(3)
        # res = minimize(optfun, x0, method='Nelder-Mead', tol=1e-14, bounds=bnds )
        # xopt = res.x
        outp = differential_evolution(optfun, bounds = bnds)
        xopt = outp.x
        if outp.fun < fstar:
            out = outp
            fstart = out.fun
        ax.scatter(x0[0],x0[1],x0[2],color=(1,.5,.5),marker='o', s= 50)
        ax.scatter(xopt[0],xopt[1],xopt[2],color=(.2,.8,.2),marker='o', s= 50)
    plt.show()
    
if features.shape[1] ==2:
    bnds = ((0.1, 1), (0.1, 1))
    plt.contourf(Xtest,Ytest, (pltd.reshape(origshape)),50,cmap = cm,label='Surrogate NN')
    # plt.scatter(x_train[:,0],x_train[:,1], c = (y_train), cmap = cm, edgecolors='k', label = 'FEM Result')
    plt.colorbar()
    plt.legend()
    plt.title('Neural Network Over Grid')
    for k in range(10):
        x0 = (np.random.rand(2) )*.9
        # res = minimize(optfun, x0, (0.1, .95), (0.1, .95), (0.1, .95),method='Nelder-Mead', tol=1e-14, bounds=bnds )
        # xopt = res.x
        out = differential_evolution(optfun, bounds = bnds)
        xopt = out.x
        plt.scatter(x0[0],x0[1],color=(.5,.5,.5), marker='o', s= 20)
        plt.scatter(xopt[0],xopt[1],color=(.2,.8,.2), marker='o', s= 20,alpha = .35)
    plt.show()
    
if features.shape[1] == 6:
    bnds = ( (0.1, .95), (0.1, .95), (0.1, .95),
            (0.1, .95), (0.1, .95), (0.1, .95))
    x0 = (np.random.rand(6) )*.9
  #  for k in range(5):
    out = differential_evolution(optfun, bounds = bnds)
    print(out.x)
    # print(out.fun)
    x0 = np.random.rand(6)
    res = minimize(optfun, x0, method='Nelder-Mead', tol=1e-14, bounds=bnds )
    print(res.x)    
    print(res.fun)
    
    
    
#%% 

''' 
=================================================================================================                

                            EVALUATE THE 'OPTIMAL' POINT WITH FEM
                
=================================================================================================
'''
# Nquads                        =   4 
a_len_eval                      =   .1
r_eval                          =   np.array(out.x)*a_len_eval#np.array([[0.94929848 ,0.94984762, 0.69426451, 0.94035399 ,0.86640294, 0.95    ]])*a_len*.95; r= r.reshape(6,)
c                               =   [30]          # if void inclusion  (if iscut)
rho                             =   [1.2]         # if void inclusion  (if iscut)
da_eval                         =   a_len_eval/13
offset_eval                     =   0
refinement_level_eval           =   4
refinement_dist_eval            =   a_len_eval/15
Nquads_eval                     =   Nquads
np1                             =   20
np2                             =   20
np3                             =   20
nvec                            =   20
fspace                          =   'CG'
meshalg_eval                    =   6

gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len_eval,da_eval,r_eval,Nquads_eval,offset_eval,
                                                meshalg_eval, refinement_level_eval,refinement_dist_eval,
                                                isrefined = True, cut = True)
mesh_comm   = MPI.COMM_WORLD
model_rank  = 0
mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
plotter     = plotmesh(mesh,fspace,ct)
# plotter.screenshot('OptimalMesh.jpeg',window_size=[1400,1400])
plotter.show()
evals_disp_eval, evec_all_eval = solve_bands(np1, np2, np3,nvec, a_len_eval, c, rho, fspace, mesh, ct)


a_len = 0.1
#################################################################
#               Evaluate the predicted output                   #
#################################################################
bgnrm_eval, gapwidths_eval, gaps_eval, lowbounds_eval, highbounds_eval = getbands(np.array(evals_disp_eval))
fstar_FEM =   normalize(objfun(np.array(evals_disp_eval), BG_op = BG_op ),targ_raw)
epsilon = np.abs(fstar_FEM- np.abs(out.fun))/( fstar_FEM)
print( ' =========== EVALUATION =============')
print(f"AD-HOC OPTIMAL: {features[targets.argsort()[-1]]}")
print(f"AD-HOC FSTAR: {np.max(targets)}")
print(f"NEURAL NET OPTIMAL: {out.x}")
print(f"NEURAL NET FSTAR: {-out.fun}")
print(f"MODEL EVALUATION: {fstar_FEM}")
print(f"EPSILON: {epsilon}")

###########################################################
# Get spline fit from the 'optimized' mesh                #
###########################################################
node_interior = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)[1]
x_int = node_interior[0::3]
y_int = node_interior[1::3]
x_int = np.concatenate([x_int,[x_int[0]]])
y_int = np.concatenate([y_int,[y_int[0]]])
xi = x_int - a_len/2
yi = y_int - a_len/2

#################################################################
#              Ad-hoc vs NN Optimal                             #
#################################################################
plt.figure(figsize=(5,5))
plt.plot(splines[targets.argsort()[-1]][:,0],splines[targets.argsort()[-1]][:,1], label = 'ad-hoc' ); 
plt.plot(xi,yi,label='NN Optimal' ); 
plt.legend()
plt.axis('equal')
plt.title('Ad-hoc Optimal Design')
plt.show()

plt.plot(Bands[targets.argsort()[-1]]/np.max(Bands[targets.argsort()[-1]]),'b')
plt.plot(evals_disp_eval /np.max(evals_disp_eval),'r--' )
plt.legend()
plt.show()
#%% Getting enrichment samples
# xbest = out.x


''' 
=================================================================================================                

                                  ENRICHMENT SAMPLING
                
=================================================================================================
'''
        
######################################################################
#                Generate New Samples                                #
######################################################################
savedata            = 1 
overwrite           = True
a_len_enrich        = a_len_eval
Enrichment_Index    = 2
epsilon_all         = []
fstar_all           = []
for Enrichment_Index in range(3):
    # Center value of new distribution to sample
    X_c = out.x
    ki = 0
    ######################################################################
    #                       Sampler Inputs                               #
    ######################################################################
    if    np.max(targets)  < fstar_FEM: mean = out.x
    else: mean = dvecs[targets.argsort()[-1]][0:-1]
    cov = np.eye(len(mean))/100
    if len(mean) ==6:
        xs1, xs2, xs3,xs4,xs5,xs6 = np.random.multivariate_normal(mean, cov, 100).T
        xsall = np.vstack((xs1,xs2,xs3,xs4,xs5,xs6))
        xsall[xsall<.1]=.1
        xsall[xsall>.95]=.95
    elif len(mean) ==3:
        xs1, xs2, xs3  = np.random.multivariate_normal(mean, cov, 100).T
        xsall = np.vstack((xs1,xs2,xs3))
        xsall[xsall<.1]=.1
        xsall[xsall>.95]=.95
        for k in range(xsall.shape[0]):
            plt.hist(xsall[k,:],alpha=.3, edgecolor = 'r')
    
    if Enrichment_Index == 0:
        features_enriched = features
        targets_enriched = targets
    features_enriched   = np.vstack( (features_enriched,     r_eval))
    targets_enriched    = np.concatenate((targets_enriched,  fstar_FEM.reshape(1,)))
    
    for sidx in range(10):
        # datapath_enrich = datapath + '//EnrichementData//BGop_'+str(BG_op) +'//enrich_loop' + str(Enrichment_Index)
        datapath_enrich = datapath[0:10] + '//EnrichementData' +datapath[22::]+  '//BGop_'+str(BG_op) +'//enrich_loop' + str(Enrichment_Index)
        if savedata:
            if not os.path.exists(datapath_enrich):
                os.makedirs(datapath_enrich+'//dispersiondata')
                os.makedirs(datapath_enrich+'//meshdata')
                os.makedirs(datapath_enrich+'//Splinecurves')
                os.makedirs(datapath_enrich+'//BGdata')
                os.makedirs(datapath_enrich+'//Dvecdata')
                os.makedirs(datapath_enrich+'//Splinepts')
        # Skip iteration if the file was already generated
        if os.path.isfile(datapath_enrich+'//dispersiondata//'+str(sidx)+'.csv'):
            if not overwrite:
                print('Skipping iteration '+str(sidx)+' because it already exists')
                continue
        # =============================
        # Define domain
        # =============================
        offset              = 0
        r_enrich            = xsall[:,sidx]*a_len
        design_vec_enrich   = np.concatenate( (r_enrich/a_len, [offset] ))
        gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len_eval,da_eval,r_enrich,Nquads_eval,offset_eval,
                                                meshalg_eval, refinement_level_eval,refinement_dist_eval,
                                                isrefined = True, cut = True)
        mesh_comm = MPI.COMM_WORLD
        model_rank = 0
        try:    mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
        except: mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
        # =============================
        # Solve the dispersion problem
        # =============================
        evals_disp_enrich, evec_all_enrich  = solve_bands(np1, np2, np3,nvec, a_len_enrich, c, rho, fspace, mesh, ct)
        bgnrm_enrich, gapwidths_enrich, gaps_enrich, lowbounds_enrich, highbounds_enrich = getbands(np.array(evals_disp_enrich))
        BGdata_enrich = gapwidths_enrich
        
        # ===============================
        #    Get geometry information   #
        # ===============================
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
        SplineDat_enrich   = np.hstack( (xi.reshape(lxi,1), yi.reshape(lxi,1) ))  
        SplinePtDat_enrich = np.hstack( (xptnd.reshape(lxp,1), yptnd.reshape(lxp,1) ))  
        disp_info_enrich   = np.array(evals_disp_enrich)
        ngaps       = len(gapwidths_enrich)
        if ngaps    == 0:   BGdata_enrich = np.zeros(4)
        else:               BGdata_enrich = np.hstack((gapwidths_enrich.reshape(ngaps,1),lowbounds_enrich.reshape(ngaps,1),
                            highbounds_enrich.reshape(ngaps,1),bgnrm_enrich.reshape(ngaps,1)))
        ######################################################################
        #                   Save new samples                                 #
        ######################################################################
        np.savetxt((datapath_enrich+'//Dvecdata//'      +str(sidx)+'.csv'),     design_vec_enrich, delimiter=",")
        np.savetxt((datapath_enrich+'//BGdata//'        +str(sidx)+'.csv'),     BGdata_enrich, delimiter=",")
        np.savetxt((datapath_enrich+'//dispersiondata//'+str(sidx)+'.csv'),     disp_info_enrich, delimiter=",")
        np.savetxt((datapath_enrich+'//Splinecurves//'  +str(sidx)+'.csv'),     SplineDat_enrich, delimiter=",")
        np.savetxt((datapath_enrich+'//Splinepts//'     +str(sidx)+'.csv'),     SplinePtDat_enrich, delimiter=",")
        gmsh.write((datapath_enrich+'//meshdata//'      +str(sidx)+'.msh'))
        
        target_enrichment   = normalize(objfun(np.array(evals_disp_enrich), BG_op = BG_op), targ_raw)
        features_enrichment = r_enrich/a_len
        features_enriched   = np.vstack( (features_enriched, features_enrichment.reshape(1,len(r_enrich))))
        targets_enriched    = np.concatenate((targets_enriched,target_enrichment.reshape(1,))) 

    
    #################################################
    #          Retrain the Neural Network           #
    #################################################
    BATCH_SIZE  = 32
    fair    = True
    x       = features_enriched
    y       = targets_enriched
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)
    x_train     = torch.Tensor(x_train).float()
    y_train     = torch.Tensor(y_train).float()
    if not fair:
        x_train         = torch.Tensor(x).float()
        y_train         = torch.Tensor(y).float()
    x_test              = torch.Tensor(x_test).float().to(device)
    y_test              = torch.Tensor(y_test).float().to(device)
    dataset_train       = TensorDataset(x_train, y_train)
    dataloader_train    = DataLoader(dataset_train,batch_size=BATCH_SIZE, shuffle=True)
    dataset_test        = TensorDataset(x_test, y_test)
    dataloader_test     = DataLoader(dataset_test,batch_size=BATCH_SIZE, shuffle=True)
    model       = Net(dimx, dimy, First = 128 , deep  = 64, Ndeep = 4, isrelu = True).to(device)
    loss_fn     = CustomLoss()
    optimizer   = torch.optim.Adam(model.parameters())
    lr          = 0.0005 
    optimizer   = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler1  = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    Nparams     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model, history, historyt =   trainingloop(model,optimizer,x_train,y_train,x_test,y_test,dataloader_train, 
                                fair = fair,  patience = 25, min_delta = 1e-6, restore_best_weights = True)

    #################################################
    # Predict over trianing data and get MSE        #
    #################################################
    x_train     = torch.Tensor(x).float()
    y_train     = torch.Tensor(y).float()
    pred = model(x_train)
    score = metrics.mean_squared_error(pred.cpu().detach(),y_train.cpu())
    print("Final score (MSE): {}".format(score))
    #################################################
    # Plot the target vs predicted values           #
    #################################################
    x1 = y_train.cpu().detach()
    x2 = pred.flatten().cpu().detach()
    args = np.argsort(x1)
    predictions = model(x_train)
    #################################################
    #       Rise/target plots for NN                #
    #################################################
    plt = learningplot(historyt,history)
    if savefigs: plt.savefig(figpath+'//Error.pdf', bbox_inches = 'tight')
    plt.show()
    plt = riseplot(targets_enriched.flatten(),predictions.detach().flatten())
    if savefigs: plt.savefig(figpath+'//RisePlotNN.pdf', bbox_inches = 'tight')
    plt.show()
    plt = targetplot(targets_enriched.flatten(),predictions.detach().flatten())
    if savefigs: plt.savefig(figpath+'//TargetPlotNN.pdf', bbox_inches = 'tight')
    plt.show()
    #################################################
    #       Optimze the new results                 #
    #################################################
    if features.shape[1] ==3:
        bnds = ((0.1, .95), (0.1, .95), (0.1, .95))
        fstar = 1e5
        xoptall, foptall = [], []
        for k in range(10):
            x0 = np.random.rand(3)
            outp = differential_evolution(optfun, bounds = bnds)
            xopt = outp.x
            if outp.fun < fstar:
                out = outp
                fstart = out.fun
    if features.shape[1] ==2:
        bnds = ((0.1, 1), (0.1, 1))
        for k in range(10):
            x0 = (np.random.rand(2) )*.9
            out = differential_evolution(optfun, bounds = bnds)
            xopt = out.x
    if features.shape[1] == 6:
        bnds = ( (0.1, .95), (0.1, .95), (0.1, .95),
                (0.1, .95), (0.1, .95), (0.1, .95))
        x0 = (np.random.rand(6) )*.95
        out = differential_evolution(optfun, bounds = bnds)
        x0 = np.random.rand(6)
        res = minimize(optfun, x0, method='Nelder-Mead', tol=1e-14, bounds=bnds )
    
    #################################################
    #      Evaluate the optimal sol.                #
    #################################################
    r_eval                          =   np.array(out.x)*a_len_eval
    gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len_eval,da_eval,r_eval,Nquads_eval,offset_eval,
                                                    meshalg_eval, refinement_level_eval,refinement_dist_eval,
                                                    isrefined = True, cut = True)
    mesh_comm   = MPI.COMM_WORLD
    model_rank  = 0
    mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
    plotter     = plotmesh(mesh,fspace,ct)
    # plotter.screenshot('OptimalMesh.jpeg',window_size=[1400,1400])
    plotter.show()
    evals_disp_eval, evec_all_eval = solve_bands(np1, np2, np3,nvec, a_len_eval, c, rho, fspace, mesh, ct)
    
    a_len = 0.1
    #################################################################
    #               Evaluate the predicted output                   #
    #################################################################
    bgnrm_eval, gapwidths_eval, gaps_eval, lowbounds_eval, highbounds_eval = getbands(np.array(evals_disp_eval))
    fstar_FEM =   normalize(objfun(np.array(evals_disp_eval), BG_op = BG_op ),targ_raw)
    epsilon = np.abs(fstar_FEM- np.abs(out.fun))/( fstar_FEM)
    print( ' =========== EVALUATION =============')
    print(f"AD-HOC OPTIMAL: {features[targets.argsort()[-1]]}")
    print(f"AD-HOC FSTAR: {np.max(targets)}")
    print(f"NEURAL NET OPTIMAL: {out.x}")
    print(f"NEURAL NET FSTAR: {-out.fun}")
    print(f"MODEL EVALUATION: {fstar_FEM}")
    print(f"EPSILON: {epsilon}")
    epsilon_all.append(epsilon)
    fstar_all.append(fstar_FEM)
