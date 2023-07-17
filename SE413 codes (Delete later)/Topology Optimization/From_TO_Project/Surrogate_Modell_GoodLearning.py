"""
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                    POST PROCESSING SIMULATION RESULTS
                
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                
    Josh Tempelman
    Universrity of Illinois
    jrt7@illinois.edu
    
    Originated      MARCH 10, 2023
    Last Modified   MARCH 24, 2023
    
    ! ! Do not distribute ! !
    
    
    About:
        This program loads in dispersion data from FEM
        and fits surrogate models to the band diagram objective functions
    
        
"""

#%%
''' 
======================================================================
                
                  SUPPORT FUNCTIONS
                
======================================================================
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler  
from matplotlib import cm
from sklearn import preprocessing as pre

from PostProcess import *
# from GenerateData import *
from FEM_Functions import *
os.sys.path.append("./../") 




#################################################
#  SUPPORT FUNCTIONS           
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

    #################################################
    #           Load in the data
    #################################################
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
        
    return dvecs,splines,BGdata,Bands, Rads

def plot_regressor(regressor, xvec, features,savefigs):
    """
    Simple function for plotting the SVR over the data
    """
    #################################################
    #        DEFINE TEST GRID
    #################################################
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

    #################################################
    #        PLOT THE REGRESSOR FUNCTION
    #################################################
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

#%%


''' 
======================================================================
                
                  LOAD DATA AND FIT REGRESSOR
                
======================================================================
'''

savefigs = True
if not savefigs:
    plt.style.use('dark_background')
else:
    plt.style.use('default')
#################################################
#                   PARAMETERS
#################################################
MshRs       = float(25)
Xdim        = 6
refinement_level = 6
Nquads      = 4
SamplrDim   = Xdim
Nsamp       = int(1e6)
LHS_Seed    = 4
path = "..//data//TrainingData"
figpath = ('..//figures//WeeklyMeetings//42723//Quads_'    + str(Nquads) + ' Xdim_' 
                                                + str(Xdim)    +  ' MshRs_'+ str(MshRs)
                                                + ' rfnmt_' +  str(refinement_level) )
isExist = os.path.exists(figpath)
if not isExist: os.makedirs(figpath)

#################################################
#        Load in the data
#################################################
dvecs,splines,BGdata,Bands, Rads = load_data(MshRs,Xdim,refinement_level, Nquads, SamplrDim, Nsamp, LHS_Seed, path)
    
#################################################
#           COMPUTE OBJECTIVE FUNCTION
#################################################
optvec      = np.zeros(len(dvecs))
optvec2     = np.zeros(len(dvecs))
xvec        = np.zeros((len(dvecs),SamplrDim))
for k in range(len(BGdata)):
    '''
    Computing some parameters from the band diagrams
    '''
    bands = Bands[k]
    bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(bands)
    optfnc = []
    if not np.any(optfnc):
        optfnc = 0
    else:
        # optfnc = optfnc[0]
        optfnc =  optfnc[-1] + optfnc[0] 

    #########################
    # First obj fnc
    #########################
    optfnc = []
    gaploc = 1e3
    
    # ==============================
    # To check if in range
    # ==============================
    if any(lowbounds):
        for j,lb in enumerate(lowbounds):
            if lowbounds[j] < gaploc and highbounds[j]>gaploc:
                optfnc = bgnrm[j] # - np.abs( gaploc- (highbounds[j]+lowbounds[j])/2)
    if not isinstance(optfnc,float):
        if not any(optfnc):
            optfnc = 0
                        
    # ==============================
    # To check if in above/below
    # ==============================
    if any(lowbounds):
        if np.min(lowbounds)<gaploc:
            optfnc = np.max(bgnrm[lowbounds<gaploc])
        else: optfnc = 0
    else: optfnc = 0
    
    optvec[k]       = optfnc
    
    
    #################################################
    #              Second obj fnc
    #################################################
    optfnc = []
    gaploc = 4e3
    
    # ==============================
    # To check if in range
    # ==============================
    if any(lowbounds):
        for j,lb in enumerate(lowbounds):
            if lowbounds[j] < gaploc and highbounds[j]>gaploc:
                optfnc = gapwidths[j]  - np.abs( gaploc- (highbounds[j]+lowbounds[j])/2)
        if not isinstance(optfnc,float):
            if not any(optfnc):
                optfnc = 0
                
    # ==============================
    # To check if in above/below
    # ==============================
    if any(lowbounds):
        if np.max(highbounds)>gaploc:
            optfnc = np.max(bgnrm[highbounds>gaploc])
        else: optfnc = 0
    else: optfnc = 0
    optvec2[k]  = optfnc
    
    # optvec2[k]  = np.mean(np.abs(np.diff(bands[:,0])))
    # optvec2[k]  = np.max(np.diff(bands.reshape(60*20,)))
    xvec[k,0]   = dvecs[k][0]
    xvec[k,1]   = dvecs[k][1]
    if len(xvec[0,:]) > 2:
        xvec[k,2]   = dvecs[k][2]
    


#################################################
#        DEFINE FEATURES AND TARGETS
#################################################
features    = np.array(dvecs);
targets     = (optvec2 + optvec)
len_feat = features.shape[1] 
if np.isclose(np.sum(features[:,len_feat-1]),0):
    features    = np.delete(features,len_feat-1,1)
    
#################################################
#        FIT A SIMPLE REGRESSOR
#################################################  
regressor       = SVR(kernel = 'rbf')
regressor.fit(features,targets)
plot_regressor(regressor, xvec, features,savefigs)



#################################################
#        Plot some splines
#################################################
arg = np.argsort(targets)
cdata = targets
cdata = pre.MinMaxScaler().fit_transform(cdata.reshape(-1,1))
plt.figure(figsize=(5,5))
plt.subplots_adjust(left    =0.1,
                    bottom  =0.1,
                    right   =0.9,
                    top     =0.9,
                    wspace  =0.05,
                    hspace  =0.02)
for idx in range(36):
    # plt.subplot(5,5,idx)
    ax = plt.subplot(6,6, idx+1)
    idx = int(np.random.rand()*len(features))
    maxarg=arg[-idx]
    ax.plot(splines[maxarg][:,0],splines[maxarg][:,1] ,c=cm.twilight_shifted(cdata[maxarg]   ) ); 
    ax.axis('equal'); plt.axis('image')
    ax.set_xlim((-.05,.05))
    ax.set_ylim((-.05,.05))
    ax.set_xticks([])
    ax.set_yticks([])

if savefigs: plt.savefig(figpath+'//candidates.pdf')
plt.show()


x1 = targets
x2 = regressor.predict(features)
args = x1.argsort()
plt.figure(figsize =(3,3))
plt.plot(x2[args],'.', color = (0,  .7, 0), label = 'SVR Pred.')
plt.plot(x1[args],'r-',label = 'Target')
plt.legend()
plt.xlabel('Sort Idx')
plt.ylabel('Target Val')
plt.title('Rise Plot')
if savefigs: plt.savefig(figpath+'//RisePlotSVR.pdf', bbox_inches = 'tight')
plt.show()

plt.figure(figsize =(3,3))
plt.plot(x1[args],x2[args],'.', color = (0,.7,0), label = 'SVR Pred.')
plt.plot(x1[args],x1[args],'r-',label = 'Target')
plt.xlabel('Target Val')
plt.ylabel('Target Val')
plt.title('Target Plot')
plt.legend()
score = metrics.mean_squared_error(x1,x2)
plt.title("Final score (MSE): {}".format(np.round(score,4)))
if savefigs: plt.savefig(figpath+'//TargetPlotSVR.pdf', bbox_inches = 'tight')
plt.show()

#%%

best = arg[-1]
best_bands = Bands[best]
plt.plot(best_bands)

#%%

''' 
======================================================================
                
                FIT NEURAL NETWORK MODEL
                
======================================================================
'''
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

#################################################
#           BACKPROP STOPPING FUNCTION
#################################################
class EarlyStopping():
  def __init__(self, patience=50, min_delta=1e-6, restore_best_weights=True):
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
    
#################################################
#               RELU NN
#################################################
class Net(nn.Module):
    def __init__(self, in_count, out_count):
        super(Net,self).__init__()
        self.fc1    = nn.Linear(in_count,128,bias = True)
        self.fc2    = nn.Linear(128,64,bias = True)
        self.fc3    = nn.Linear(64,64,bias = True)
        self.fc4    = nn.Linear(64,64,bias = True)
        self.do     = nn.Dropout(.1)
        self.fcend  = nn.Linear(64,out_count)
        self.tanh   = nn.Tanh()
        self.relu   = nn.ReLU()
        self.seq = nn.Sequential(
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
        x = self.relu(self.fc3(x))
        for i in range(4):
            x = self.relu(self.fc4(x)) 
            # x = self.do(x)
        # x = F.relu(self.fc4(x)) 
        # x = F.relu(self.c1(x))
        # x = F.relu(self.fc5(x))
        # x = F.relu(self.fc6(x))
        return self.fcend(x)

#################################################
#               TANH NN
#################################################
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

#################################################
#      SELECT DATA SETS
#################################################
scaler = MinMaxScaler()
x = features
y = (targets)

#################################################
#       DEFINE TRAINING/TESTING DAT
#################################################
BATCH_SIZE = 32             # int(len(features)/8)
#################################################
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Numpy to Torch Tensor
x_train = torch.Tensor(x_train).float()
y_train = torch.Tensor(y_train).float()
# x_train = torch.Tensor(x).float()
# y_train = torch.Tensor(y).float()
x_test  = torch.Tensor(x_test).float().to(device)
y_test  = torch.Tensor(y_test).float().to(device)
dataset_train       = TensorDataset(x_train, y_train)
dataloader_train    = DataLoader(dataset_train,batch_size=BATCH_SIZE, shuffle=True)
dataset_test        = TensorDataset(x_test, y_test)
dataloader_test     = DataLoader(dataset_test,batch_size=BATCH_SIZE, shuffle=True)

#################################################
#       MODEL INITIALIZATION
#################################################
model       = Net(x.shape[1],1).to(device)
# model       = NetTanh(x.shape[1],64,128 ,1).to(device)
loss_fn     = nn.MSELoss()
optimizer   = torch.optim.Adam(model.parameters())
lr          = 0.00035
optimizer   = torch.optim.Adam(model.parameters(), lr = lr)
scheduler1  =  torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

#################################################
#       OPTIMIZATION LOOP
#################################################
es      = EarlyStopping()
epoch   = 0
done    = False
history = []
historyt = []
while epoch<500 and not done:
    epoch   += 1
    steps   = list(enumerate(dataloader_train))
    pbar    = tqdm.tqdm(steps)
    model.train()
    for i, (x_batch, y_batch) in pbar:
        
        y_batch_pred = model(x_batch.to(device)).flatten()
        loss = loss_fn(y_batch_pred , y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), (i + 1)* len(x_batch)
        
        if i == len(steps)-1:
            model.eval()
            pred = model(x_test).flatten()
            vloss = loss_fn(pred, y_test)
            history.append(float(vloss))
            
            pred = model(x_train).flatten()
            tloss = loss_fn(pred, y_train)
            historyt.append(float(tloss))
            if es(model,vloss): done = True
            pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, EStop:[{es.status}]")
        else:
            pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
    y_pred = model(x_train)
    mse = loss_fn(y_pred,y_train)
    
    # scheduler1.step()
    
#%%


''' 
======================================================================
                
               EVALUATE NN FIT
                
======================================================================
'''

x_train = torch.Tensor(x).float()
y_train = torch.Tensor(y).float()

from sklearn import metrics
#################################################
# Predict over trianing data and get MSE
#################################################
pred = model(x_train)
score = metrics.mean_squared_error(pred.cpu().detach(),y_train.cpu())
print("Final score (MSE): {}".format(score))

#################################################
# Plot the target vs predicted values
#################################################
x1 = y_train.cpu().detach()
x2 = pred.flatten().cpu().detach()
args = np.argsort(x1)
plt.figure(figsize =(3,3))
plt.plot(x2[args],'.', color = (0,  .7, 0), label = 'NN Pred.')
plt.plot(x1[args],'r-',label = 'Target')
plt.legend()
plt.xlabel('Sort Idx')
plt.ylabel('Target Val')
plt.title('Rise Plot')
if savefigs: plt.savefig(figpath+'//RisePlotNN.pdf', bbox_inches = 'tight')
plt.show()

plt.figure(figsize =(3,3))
plt.plot(historyt,'b-',label = 'Training')
plt.plot(history,'r-',label = 'Validation')
# plt.plot(x2[args],'.', color = (0,  .7, 0), label = 'NN Pred.')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('lr = ' + str(optimizer.param_groups[0]['initial_lr'])+ ' $n_b$ = '+str(BATCH_SIZE))
if savefigs: plt.savefig(figpath+'//Error.pdf', bbox_inches = 'tight')
plt.show()

plt.figure(figsize =(3,3))
plt.plot(x1[args],x2[args],'.', color = (0,.7,0), label = 'NN Pred.')
plt.plot(x1[args],x1[args],'r-',label = 'Target')
plt.xlabel('Target Val')
plt.ylabel('Target Val')
plt.legend()
sc = np.round(score,5)
plt.title("Final score (MSE): " + str(sc))
if savefigs: plt.savefig(figpath+'//TargetPlotNN.pdf', bbox_inches = 'tight')
plt.show()


#################################################
# Set grid to view NN model over
#################################################
if features.shape[1] == 3:
    Ntest  = 24
    xtest  = np.linspace(0,1,Ntest)
    ytest  = np.linspace(0,1,Ntest)
    ztest  = np.linspace(0,1,Ntest)
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
    
#################################################
# Predict NN on Grid and Plot
#################################################
if x_train.shape[1] == 3:
    pred_grid = model(torch.Tensor(testdat))
    fig     =plt.figure(figsize = (5,4))
    ax      = plt.axes(projection='3d')
    s1=ax.scatter(x_train[:,0],x_train[:,1],x_train[:,2], c = y_train, cmap =  'twilight_shifted', vmin = torch.min(y_train).numpy(), vmax = torch.max(y_train).numpy() )
    ax.view_init(elev=20, azim=80, roll=0)      
    ax.set_title('Training Data Samples (FEM)')
    plt.colorbar(s1)
    if savefigs: plt.savefig(figpath+'//Targets.pdf', bbox_inches = 'tight')
    plt.show()

    fig     = plt.figure(figsize = (5,4))
    ax      = plt.axes(projection='3d')
    s1=ax.scatter(x_train[:,0],x_train[:,1],x_train[:,2], c = pred.flatten().cpu().detach(), cmap =  'twilight_shifted', vmin = torch.min(y_train).numpy(), vmax = torch.max(y_train).numpy() )
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
    # plt.colorbar(s1)
    ax.set_title('Neural Network Over Grid')
    # if savefigs: plt.savefig('OptFig1NN.pdf')
    if savefigs: plt.savefig(figpath+'//GridPred_NN.pdf', bbox_inches = 'tight')
    plt.show()

elif x_train.shape[1] == 2:
    pred_grid = model(torch.Tensor(testdat))
    fig     = plt.figure()
    ax      = plt.axes()
    pltd = pred_grid.flatten().cpu().detach().numpy()
    cm ='twilight_shifted'
    cm = plt.get_cmap('twilight_shifted', 500)   # 11 discrete colors
    plt.figure(figsize =(5,5))
    # plt.contourf(Xtest,Ytest, (pltd.reshape(origshape)),50,cmap = cm)
    plt.imshow((pltd.reshape(origshape)),cmap = cm, origin='lower', extent =(0,1,0,1) ,label='Surrogate NN')
    plt.scatter(x_train[:,0],x_train[:,1], c = (y_train), s =20, cmap = cm, edgecolors='k', label = 'FEM Result')
    plt.colorbar()
    plt.title('NN Vs Targets')
    # plt.legend()
    plt.xlabel('Design Feature 1')
    plt.ylabel('Design Feature 2')
    if savefigs: plt.savefig(figpath+'//ContourNN.pdf', bbox_inches = 'tight')
    plt.show()



#%%


''' Optimization '''
from scipy.optimize import differential_evolution
from scipy.optimize import minimize

def optfun(x):
    feval  = model(torch.Tensor(x)).detach()
    # print(feval)
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
    
    for k in range(10):
        x0 = np.random.rand(3)
        # res = minimize(optfun, x0, method='Nelder-Mead', tol=1e-14, bounds=bnds )
        # xopt = res.x
        out = differential_evolution(optfun, bounds = bnds)
        xopt = out.x
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
    x0 = (np.random.rand(2) )*.9
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
    EVALUATE ACCURACY OF MODEL
'''
# Nquads                  =   4 
a_len                   =   .1
r                       =   np.array(out.x)*a_len #np.array([[0.94929848 ,0.94984762, 0.69426451, 0.94035399 ,0.86640294, 0.95    ]])*a_len*.95; r= r.reshape(6,)
c                       =   [30]          # if void inclusion  (if iscut)
rho                     =   [1.2]         # if void inclusion  (if iscut)
da                      =   a_len/25
offset                  =   0
meshalg                 =   7
refinement_level        =   6
refinement_dist         =   a_len/6
np1                     =   20
np2                     =   20
np3                     =   20
nvec                    =   20
fspace                  =   'CG'
meshalg                 =   6

gmsh.model, xpt, ypt    = get_mesh_SquareSpline(a_len ,da,r,Nquads,offset,meshalg,
                                                refinement_level,refinement_dist,
                                                isrefined = True, cut = True)
mesh_comm = MPI.COMM_WORLD
model_rank = 0
mesh, ct, _ = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)
evals_disp, evec_all = solve_bands(np1, np2, np3,nvec, a_len, c, rho, fspace, mesh,ct)
#%%
plt = plotbands(np.array(evals_disp))
# plt.saveas('OptimalBG.svg')
plt.show()
plotter = plotmesh(mesh,fspace,ct)
# plotter.screenshot('OptimalMesh.jpeg',window_size=[1400,1400])
plotter.show()


#%% Getting enrichment samples
xbest = out.x

######################################################################
#                       Sampler Inputs                               #
######################################################################
from scipy.stats import qmc
LHS_Seed    = 4
Nsamp       = int(1e6)
sampler     = qmc.LatinHypercube(d=6, seed= LHS_Seed)
sample      = sampler.random(n=Nsamp)
    