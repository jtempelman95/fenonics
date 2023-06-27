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
#%%


''' 
======================================================================
                
                  LOAD DATA AND FIT REGRESSOR
                
======================================================================
'''

#################################################
#               PARAMETERS
#################################################
MshRs       = float(25)
Xdim        = int(3)
refinement_level = 6
SamplrDim   = 3
Nsamp       = int(1e6)
LHS_Seed    = 4
Nquads      = 8


#################################################
#               LOAD DATA LOOP
#################################################
datapath = ('data//TrainingData//SamplrSeed '  + str(LHS_Seed) +' SamplrDim '+  str(SamplrDim)   +' SamplrNgen '+  str(Nsamp)   
                                                + '//Quads_' + str(Nquads) + ' Xdim_' 
                                                + str(Xdim)    +  ' MshRs_'+ str(MshRs)
                                                + ' rfnmt_' +  str(refinement_level) )
os.listdir(datapath)
dirs = os.listdir(datapath)
files = os.listdir((datapath+'//'+dirs[0]))
nfile = len(files)

# if not np.any(dvecs):
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
#################################################
#           COMPUTE OBJECTIVE FUNCTION
#################################################
optvec  = np.zeros(nfile)
optvec2 =  np.zeros(nfile)
xvec    = np.zeros((nfile,SamplrDim))
for k in range(len(BGdata)):
    bands = Bands[k]
    bgnrm, gapwidths, gaps, lowbounds, highbounds = getbands(bands)
    optfnc = highbounds*bgnrm
    if not np.any(optfnc):
        optfnc = 0
    else:
        # optfnc = optfnc[0]
        optfnc =  optfnc[-1] + optfnc[0] 
    
    
        
    optvec[k]   = optfnc
    # optvec2[k]  = np.max(np.abs(np.diff(bands[:,0])))
    optvec2[k]  = np.max(np.diff(bands.reshape(60*20,)))
    xvec[k,0]   = dvecs[k][0]
    xvec[k,1]   = dvecs[k][1]
    if len(xvec[0,:]) > 2:
        xvec[k,2]   = dvecs[k][2]
    


#################################################
#        DEFINE FEATURES AND TARGETS
#################################################
features    = np.array(dvecs);
targets     = (optvec2/np.max(optvec2) + optvec/np.max(optvec))
# targets   = optvec2
len_feat = features.shape[1] 
if np.isclose(np.sum(features[:,len_feat-1]),0):
    features    = np.delete(features,len_feat-1,1)
    
#################################################
#        FIT A SIMPLE REGRESSOR
#################################################
plt.style.use('dark_background')
from skimage import measure
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
regressor       = SVR(kernel = 'rbf')
regressor.fit(features,targets)

#################################################
#        DEFINE TEST GRID
#################################################
if features.shape[1] >2:
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
elif features.shape[1] == 2:
    Ntest  = 100
    xtest  = np.linspace(0,.9,Ntest)
    ytest  = np.linspace(0,.9,Ntest)
    Xtest,Ytest = np.meshgrid(xtest,ytest)
    Xtst    = Xtest.reshape(Ntest**2,1)
    Ytst    = Ytest.reshape(Ntest**2,1)
    testdat = np.hstack( ( Xtst,Ytst))
    testpt  = y_pred = regressor.predict(testdat )

origshape = Xtest.shape

#################################################
#        PLOT THE REGRESSOR FUNCTION
#################################################
if features.shape[1] >2:
    
    fig     = plt.figure()
    ax      = plt.axes(projection='3d')
    ax.scatter(xvec[:,0],xvec[:,1],xvec[:,2], c = optvec, cmap = "jet")
    ax.view_init(elev=25, azim=-35, roll=0)    
    plt.show()
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    s1= ax.scatter(Xtst,Ytst,Ztst, c = testpt, cmap = "jet")
    plt.colorbar(s1)
    ax.view_init(elev=25, azim=-35, roll=0)  
    plt.show()  
    
elif features.shape[1] == 2:
    
    fig     = plt.figure()
    ax      = plt.axes()
    ax.scatter(xvec[:,0],xvec[:,1], c = optvec, cmap = "jet")
    plt.show()
    plt.imshow(testpt.reshape(origshape),cmap = 'twilight_shifted', origin='lower', extent =(0,1,0,1) )
    plt.scatter(xvec[:,0],xvec[:,1], c = targets, cmap = 'twilight_shifted', edgecolors='k')
    plt.show()

arg = np.argsort(targets)
for idx in range(4):
    maxarg=arg[-idx]
    plt.plot(splines[maxarg][:,0],splines[maxarg][:,1] ); 
    plt.axis('equal'); plt.axis('image')
    

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
  def __init__(self, patience=5, min_delta=1e-10, restore_best_weights=True):
    self.patience = patience
    self.min_delta = min_delta
    self.restore_best_weights = restore_best_weights
    self.best_model = None
    self.best_loss = None
    self.counter = 0
    self.status = ""
    
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
#               RELU MLP
#################################################
class Net(nn.Module):
    def __init__(self, in_count, out_count):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(in_count,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,264)
        self.fc4 = nn.Linear(264,264)
        self.do = nn.Dropout(.25)
        self.fc5 = nn.Linear(264,64)
        self.fc6 = nn.Linear(64,32)
        self.fcend = nn.Linear(32,out_count)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        for i in range(4):
            x = F.relu(self.fc4(x)) 
            # x = F.relu(self.do(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.fcend(x)

#################################################
#               TANH MLP
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
        for i in range(4):
            h_relu = self.middle(h_relu).tanh()
            # x = F.relu(x)
        phi = self.lasthiddenlayer(h_relu)
        return phi

#################################################
#      SELECT DATA SETS
#################################################
x = features
# x = np.array(Rads)
y = -(targets)

#################################################
#       DEFINE TRAINING/TESTING DAT
#################################################
BATCH_SIZE = 16
#################################################
x_train, x_test, y_train, y_test = train_test_split(    
    x, y, test_size=0.005, random_state=42)
# Numpy to Torch Tensor
# x_train = torch.Tensor(x_train).float()
# y_train = torch.Tensor(y_train).float()
x_train = torch.Tensor(x).float()
y_train = torch.Tensor(y).float()
x_test = torch.Tensor(x_test).float().to(device)
y_test = torch.Tensor(y_test).float().to(device)

dataset_train = TensorDataset(x_train, y_train)
dataloader_train = DataLoader(dataset_train,batch_size=BATCH_SIZE, shuffle=True)
dataset_test = TensorDataset(x_test, y_test)
dataloader_test = DataLoader(dataset_test,batch_size=BATCH_SIZE, shuffle=True)

#################################################
#       MODEL INITIALIZATION
#################################################
model       = Net(x.shape[1],1).to(device)
loss_fn     = nn.MSELoss()
optimizer   = torch.optim.Adam(model.parameters())

#################################################
#       OPTIMIZATION LOOP
#################################################
es = EarlyStopping()
epoch = 0
done = False
while epoch<100 and not done:
  epoch += 1
  steps = list(enumerate(dataloader_train))
  pbar = tqdm.tqdm(steps)
  model.train()
  for i, (x_batch, y_batch) in pbar:
    y_batch_pred = model(x_batch.to(device)).flatten()
    loss = loss_fn(y_batch_pred, y_batch.to(device))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss, current = loss.item(), (i + 1)* len(x_batch)
    if i == len(steps)-1:
      model.eval()
      pred = model(x_train).flatten()
      vloss = loss_fn(pred, y_train)
      if es(model,vloss): done = True
      pbar.set_description(f"Epoch: {epoch}, tloss: {loss}, vloss: {vloss:>7f}, EStop:[{es.status}]")
    else:
      pbar.set_description(f"Epoch: {epoch}, tloss {loss:}")
      
#%%


''' 
======================================================================
                
               EVALUATE NN FIT
                
======================================================================
'''

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
plt.plot(x1[args],'r-',label = 'Target')
plt.plot(x2[args],'.', color = (0,  .7, 0), label = 'MLP Pred.')
plt.legend()
plt.xlabel('Sort Idx')
plt.ylabel('Target Val')
plt.show()
plt.plot(x1[args],x1[args],'r-',label = 'Target')
plt.plot(x1[args],x2[args],'.', color = (0,.7,0), label = 'MLP Pred.')
plt.xlabel('Target Val')
plt.ylabel('Target Val')
plt.legend()
plt.title('Target Vs NN')

#################################################
# Set grid to view NN model over
#################################################
if features.shape[1] >2:
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
pred_grid = model(torch.Tensor(testdat))
if x_train.shape[1]>2:
    fig     = plt.figure()
    ax      = plt.axes(projection='3d')
    s1=ax.scatter(x_train[:,0],x_train[:,1],x_train[:,2], c = y_train, cmap = "jet", vmin = torch.min(y_train).numpy(), vmax = torch.max(y_train).numpy() )
    ax.view_init(elev=20, azim=80, roll=0)      
    ax.set_title('Training Data Samples (FEM)')
    plt.colorbar(s1)
    plt.show()

    fig     = plt.figure()
    ax      = plt.axes(projection='3d')
    s1=ax.scatter(x_train[:,0],x_train[:,1],x_train[:,2], c = pred.flatten().cpu().detach(), cmap = "jet", vmin = torch.min(y_train).numpy(), vmax = torch.max(y_train).numpy() )
    ax.view_init(elev=20, azim=80, roll=0)       
    ax.set_title('Neural Network on Samples')
    plt.colorbar(s1)
    plt.show()
    
    fig     = plt.figure()
    ax      = plt.axes(projection='3d')
    s1=ax.scatter(testdat[:,0],testdat[:,1],testdat[:,2],alpha = .1, c = pred_grid.flatten().cpu().detach(),  cmap = 'twilight_shifted')
    ax.view_init(elev=20, azim=80, roll=0)    
    # plt.colorbar(s1)
    ax.set_title('Neural Network Over Grid')
    # plt.savefig('OptFig1NN.pdf')
    plt.show()

else:
    fig     = plt.figure()
    ax      = plt.axes()
    pltd = pred_grid.flatten().cpu().detach().numpy()
    cm ='twilight_shifted'
    cm = plt.get_cmap('twilight_shifted', 40)   # 11 discrete colors
    # plt.imshow(np.log(pltd.reshape(origshape)),cmap = cm, origin='lower', extent =(0,1,0,1) ,label='Surrogate NN')
    
    plt.contourf(Xtest,Ytest, (pltd.reshape(origshape)),50,cmap = cm)
    plt.scatter(x_train[:,0],x_train[:,1], c = (y_train), s =20, cmap = cm, edgecolors=(.5,.5,.5), label = 'FEM Result')
    plt.colorbar()
    plt.title('MLP Vs Targets')
    plt.legend()
    plt.show()


#%%



''' Optimization '''

from scipy.optimize import minimize, rosen, rosen_der

def optfun(x):
    feval  = model(torch.Tensor(x)).detach()
    # print(feval)
    return (feval)



cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})



#################################################
# PLOT THE OPTIMIZATION RESULTS
#################################################
if features.shape[1] ==3:
    bnds = ((0.1, 1), (0.1, 1), (0.1, 1))
    fig     = plt.figure()
    ax      = plt.axes(projection='3d')
    cdata = (pred_grid.flatten().cpu().detach())
    s1=ax.scatter(testdat[:,0],testdat[:,1],testdat[:,2],alpha = .025, c = cdata, cmap = 'twilight_shifted')
    ax.view_init(elev=20, azim=80, roll=0)    
    plt.colorbar(s1)
    ax.set_title('Neural Network Over Grid')
    
    for k in range(100):
        x0 = np.random.rand(3)
        res = minimize(optfun, x0, method='Nelder-Mead', tol=1e-6, bounds=bnds )
        xopt = res.x
        ax.scatter(x0[0],x0[1],x0[2],color=(1,.5,.5),marker='o', s= 50)
        ax.scatter(xopt[0],xopt[1],xopt[2],color=(.2,.8,.2),marker='o', s= 50)
        print(optfun(xopt))
    plt.show()
    
if features.shape[1] ==2:
    bnds = ((0.1, .9), (0.1, .9))
    plt.contourf(Xtest,Ytest, (pltd.reshape(origshape)),50,cmap = cm,label='Surrogate NN')
    # plt.scatter(x_train[:,0],x_train[:,1], c = (y_train), cmap = cm, edgecolors='k', label = 'FEM Result')
    plt.colorbar()
    plt.legend()

    plt.title('Neural Network Over Grid')
    for k in range(150):
        x0 = (np.random.rand(2) )*.9
        res = minimize(optfun, x0, method='Nelder-Mead', tol=1e-14, bounds=bnds )
        xopt = res.x
        plt.scatter(x0[0],x0[1],color=(.5,.5,.5), marker='o', s= 20)
        plt.scatter(xopt[0],xopt[1],color=(.2,.8,.2), marker='o', s= 20,alpha = .35)
    plt.show()